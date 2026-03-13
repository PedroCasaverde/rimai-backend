import os
import re as _re
import json
import uuid
import time
import boto3
import fitz  # pymupdf
import requests
import yt_dlp
from openai import OpenAI
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT

s3 = boto3.client('s3')

PROMPTS = {
    'resumen': (
        "Eres un experto en síntesis académica{disciplina_ctx}. A partir del siguiente contenido "
        "(transcripción de clase y/o apuntes), genera un resumen ejecutivo claro "
        "y estructurado en {paginas} páginas aproximadamente. Usa títulos y subtítulos. "
        "{analogias_ctx}"
        "Responde en español."
    ),
    'guia': (
        "Eres un experto pedagogo{disciplina_ctx}. A partir del siguiente contenido, genera una "
        "guía de estudio completa en {paginas} páginas aproximadamente con: conceptos clave, "
        "explicaciones, ejemplos y puntos importantes para recordar. "
        "{analogias_ctx}"
        "Responde en español."
    ),
    'banco_preguntas': (
        "Eres un experto evaluador{disciplina_ctx}. A partir del siguiente contenido, genera un banco de preguntas de "
        "estudio variadas (conceptuales, de aplicación y análisis) con sus "
        "respuestas completas. Genera suficientes preguntas para llenar aproximadamente "
        "{paginas} páginas. "
        "{analogias_ctx}"
        "Responde en español."
    ),
    'alerta_examen': (
        "Eres un asistente académico experto{disciplina_ctx} que analiza transcripciones de clases. "
        "Tu tarea es detectar TODAS las pistas, indirectas o menciones directas del profesor "
        "sobre qué temas, ejercicios, conceptos o tipos de preguntas podrían entrar en el examen. "
        "Busca frases como: 'esto es importante', 'esto entra en el examen', 'esto lo voy a evaluar', "
        "'presten atención a esto', 'esto es tipo examen', 'practiquen esto', 'no se olviden de...', "
        "'esto siempre cae', 'revisen bien esto', o cualquier otra indirecta similar. "
        "Para cada alerta encontrada, indica:\n"
        "1. La frase textual o paráfrasis de lo que dijo el profesor\n"
        "2. El tema o concepto al que se refiere\n"
        "3. Nivel de certeza (DIRECTA: lo dijo explícitamente / INDIRECTA: lo insinuó)\n"
        "4. Recomendación de estudio para ese punto\n\n"
        "Si no encuentras ninguna pista de examen, indícalo claramente. "
        "{analogias_ctx}"
        "Responde en español."
    ),
    'analogias': (
        "Eres un experto pedagogo{disciplina_ctx} que domina el arte de las analogías. "
        "A partir del siguiente contenido académico, explica los conceptos "
        "más importantes usando analogías {analogia_tema}. "
        "Cada analogía debe ser clara, memorable y directamente relacionada "
        "con el concepto que explica.\n\n"
        "Formato para cada concepto:\n"
        "**Concepto** → Analogía {analogia_tema} → Por qué funciona\n\n"
        "Genera suficientes analogías para cubrir aproximadamente {paginas} páginas. "
        "Responde en español."
    ),
}

ANALOGIAS_INSTRUCCION = (
    "Usa analogías claras y cotidianas para explicar los conceptos más complejos o abstractos, "
    "de modo que sean accesibles para alguien que se encuentra por primera vez con el tema. "
)


class ZoomDownloadError(Exception):
    """Error al descargar grabación de Zoom"""
    pass


def download_zoom_video(url):
    """
    Descarga una grabación pública de Zoom usando yt-dlp.
    Retorna la ruta local del archivo descargado.
    Lanza ZoomDownloadError si la URL no es válida o requiere autenticación.
    """
    # Validar que sea una URL de Zoom
    if not _re.match(r'https?://[\w.-]*zoom\.us/', url):
        raise ZoomDownloadError("El enlace no es una grabación de Zoom válida")

    local_filename = f"/tmp/zoom_{uuid.uuid4().hex[:12]}"
    ydl_opts = {
        'outtmpl': f'{local_filename}.%(ext)s',
        'quiet': True,
        'no_warnings': True,
        'format': 'best',
        # Sin credenciales — solo grabaciones públicas
        'username': None,
        'password': None,
    }

    print(f"  🔗 Descargando grabación de Zoom: {url[:80]}...")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            downloaded_file = ydl.prepare_filename(info)
            print(f"  ✅ Zoom descargado: {os.path.basename(downloaded_file)}")
            return downloaded_file
    except yt_dlp.utils.DownloadError as e:
        error_msg = str(e).lower()
        if any(kw in error_msg for kw in ['login', 'password', 'sign in', 'authenticate', 'private']):
            raise ZoomDownloadError(
                "La grabación de Zoom no es pública. "
                "Descarga el video manualmente y súbelo."
            )
        raise ZoomDownloadError(
            "No se pudo descargar la grabación. "
            "Verifica que el enlace sea correcto y público."
        )
    except Exception as e:
        raise ZoomDownloadError(
            f"No se pudo descargar la grabación. "
            f"Verifica que el enlace sea correcto y público."
        )


def upload_zoom_to_s3(local_path, bucket):
    """Sube un video de Zoom descargado a S3 y retorna el S3 key"""
    filename = os.path.basename(local_path)
    s3_key = f"zoom_downloads/{filename}"
    print(f"  ⬆️ Subiendo video de Zoom a s3://{bucket}/{s3_key}")
    s3.upload_file(local_path, bucket, s3_key)
    print(f"  ✅ Subido exitosamente")
    return s3_key


def download_s3_file(bucket, key, local_path):
    """Descarga un archivo de S3"""
    print(f"  ⬇️ Descargando s3://{bucket}/{key}")
    s3.download_file(bucket, key, local_path)
    return local_path


def extract_pdf_text(pdf_path):
    """Extrae texto de un PDF usando PyMuPDF"""
    print(f"  📄 Extrayendo texto de PDF: {os.path.basename(pdf_path)}")
    doc = fitz.open(pdf_path)
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    text = "\n".join(text_parts).strip()
    print(f"  ✅ Extraídos {len(text)} caracteres del PDF")
    return text


def transcribe_videos(video_keys, input_bucket, language, num_speakers):
    """Transcribe múltiples videos usando faster-whisper (reutiliza lógica existente)"""
    from main import convert_to_wav, diarize_audio, transcribe_with_whisper
    from main import merge_diarization_and_transcription, format_output
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_transcriptions = []

    for key in video_keys:
        print(f"\n🎬 Procesando video: {key}")
        local_path = f"/tmp/{os.path.basename(key)}"
        local_wav = f"/tmp/{os.path.basename(key)}_converted.wav"

        s3.download_file(input_bucket, key, local_path)

        if not convert_to_wav(local_path, local_wav):
            local_wav = local_path

        # Configurar speakers para este video
        original_env = os.environ.get('NUM_SPEAKERS', None)
        if num_speakers:
            os.environ['NUM_SPEAKERS'] = str(num_speakers)

        diarization = diarize_audio(local_wav, device)
        transcription = transcribe_with_whisper(local_wav, device, language)

        # Restaurar env
        if original_env is not None:
            os.environ['NUM_SPEAKERS'] = original_env
        elif 'NUM_SPEAKERS' in os.environ:
            del os.environ['NUM_SPEAKERS']

        if transcription:
            merged = merge_diarization_and_transcription(transcription, diarization)
            formatted = format_output(merged, is_structured=True)
            all_transcriptions.append(formatted)
            print(f"  ✅ Video transcrito: {len(formatted)} caracteres")
        else:
            print(f"  ❌ Error transcribiendo: {key}")

        # Limpiar archivos temporales
        for f in [local_path, local_wav]:
            if os.path.exists(f):
                os.remove(f)

    return "\n\n---\n\n".join(all_transcriptions)


# Acumulador de tokens y costos de OpenAI para el job actual
_ai_usage = {
    'total_input_tokens': 0,
    'total_output_tokens': 0,
    'total_cost': 0.0,
    'calls': 0,
}

# Precios OpenAI (USD por 1M tokens) — actualizar si cambian
_AI_PRICING = {
    'gpt-4o-mini': {'input': 0.15, 'output': 0.60},
    'gpt-4o': {'input': 2.50, 'output': 10.00},
}


def call_ai(prompt, content, model="gpt-4o-mini"):
    """Llama a la IA generativa (OpenAI GPT-4o-mini)"""
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY no está configurada")

    client = OpenAI(api_key=api_key)

    # Truncar contenido si excede el límite del modelo (~128k tokens)
    max_chars = 400000  # ~100k tokens aprox
    if len(content) > max_chars:
        print(f"  ⚠️ Contenido truncado de {len(content)} a {max_chars} caracteres")
        content = content[:max_chars]

    print(f"  🤖 Llamando a {model}...")
    call_start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": content}
        ],
        temperature=0.3,
        max_tokens=16000,
    )
    call_duration = time.time() - call_start

    result = response.choices[0].message.content

    # Rastrear tokens y costos
    usage = response.usage
    if usage:
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens
        pricing = _AI_PRICING.get(model, _AI_PRICING['gpt-4o-mini'])
        cost = (input_tokens * pricing['input'] + output_tokens * pricing['output']) / 1_000_000

        _ai_usage['total_input_tokens'] += input_tokens
        _ai_usage['total_output_tokens'] += output_tokens
        _ai_usage['total_cost'] += cost
        _ai_usage['calls'] += 1

        print(f"  ✅ Respuesta: {len(result)} chars | {input_tokens}+{output_tokens} tokens | ${cost:.4f} | {call_duration:.1f}s")
    else:
        print(f"  ✅ Respuesta recibida: {len(result)} caracteres | {call_duration:.1f}s")

    return result


def generate_pdf(sections, output_path, title="MentorIA - Resultado"):
    """Genera un PDF con los resultados usando ReportLab"""
    print(f"  📝 Generando PDF: {output_path}")

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()

    # Estilos personalizados
    title_style = ParagraphStyle(
        'MentorTitle',
        parent=styles['Title'],
        fontSize=20,
        spaceAfter=20,
    )
    heading_style = ParagraphStyle(
        'MentorHeading',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=20,
    )
    body_style = ParagraphStyle(
        'MentorBody',
        parent=styles['Normal'],
        fontSize=11,
        leading=16,
        alignment=TA_JUSTIFY,
        spaceAfter=8,
    )
    subheading_style = ParagraphStyle(
        'MentorSubheading',
        parent=styles['Heading2'],
        fontSize=13,
        spaceAfter=8,
        spaceBefore=14,
    )

    elements = []
    elements.append(Paragraph(title, title_style))
    elements.append(Spacer(1, 12))

    section_titles = {
        'resumen': 'Resumen Ejecutivo',
        'guia': 'Guía de Estudio',
        'banco_preguntas': 'Banco de Preguntas',
        'alerta_examen': 'Alertas de Examen',
        'analogias': 'Analogías Explicativas',
    }

    for section_type, content in sections:
        section_name = section_titles.get(section_type, section_type)
        elements.append(Paragraph(section_name, heading_style))
        elements.append(Spacer(1, 6))

        # Procesar el contenido línea por línea
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                elements.append(Spacer(1, 6))
                continue

            # Escapar caracteres especiales de XML para ReportLab
            safe_line = (line.replace('&', '&amp;')
                            .replace('<', '&lt;')
                            .replace('>', '&gt;'))

            # Detectar encabezados markdown
            if safe_line.startswith('### '):
                elements.append(Paragraph(safe_line[4:], subheading_style))
            elif safe_line.startswith('## '):
                elements.append(Paragraph(safe_line[3:], subheading_style))
            elif safe_line.startswith('# '):
                elements.append(Paragraph(safe_line[2:], heading_style))
            elif safe_line.startswith('**') and safe_line.endswith('**'):
                bold_text = f"<b>{safe_line[2:-2]}</b>"
                elements.append(Paragraph(bold_text, body_style))
            else:
                # Convertir negritas inline **texto**
                import re
                safe_line = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', safe_line)
                elements.append(Paragraph(safe_line, body_style))

    doc.build(elements)
    print(f"  ✅ PDF generado exitosamente")
    return output_path


def send_callback(callback_url, payload):
    """Envía el resultado al callback_url"""
    if not callback_url:
        print("  ⚠️ No hay callback_url, saltando notificación")
        return

    print(f"  📤 Enviando callback a: {callback_url}")
    try:
        resp = requests.post(
            callback_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        print(f"  ✅ Callback enviado (status: {resp.status_code})")
    except Exception as e:
        print(f"  ❌ Error enviando callback: {e}")


def process_mentoria():
    """Pipeline principal de MentorIA"""
    job_start = time.time()
    print("🧠 Iniciando pipeline MentorIA...")

    # Leer configuración desde variables de entorno
    input_bucket = os.environ.get('S3_INPUT_BUCKET')
    output_bucket = os.environ.get('S3_OUTPUT_BUCKET')
    job_id = os.environ.get('AWS_BATCH_JOB_ID', os.environ.get('JOB_ID', 'unknown'))
    callback_url = os.environ.get('CALLBACK_URL', '')
    language = os.environ.get('LANGUAGE', 'auto')
    num_speakers = os.environ.get('NUM_SPEAKERS', None)

    # Configuración de MentorIA (viene serializada como JSON)
    job_config_str = os.environ.get('JOB_CONFIG', '{}')
    try:
        job_config = json.loads(job_config_str)
    except json.JSONDecodeError:
        print("❌ Error parseando JOB_CONFIG")
        return

    video_keys = list(job_config.get('video_keys', []))
    zoom_urls = job_config.get('zoom_urls', [])
    pdf_keys = job_config.get('pdf_keys', [])
    outputs = job_config.get('outputs', ['resumen'])
    paginas = job_config.get('paginas_output', 3)
    disciplina = job_config.get('disciplina', '')
    usar_analogias = job_config.get('usar_analogias', False)
    analogia_contexto = (job_config.get('analogia_contexto') or '')[:30].strip()

    print(f"📋 Configuración:")
    print(f"   Videos: {len(video_keys)}")
    print(f"   Zoom URLs: {len(zoom_urls)}")
    print(f"   PDFs: {len(pdf_keys)}")
    print(f"   Outputs: {outputs}")
    print(f"   Páginas: {paginas}")
    print(f"   Disciplina: {disciplina or 'no especificada'}")
    print(f"   Usar analogías: {usar_analogias}")
    if analogia_contexto:
        print(f"   Contexto analogías: {analogia_contexto}")

    # --- PASO 0: Descargar grabaciones de Zoom ---
    if zoom_urls:
        print(f"\n🔗 PASO 0: Descargando {len(zoom_urls)} grabaciones de Zoom...")
        for url in zoom_urls:
            try:
                local_path = download_zoom_video(url)
                s3_key = upload_zoom_to_s3(local_path, input_bucket)
                video_keys.append(s3_key)
                print(f"  ✅ Zoom → {s3_key}")
                # Limpiar archivo local
                if os.path.exists(local_path):
                    os.remove(local_path)
            except ZoomDownloadError as e:
                print(f"  ❌ Error Zoom: {e}")
                send_callback(callback_url, {
                    "job_id": job_id,
                    "status": "error",
                    "error": str(e),
                })
                return

    # --- PASO 1: Transcribir videos ---
    transcription_text = ""
    if video_keys:
        print("\n📹 PASO 1: Transcribiendo videos...")
        transcription_text = transcribe_videos(
            video_keys, input_bucket, language, num_speakers
        )
        print(f"✅ Transcripción total: {len(transcription_text)} caracteres")
    else:
        print("\n⏭️ PASO 1: Sin videos, saltando transcripción")

    # --- PASO 2: Extraer texto de PDFs ---
    pdf_texts = []
    if pdf_keys:
        print("\n📄 PASO 2: Extrayendo texto de PDFs...")
        for key in pdf_keys:
            local_path = f"/tmp/{os.path.basename(key)}"
            download_s3_file(input_bucket, key, local_path)
            text = extract_pdf_text(local_path)
            if text:
                pdf_texts.append(text)
            if os.path.exists(local_path):
                os.remove(local_path)
    else:
        print("\n⏭️ PASO 2: Sin PDFs, saltando extracción")

    pdf_text = "\n\n".join(pdf_texts)

    # --- PASO 3: Combinar todo el texto ---
    print("\n📝 PASO 3: Combinando contenido...")
    combined_parts = []
    if transcription_text:
        combined_parts.append(f"=== TRANSCRIPCIÓN DE CLASES ===\n{transcription_text}")
    if pdf_text:
        combined_parts.append(f"=== APUNTES Y MATERIAL DE APOYO ===\n{pdf_text}")

    combined_text = "\n\n".join(combined_parts)

    if not combined_text.strip():
        print("❌ No hay contenido para procesar")
        send_callback(callback_url, {
            "job_id": job_id,
            "status": "error",
            "error": "No se pudo extraer contenido de los archivos proporcionados"
        })
        return

    print(f"✅ Contenido combinado: {len(combined_text)} caracteres")

    # --- PASO 4: Generar outputs con IA ---
    print("\n🤖 PASO 4: Generando contenido con IA...")
    sections = []

    # Preparar contextos para los prompts
    disciplina_ctx = f" especializado en {disciplina}" if disciplina else ""
    analogias_ctx = ANALOGIAS_INSTRUCCION if usar_analogias else ""

    # Contexto para el output "analogias"
    if analogia_contexto:
        analogia_tema = f"con '{analogia_contexto}'"
    else:
        analogia_tema = "claras y cotidianas"

    for output_type in outputs:
        if output_type not in PROMPTS:
            print(f"  ⚠️ Tipo de output desconocido: {output_type}, saltando")
            continue

        prompt_template = PROMPTS[output_type]
        format_vars = {
            'paginas': paginas,
            'disciplina_ctx': disciplina_ctx,
            'analogias_ctx': analogias_ctx,
            'analogia_tema': analogia_tema,
        }
        prompt = prompt_template.format_map(
            type('FormatDict', (dict,), {'__missing__': lambda self, key: f'{{{key}}}'})(format_vars)
        )

        print(f"\n  📝 Generando: {output_type}")
        result = call_ai(prompt, combined_text)
        sections.append((output_type, result))

    if not sections:
        print("❌ No se generaron secciones")
        send_callback(callback_url, {
            "job_id": job_id,
            "status": "error",
            "error": "No se pudieron generar los outputs solicitados"
        })
        return

    # --- PASO 5: Generar PDF ---
    print("\n📄 PASO 5: Generando PDF del resultado...")
    pdf_output_path = f"/tmp/mentoria_{job_id}.pdf"
    generate_pdf(sections, pdf_output_path, title="MentorIA - Material de Estudio")

    # --- PASO 6: Subir PDF a S3 ---
    print("\n⬆️ PASO 6: Subiendo PDF a S3...")
    result_s3_key = f"mentoria/{job_id}/resultado.pdf"
    try:
        s3.upload_file(pdf_output_path, output_bucket, result_s3_key)
        print(f"✅ PDF subido: s3://{output_bucket}/{result_s3_key}")
    except Exception as e:
        print(f"❌ Error subiendo PDF: {e}")
        send_callback(callback_url, {
            "job_id": job_id,
            "status": "error",
            "error": f"Error subiendo resultado: {str(e)}"
        })
        return

    # También subir JSON con el texto crudo de cada sección
    json_result = {
        "job_id": job_id,
        "sections": {s_type: content for s_type, content in sections},
        "pdf_s3_key": result_s3_key,
    }
    json_output_path = f"/tmp/mentoria_{job_id}.json"
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(json_result, f, ensure_ascii=False, indent=2)

    json_s3_key = f"mentoria/{job_id}/resultado.json"
    s3.upload_file(json_output_path, output_bucket, json_s3_key)
    print(f"✅ JSON subido: s3://{output_bucket}/{json_s3_key}")

    # --- PASO 7: Callback ---
    print("\n📤 PASO 7: Notificando resultado...")
    send_callback(callback_url, {
        "job_id": job_id,
        "status": "completed",
        "result_s3_key": result_s3_key,
        "json_s3_key": json_s3_key,
    })

    # --- RESUMEN DE COSTOS ---
    elapsed = time.time() - job_start
    elapsed_min = elapsed / 60
    elapsed_hr = elapsed / 3600

    # Costo de instancia Batch (GPU)
    batch_cost_hr = float(os.environ.get('AWS_BATCH_COST_PER_HOUR', '1.006'))  # g5.xlarge default
    batch_cost = elapsed_hr * batch_cost_hr

    import torch
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    print("\n" + "=" * 55)
    print("💰 RESUMEN DE RECURSOS Y COSTOS — MentorIA")
    print("=" * 55)
    print(f"   Tiempo total:          {elapsed_min:.1f} min ({elapsed:.0f}s)")
    print(f"   GPU utilizada:         {gpu_name}")
    print(f"   Instancia Batch:       ${batch_cost_hr:.3f}/hr")
    print(f"   ─── Desglose ───")
    print(f"   Costo Batch (GPU):     ${batch_cost:.4f}")
    if _ai_usage['calls'] > 0:
        print(f"   Costo OpenAI:          ${_ai_usage['total_cost']:.4f}")
        print(f"     └ {_ai_usage['calls']} llamadas | {_ai_usage['total_input_tokens']:,} input + {_ai_usage['total_output_tokens']:,} output tokens")
    print(f"   Costo S3 (estimado):   ~$0.0001")
    total = batch_cost + _ai_usage['total_cost'] + 0.0001
    print(f"   ─────────────────────────────")
    print(f"   COSTO TOTAL ESTIMADO:  ${total:.4f}")
    print("=" * 55)

    print("\n🎉 ¡Pipeline MentorIA completado exitosamente!")


if __name__ == "__main__":
    process_mentoria()
