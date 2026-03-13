import os
import time
import boto3
import torch
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pydub import AudioSegment
import json

# --- COSTOS AWS (USD/hora, on-demand us-east-1) ---
# Ajustar según la instancia real de tu Job Definition en AWS Batch
AWS_BATCH_COST_PER_HOUR = float(os.environ.get('AWS_BATCH_COST_PER_HOUR', '1.006'))  # g5.xlarge default

# --- CONFIGURACIÓN ---
S3_INPUT_BUCKET = os.environ.get('S3_INPUT_BUCKET')
S3_OUTPUT_BUCKET = os.environ.get('S3_OUTPUT_BUCKET')
FILE_NAME = os.environ.get('FILE_NAME')
USER_EMAIL = os.environ.get('USER_EMAIL')
LANGUAGE = os.environ.get('LANGUAGE', 'auto')
HF_TOKEN = os.environ.get('HF_TOKEN')
NUM_SPEAKERS_ENV = os.environ.get('NUM_SPEAKERS', None)
s3 = boto3.client('s3')

def convert_to_wav(input_path, output_path):
    """Convierte cualquier audio/video a WAV mono 16kHz"""
    try:
        print("🔄 Convirtiendo a formato WAV...")
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        audio.export(output_path, format="wav")
        print("✅ Conversión completada")
        return True
    except Exception as e:
        print(f"❌ Error en conversión: {e}")
        return False

# ELIMINADO: detect_language() ya no es necesario.
# faster-whisper detecta el idioma automáticamente durante la transcripción.

def diarize_audio(audio_path, device):
    """Identifica quién habla y cuándo (Dinámico)"""
    print("👥 Iniciando diarización de hablantes...")
    
    if not HF_TOKEN:
        print("⚠️ No hay HF_TOKEN. Saltando diarización.")
        return None
    
    # NUEVO: Si es 1 solo hablante, no gastar GPU en diarización
    if NUM_SPEAKERS_ENV and NUM_SPEAKERS_ENV == "1":
        print("☝️ Un solo hablante indicado, saltando diarización.")
        return None
    
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=HF_TOKEN
        )
        pipeline.to(torch.device(device))
        
        # --- LÓGICA DINÁMICA DE HABLANTES ---
        diarization_params = {}
        
        # Si recibimos un número válido (ej: "2", "3"), lo forzamos.
        # Si no recibimos nada (None o vacío), dejamos que Pyannote decida.
        if NUM_SPEAKERS_ENV and NUM_SPEAKERS_ENV.isdigit() and int(NUM_SPEAKERS_ENV) > 0:
            n_speakers = int(NUM_SPEAKERS_ENV)
            print(f"🔒 Forzando detección a exactamente {n_speakers} hablantes.")
            diarization_params = {
                "min_speakers": n_speakers,
                "max_speakers": n_speakers
            }
        else:
            print("🔓 Modo Automático: Detectando cantidad de hablantes...")
            diarization_params = {
                "min_speakers": 1,
                "max_speakers": 20
            }

        diarization = pipeline(audio_path, **diarization_params)
        
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker
            })
        
        num_speakers_detected = len(set([s['speaker'] for s in segments]))
        print(f"✅ Diarización completada. {num_speakers_detected} hablantes encontrados.")
        return segments
        
    except Exception as e:
        print(f"❌ Error en diarización: {e}")
        return None

def transcribe_with_whisper(audio_path, device, language='auto'):
    """Transcribe con Faster-Whisper (multiidioma, 4x más rápido)"""
    print(f"🎙️ Transcribiendo con Faster-Whisper (idioma: {language.upper()})...")
    
    try:
        # CAMBIO: WhisperModel en vez de whisper.load_model
        # compute_type="float16" equivale a fp16=True del original
        model = WhisperModel(
            "large-v3",
            device=device,
            compute_type="float16" if device == "cuda" else "int8"
        )
        
        # CAMBIO: faster-whisper detecta el idioma internamente,
        # ya no necesitamos la función detect_language() separada
        if language == 'auto':
            language = None  # faster-whisper auto-detecta si es None
        
        # --- MISMOS PARÁMETROS ANTI-BUCLE QUE TENÍAS ---
        segments_generator, info = model.transcribe(
            audio_path,
            word_timestamps=True,
            beam_size=5,
            best_of=5,
            # Permitimos que varíe la temperatura si se atasca
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            # ¡CRÍTICO! Desactivar esto evita que repita texto infinitamente
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,     # NOTA: en faster-whisper es log_prob_threshold
            no_speech_threshold=0.6,
            language=language,
            # NUEVO: VAD filter - filtra silencios ANTES de transcribir
            # Reduce tiempo de proceso 10-30% y reduce alucinaciones en silencios
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        
        # CAMBIO: faster-whisper devuelve un generador, hay que materializarlo
        result_segments = []
        full_text = []
        for seg in segments_generator:
            result_segments.append({
                'start': seg.start,
                'end': seg.end,
                'text': seg.text
            })
            full_text.append(seg.text.strip())
        
        detected_lang = info.language
        print(f"✅ Transcripción completada en {detected_lang.upper()}")
        
        # Devolvemos en el MISMO formato que el código original espera
        return {
            'segments': result_segments,
            'language': detected_lang,
            'text': ' '.join(full_text)
        }
        
    except Exception as e:
        print(f"❌ Error en transcripción: {e}")
        return None

def merge_diarization_and_transcription(transcription, diarization):
    """Combina la transcripción con los hablantes identificados"""
    if not diarization:
        return [{'speaker': 'SPEAKER_00', 'text': transcription['text'], 'start': 0, 'end': 0}]
    
    print("🔗 Fusionando transcripción con hablantes...")
    
    segments_with_words = []
    if 'segments' not in transcription:
         return [{'speaker': 'ERROR', 'text': 'Error en transcripción interna', 'start': 0, 'end': 0}]

    for segment in transcription['segments']:
        segments_with_words.append({
            'start': segment['start'],
            'end': segment['end'],
            'text': segment['text']
        })
    
    result = []
    for seg in segments_with_words:
        seg_middle = (seg['start'] + seg['end']) / 2
        
        speaker = "SPEAKER_UNKNOWN"
        for d in diarization:
            if d['start'] <= seg_middle <= d['end']:
                speaker = d['speaker']
                break
        
        result.append({
            'speaker': speaker,
            'start': seg['start'],
            'end': seg['end'],
            'text': seg['text'].strip()
        })
    
    return result

def format_output(merged_data, is_structured=True):
    """Formatea la salida para lectura humana"""
    if not merged_data: return ""
    if isinstance(merged_data, str): return merged_data
    
    if not is_structured:
        return json.dumps(merged_data, indent=2, ensure_ascii=False)
    
    output = []
    current_speaker = None
    current_text = []
    
    for item in merged_data:
        speaker = item.get('speaker', 'UNKNOWN')
        text = item.get('text', '')
        
        if speaker != current_speaker:
            if current_speaker:
                output.append(f"\n[{current_speaker}]: {' '.join(current_text)}")
            current_speaker = speaker
            current_text = [text]
        else:
            current_text.append(text)
    
    if current_speaker:
        output.append(f"\n[{current_speaker}]: {' '.join(current_text)}")
    
    return '\n'.join(output)

def download_zoom_for_transcription():
    """Descarga grabación de Zoom y retorna el nombre del archivo local"""
    zoom_url = os.environ.get('ZOOM_URL', '')
    if not zoom_url:
        return None

    print(f"🔗 Descargando grabación de Zoom...")
    try:
        from mentoria import download_zoom_video, upload_zoom_to_s3
        local_path = download_zoom_video(zoom_url)
        # Subir a S3 para consistencia
        s3_key = upload_zoom_to_s3(local_path, S3_INPUT_BUCKET)
        print(f"✅ Zoom descargado y subido como: {s3_key}")
        return s3_key, local_path
    except Exception as e:
        print(f"❌ Error descargando Zoom: {e}")
        return None


def print_cost_summary(elapsed_seconds, gpu_name=""):
    """Imprime resumen de recursos y costos estimados"""
    elapsed_min = elapsed_seconds / 60
    elapsed_hr = elapsed_seconds / 3600
    batch_cost = elapsed_hr * AWS_BATCH_COST_PER_HOUR

    print("\n" + "=" * 50)
    print("💰 RESUMEN DE RECURSOS Y COSTOS")
    print("=" * 50)
    print(f"   Tiempo total:        {elapsed_min:.1f} min ({elapsed_seconds:.0f}s)")
    if gpu_name:
        print(f"   GPU utilizada:       {gpu_name}")
    print(f"   Instancia Batch:     ${AWS_BATCH_COST_PER_HOUR:.3f}/hr")
    print(f"   Costo Batch (GPU):   ${batch_cost:.4f}")
    print(f"   Costo S3 (estimado): ~$0.0001")
    print(f"   ─────────────────────────────")
    print(f"   COSTO TOTAL ESTIMADO: ${batch_cost + 0.0001:.4f}")
    print("=" * 50)


def process_media():
    job_start = time.time()
    print("🚀 Iniciando proceso de transcripción Rimai con diarización...")
    print(f"📊 GPU disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"📊 GPU: {torch.cuda.get_device_name(0)}")

    # Si viene una URL de Zoom, descargarla primero
    zoom_result = download_zoom_for_transcription()
    file_name = FILE_NAME

    if zoom_result:
        s3_key, _ = zoom_result
        file_name = s3_key
    elif not S3_INPUT_BUCKET or not file_name:
        print("❌ Error: Faltan variables de entorno (FILE_NAME o ZOOM_URL).")
        return

    local_input_path = f"/tmp/{os.path.basename(file_name)}"
    base_name = os.path.basename(file_name)
    local_wav_path = f"/tmp/{base_name}_converted.wav"
    local_output_path = f"/tmp/{base_name}_transcription.txt"
    local_json_path = f"/tmp/{base_name}_transcription.json"
    s3_key_output_txt = f"transcriptions/{base_name}.txt"
    s3_key_output_json = f"transcriptions/{base_name}.json"

    print(f"⬇️ Descargando: {file_name}...")
    try:
        s3.download_file(S3_INPUT_BUCKET, file_name, local_input_path)
    except Exception as e:
        print(f"❌ Error descargando: {e}")
        return

    if not convert_to_wav(local_input_path, local_wav_path):
        local_wav_path = local_input_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Diarización
    diarization = diarize_audio(local_wav_path, device)
    
    # 2. Transcripción
    transcription = transcribe_with_whisper(local_wav_path, device, LANGUAGE)
    
    if not transcription:
        print("❌ Falló la transcripción.")
        return

    # 3. Fusión
    merged = merge_diarization_and_transcription(transcription, diarization)
    
    # 4. Resultados
    formatted_text = format_output(merged, is_structured=True)
    json_output = format_output(merged, is_structured=False)

    detected_language = transcription.get('language', LANGUAGE)
    
    # Contamos hablantes únicos
    speakers_set = set()
    if isinstance(merged, list):
        for m in merged:
            if 'speaker' in m: speakers_set.add(m['speaker'])
            
    reporte_txt = f"""
--- REPORTE RIMAI ---
Archivo: {file_name}
Usuario: {USER_EMAIL}
Idioma: {detected_language.upper()}
Procesado en: {device.upper()}
Hablantes detectados: {len(speakers_set)}
Motor: Faster-Whisper large-v3 (VAD + CTranslate2)
---------------------
{formatted_text}
"""

    print(f"⬆️ Subiendo resultados...")
    try:
        with open(local_output_path, "w", encoding="utf-8") as f:
            f.write(reporte_txt)
        s3.upload_file(local_output_path, S3_OUTPUT_BUCKET, s3_key_output_txt)
        
        with open(local_json_path, "w", encoding="utf-8") as f:
            f.write(json_output)
        s3.upload_file(local_json_path, S3_OUTPUT_BUCKET, s3_key_output_json)
        
        print(f"🎉 ¡ÉXITO COMPLETO!!")

    except Exception as e:
        print(f"❌ Error subiendo: {e}")

    # Resumen de costos
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print_cost_summary(time.time() - job_start, gpu_name)

if __name__ == "__main__":
    job_type = os.environ.get('JOB_TYPE', 'transcription')

    if job_type == 'mentoria':
        from mentoria import process_mentoria
        process_mentoria()
    else:
        process_media()