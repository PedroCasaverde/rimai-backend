import os
import boto3
import whisper
import torch
from pyannote.audio import Pipeline
from pydub import AudioSegment
import json

# --- CONFIGURACIÓN ---
S3_INPUT_BUCKET = os.environ.get('S3_INPUT_BUCKET')
S3_OUTPUT_BUCKET = os.environ.get('S3_OUTPUT_BUCKET')
FILE_NAME = os.environ.get('FILE_NAME')
USER_EMAIL = os.environ.get('USER_EMAIL')

# Token de HuggingFace (necesario para pyannote)
# Debes obtenerlo en: https://huggingface.co/settings/tokens
# Y aceptar los términos en: https://huggingface.co/pyannote/speaker-diarization-3.1
HF_TOKEN = os.environ.get('HF_TOKEN')

s3 = boto3.client('s3')

def convert_to_wav(input_path, output_path):
    """Convierte cualquier audio/video a WAV mono 16kHz"""
    try:
        print("🔄 Convirtiendo a formato WAV...")
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1)  # Mono
        audio = audio.set_frame_rate(16000)  # 16kHz
        audio.export(output_path, format="wav")
        print("✅ Conversión completada")
        return True
    except Exception as e:
        print(f"❌ Error en conversión: {e}")
        return False

def diarize_audio(audio_path, device):
    """Identifica quién habla y cuándo"""
    print("👥 Iniciando diarización de hablantes...")
    
    if not HF_TOKEN:
        print("⚠️ No hay HF_TOKEN. Saltando diarización.")
        return None
    
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_TOKEN
        )
        pipeline.to(torch.device(device))
        
        diarization = pipeline(audio_path)
        
        # Convertir a formato usable
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker
            })
        
        print(f"✅ Diarización completada. {len(set([s['speaker'] for s in segments]))} hablantes detectados")
        return segments
        
    except Exception as e:
        print(f"❌ Error en diarización: {e}")
        return None

def transcribe_with_whisper(audio_path, device):
    """Transcribe con timestamps"""
    print("🎙️ Transcribiendo con Whisper...")
    
    try:
        model = whisper.load_model("medium", device=device)
        result = model.transcribe(
            audio_path,
            language="es",  # Cambia si es otro idioma
            word_timestamps=True,
            verbose=False
        )
        print("✅ Transcripción completada")
        return result
        
    except Exception as e:
        print(f"❌ Error en transcripción: {e}")
        return None

def merge_diarization_and_transcription(transcription, diarization):
    """Combina la transcripción con los hablantes identificados"""
    if not diarization:
        return transcription['text']
    
    print("🔗 Fusionando transcripción con hablantes...")
    
    # Extraer segmentos con palabras
    segments_with_words = []
    for segment in transcription['segments']:
        segments_with_words.append({
            'start': segment['start'],
            'end': segment['end'],
            'text': segment['text']
        })
    
    # Asignar hablante a cada segmento
    result = []
    for seg in segments_with_words:
        seg_middle = (seg['start'] + seg['end']) / 2
        
        # Encontrar qué hablante estaba activo en ese momento
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
    if isinstance(merged_data, str):
        return merged_data
    
    if not is_structured:
        return json.dumps(merged_data, indent=2, ensure_ascii=False)
    
    # Formato conversacional
    output = []
    current_speaker = None
    current_text = []
    
    for item in merged_data:
        speaker = item['speaker']
        text = item['text']
        
        if speaker != current_speaker:
            if current_speaker:
                output.append(f"{current_speaker}: {' '.join(current_text)}\n")
            current_speaker = speaker
            current_text = [text]
        else:
            current_text.append(text)
    
    # Añadir último hablante
    if current_speaker:
        output.append(f"{current_speaker}: {' '.join(current_text)}\n")
    
    return '\n'.join(output)

def process_media():
    print("🚀 Iniciando proceso de transcripción Rimai con diarización...")

    # 0. Verificaciones
    if not S3_INPUT_BUCKET or not FILE_NAME:
        print("❌ Error: Faltan variables de entorno.")
        return

    # 1. Preparar rutas
    local_input_path = f"/tmp/{FILE_NAME}"
    local_wav_path = f"/tmp/{FILE_NAME}_converted.wav"
    local_output_path = f"/tmp/{FILE_NAME}_transcription.txt"
    local_json_path = f"/tmp/{FILE_NAME}_transcription.json"
    s3_key_output_txt = f"transcriptions/{FILE_NAME}.txt"
    s3_key_output_json = f"transcriptions/{FILE_NAME}.json"

    # 2. Descargar archivo
    print(f"⬇️ Descargando: {FILE_NAME}...")
    try:
        s3.download_file(S3_INPUT_BUCKET, FILE_NAME, local_input_path)
        print("✅ Descarga completada")
    except Exception as e:
        print(f"❌ Error descargando: {e}")
        return

    # 3. Convertir a WAV
    if not convert_to_wav(local_input_path, local_wav_path):
        print("⚠️ No se pudo convertir, intentando con archivo original...")
        local_wav_path = local_input_path

    # 4. Detectar dispositivo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Usando: {device.upper()}")

    # 5. Diarización
    diarization = diarize_audio(local_wav_path, device)

    # 6. Transcripción
    transcription = transcribe_with_whisper(local_wav_path, device)
    if not transcription:
        return

    # 7. Fusionar
    merged = merge_diarization_and_transcription(transcription, diarization)

    # 8. Crear reportes
    formatted_text = format_output(merged, is_structured=True)
    json_output = format_output(merged, is_structured=False)

    reporte_txt = f"""
--- REPORTE RIMAI ---
Video: {FILE_NAME}
Usuario: {USER_EMAIL}
Procesado en: {device.upper()}
Hablantes detectados: {len(set([s['speaker'] for s in merged])) if isinstance(merged, list) else 'N/A'}
---------------------

{formatted_text}
"""

    # 9. Guardar y subir
    print(f"⬆️ Subiendo resultados...")
    try:
        # TXT
        with open(local_output_path, "w", encoding="utf-8") as f:
            f.write(reporte_txt)
        s3.upload_file(local_output_path, S3_OUTPUT_BUCKET, s3_key_output_txt)
        
        # JSON
        with open(local_json_path, "w", encoding="utf-8") as f:
            f.write(json_output)
        s3.upload_file(local_json_path, S3_OUTPUT_BUCKET, s3_key_output_json)
        
        print(f"🎉 ¡ÉXITO!")
        print(f"   TXT: s3://{S3_OUTPUT_BUCKET}/{s3_key_output_txt}")
        print(f"   JSON: s3://{S3_OUTPUT_BUCKET}/{s3_key_output_json}")
        
    except Exception as e:
        print(f"❌ Error subiendo: {e}")

if __name__ == "__main__":
    process_media()