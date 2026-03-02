import os
import boto3
import torch
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pydub import AudioSegment
import json

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
            use_auth_token=HF_TOKEN
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
            "large-v3-turbo",
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

def process_media():
    print("🚀 Iniciando proceso de transcripción Rimai con diarización...")
    print(f"📊 GPU disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
    
    if not S3_INPUT_BUCKET or not FILE_NAME:
        print("❌ Error: Faltan variables de entorno.")
        return

    local_input_path = f"/tmp/{FILE_NAME}"
    local_wav_path = f"/tmp/{FILE_NAME}_converted.wav"
    local_output_path = f"/tmp/{FILE_NAME}_transcription.txt"
    local_json_path = f"/tmp/{FILE_NAME}_transcription.json"
    s3_key_output_txt = f"transcriptions/{FILE_NAME}.txt"
    s3_key_output_json = f"transcriptions/{FILE_NAME}.json"

    print(f"⬇️ Descargando: {FILE_NAME}...")
    try:
        s3.download_file(S3_INPUT_BUCKET, FILE_NAME, local_input_path)
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
Archivo: {FILE_NAME}
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

if __name__ == "__main__":
    process_media()