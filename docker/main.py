import os
import boto3
import whisper
import time
from datetime import datetime
from decimal import Decimal

# Configuración de AWS (Las variables vendrán del entorno de AWS Batch)
S3_BUCKET_INPUT = os.environ.get('S3_BUCKET_INPUT')
S3_BUCKET_OUTPUT = os.environ.get('S3_BUCKET_OUTPUT')
DYNAMODB_TABLE = os.environ.get('DYNAMODB_TABLE')

s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(DYNAMODB_TABLE)

def update_user_balance(username, duration_seconds, used_translation):
    """
    Descuenta el saldo en DynamoDB.
    """
    try:
        # Preparamos la actualización
        update_expr = "set transcription_balance_seconds = transcription_balance_seconds - :d"
        expr_values = {':d': Decimal(str(duration_seconds))}
        
        if used_translation:
            update_expr += ", translation_credits = translation_credits - :c"
            expr_values[':c'] = 1

        table.update_item(
            Key={'username': username},
            UpdateExpression=update_expr,
            ExpressionAttributeValues=expr_values
        )
        print(f"✅ Saldo actualizado para {username}")
    except Exception as e:
        print(f"❌ Error actualizando saldo: {e}")

def process_media(file_key, username, language_src, translate_target=None):
    # 1. Descargar archivo
    local_filename = "/tmp/" + file_key.split('/')[-1]
    print(f"⬇️ Descargando {file_key}...")
    s3.download_file(S3_BUCKET_INPUT, file_key, local_filename)

    # 2. Cargar Whisper (Usamos 'medium' para balance calidad/velocidad)
    print("🧠 Cargando modelo Whisper...")
    model = whisper.load_model("medium") # O "large" si tienes buena GPU

    # 3. Transcribir (Esto hace la magia)
    print("🎙️ Transcribiendo...")
    # task="translate" en whisper solo traduce a Inglés. 
    # Si quieres custom, aquí iría la lógica extra de traducción.
    result = model.transcribe(local_filename, language=language_src)
    
    # Calcular duración real para cobro
    duration = result['segments'][-1]['end'] if result['segments'] else 0
    print(f"⏱️ Duración detectada: {duration} segundos")

    # 4. Formatear Salida (Simplificado)
    output_text = f"REPORTE DE TRANSCRIPCIÓN\nUsuario: {username}\nDuración: {duration}s\n\n"
    
    for segment in result['segments']:
        # Aquí simularíamos la diarización si usáramos WhisperX
        # Por ahora Whisper nativo no separa interlocutores perfectamente sin ayuda externa
        start = time.strftime('%H:%M:%S', time.gmtime(segment['start']))
        text = segment['text']
        
        # Lógica de traducción simple (Ejemplo)
        if translate_target:
             # Aquí llamarías a tu librería de traducción (ej. deep_translator)
             # text = traducir(text, translate_target)
             output_text += f"[{start}] (TRADUCIDO): {text}\n"
        else:
             output_text += f"[{start}]: {text}\n"

    # 5. Guardar y Subir
    output_key = file_key + "_resultado.txt"
    local_output = "/tmp/resultado.txt"
    
    with open(local_output, "w", encoding="utf-8") as f:
        f.write(output_text)
        
    s3.upload_file(local_output, S3_BUCKET_OUTPUT, output_key)
    print("✅ Archivo subido a S3 Output")

    # 6. Cobrar al usuario
    update_user_balance(username, duration, bool(translate_target))

if __name__ == "__main__":
    # Estos argumentos vendrán del Job de AWS Batch
    # Se pasan como variables de entorno al contenedor
    FILE_KEY = os.environ.get('FILE_KEY')
    USERNAME = os.environ.get('USERNAME')
    LANG_SRC = os.environ.get('LANG_SRC', 'es')
    TRANSLATE_TO = os.environ.get('TRANSLATE_TO', '') # Vacío si no traduce

    process_media(FILE_KEY, USERNAME, LANG_SRC, TRANSLATE_TO)