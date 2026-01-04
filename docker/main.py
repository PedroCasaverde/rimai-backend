import os
import boto3
import whisper
import torch # Importamos torch para verificar la GPU

# --- CONFIGURACIÓN ---
# Estas variables vienen de las "Environment Variables" que configuramos en la Lambda
S3_INPUT_BUCKET = os.environ.get('S3_INPUT_BUCKET')
S3_OUTPUT_BUCKET = os.environ.get('S3_OUTPUT_BUCKET')
FILE_NAME = os.environ.get('FILE_NAME')   # <--- OJO: En la Lambda usamos 'FILE_NAME'
USER_EMAIL = os.environ.get('USER_EMAIL') # <--- Para usarlo en el reporte

s3 = boto3.client('s3')

def process_media():
    print("🚀 Iniciando proceso de transcripción Rimai...")

    # 0. Verificaciones de seguridad
    if not S3_INPUT_BUCKET or not FILE_NAME:
        print("❌ Error: Faltan variables de entorno.")
        return

    # 1. Preparar rutas
    local_input_path = f"/tmp/{FILE_NAME}"
    local_output_path = f"/tmp/{FILE_NAME}.txt"
    s3_key_output = f"transcriptions/{FILE_NAME}.txt"

    # 2. Descargar archivo de S3
    print(f"⬇️ Descargando archivo: {FILE_NAME} desde {S3_INPUT_BUCKET}...")
    try:
        s3.download_file(S3_INPUT_BUCKET, FILE_NAME, local_input_path)
        print("✅ Descarga completada.")
    except Exception as e:
        print(f"❌ Error descargando de S3: {e}")
        return

    # 3. Cargar Whisper y verificar GPU
    # Esto es vital para saber si AWS nos dio la tarjeta gráfica
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🧠 Cargando modelo Whisper en dispositivo: {device.upper()} ...")
    
    try:
        # Usamos 'medium' o 'small' para probar rápido. 'large-v2' para producción.
        model = whisper.load_model("medium", device=device)
    except Exception as e:
        print(f"❌ Error cargando el modelo: {e}")
        return

    # 4. Transcribir
    print("🎙️ Transcribiendo (esto puede tardar)...")
    try:
        result = model.transcribe(local_input_path)
        transcribed_text = result['text']
        print("✅ Transcripción finalizada.")
    except Exception as e:
        print(f"❌ Error durante la transcripción: {e}")
        return

    # 5. Crear reporte simple
    reporte = f"""
    --- REPORTE RIMAI ---
    Video: {FILE_NAME}
    Usuario: {USER_EMAIL}
    Procesado en: {device}
    ---------------------
    
    {transcribed_text}
    """

    # 6. Guardar y Subir a S3 Output
    print(f"⬆️ Subiendo resultados a {S3_OUTPUT_BUCKET}...")
    try:
        with open(local_output_path, "w", encoding="utf-8") as f:
            f.write(reporte)
        
        s3.upload_file(local_output_path, S3_OUTPUT_BUCKET, s3_key_output)
        print(f"🎉 ¡ÉXITO! Archivo disponible en: s3://{S3_OUTPUT_BUCKET}/{s3_key_output}")
    except Exception as e:
        print(f"❌ Error subiendo resultado: {e}")

if __name__ == "__main__":
    process_media()