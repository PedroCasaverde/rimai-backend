import requests
import sys

# --- CONFIGURACIÓN ---
# 1. PEGA AQUÍ TU URL DE LAMBDA (La que acabas de copiar)
LAMBDA_URL = "https://gjdvwxf2a26ir3rsdsbgiomwdy0fbvlz.lambda-url.us-east-1.on.aws/" 

# 2. CONFIGURA LOS DATOS DE PRUEBA
# Este archivo TIENE que estar ya subido en tu bucket de entrada (S3)
MI_VIDEO = "test_whisper.mp3" 
MI_EMAIL = "casaverde.analitica@gmail.com"
# ---------------------

def trigger_transcription():
    print(f"🚀 Enviando orden para: {MI_VIDEO}...")
    
    payload = {
        "fileName": MI_VIDEO,
        "email": MI_EMAIL
    }
    
    try:
        response = requests.post(LAMBDA_URL, json=payload)
        
        if response.status_code == 200:
            print("\n✅ ¡Éxito! La Lambda recibió la orden.")
            print(f"Respuesta del servidor: {response.json()}")
            print("\n--- PASOS SIGUIENTES ---")
            print("1. Ve a la consola de AWS Batch -> Jobs.")
            print("2. Busca en la cola 'Rimai-Queue'.")
            print("3. Deberías ver el trabajo pasando de RUNNABLE a RUNNING.")
        else:
            print(f"\n❌ Error {response.status_code}:", response.text)
            
    except Exception as e:
        print(f"\n❌ Error de conexión: {str(e)}")

if __name__ == "__main__":
    trigger_transcription()