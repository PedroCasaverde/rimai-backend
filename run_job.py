import requests
import sys

# --- CONFIGURACIÓN ---
LAMBDA_URL = "https://gjdvwxf2a26ir3rsdsbgiomwdy0fbvlz.lambda-url.us-east-1.on.aws/"

MI_VIDEO = "Clase-2-ML.mp4"
MI_EMAIL = "casaverde.analitica@gmail.com"
MI_IDIOMA = "es" # Opciones: 'auto', 'es', 'en', 'fr', 'de', 'it', 'pt', 'ja', 'zh', etc.
MI_SPEAKERS = 1  
# ---------------------

def trigger_transcription(video=None, email=None, language=None, speakers=None):
    """
    Envía trabajo de transcripción a la Lambda.
    """
    # Si no pasan argumentos, usa las constantes de arriba
    video = video or MI_VIDEO
    email = email or MI_EMAIL
    language = language or MI_IDIOMA
    speakers = speakers or MI_SPEAKERS 
    
    print(f"🚀 Enviando orden para: {video}")
    print(f"📧 Email: {email}")
    print(f"🌍 Idioma: {language.upper()}")
    
    if speakers:
        print(f"👥 Hablantes forzados: {speakers}")
    else:
        print(f"👥 Hablantes: AUTO (El sistema decidirá)")
    
    payload = {
        "fileName": video,
        "email": email,
        "language": language,
        "speakers": speakers # Enviamos el dato a la Lambda
    }
    
    try:
        response = requests.post(LAMBDA_URL, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ ¡Éxito! La Lambda recibió la orden.")
            print(f"Job ID: {result.get('jobId')}")
            print(f"Configuración recibida: {result.get('speakers', 'Auto')} hablantes")
            print("\n--- PASOS SIGUIENTES ---")
            print("1. Ve a AWS Batch -> Jobs")
            print("2. Espera a que termine (RUNNABLE -> SUCCEEDED)")
        else:
            print(f"\n❌ Error {response.status_code}:", response.text)
            
    except Exception as e:
        print(f"\n❌ Error de conexión: {str(e)}")

if __name__ == "__main__":
    # Si ejecutas "python run_job.py" sin nada más, usará MI_SPEAKERS = 2
    
    # Opcional: Acepta argumentos por consola si quieres sobreescribir
    video = sys.argv[1] if len(sys.argv) > 1 else None
    email = sys.argv[2] if len(sys.argv) > 2 else None
    language = sys.argv[3] if len(sys.argv) > 3 else None
    speakers = sys.argv[4] if len(sys.argv) > 4 else None

    trigger_transcription(video, email, language, speakers)