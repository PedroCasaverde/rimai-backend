import requests
import sys

# --- CONFIGURACIÓN ---
LAMBDA_URL = "https://gjdvwxf2a26ir3rsdsbgiomwdy0fbvlz.lambda-url.us-east-1.on.aws/"

MI_VIDEO = "test_cortazar.mp4"
MI_EMAIL = "casaverde.analitica@gmail.com"
MI_IDIOMA = "es"  # Opciones: 'auto', 'es', 'en', 'fr', 'de', 'it', 'pt', 'ja', 'zh', etc.
# ---------------------

def trigger_transcription(video=None, email=None, language=None):
    """
    Envía trabajo de transcripción
    
    Parámetros:
        video (str): Nombre del archivo en S3
        email (str): Email del usuario
        language (str): Código de idioma ISO ('auto', 'es', 'en', etc.)
    """
    video = video or MI_VIDEO
    email = email or MI_EMAIL
    language = language or MI_IDIOMA
    
    print(f"🚀 Enviando orden para: {video}")
    print(f"📧 Email: {email}")
    print(f"🌍 Idioma: {language.upper()}")
    
    payload = {
        "fileName": video,
        "email": email,
        "language": language
    }
    
    try:
        response = requests.post(LAMBDA_URL, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ ¡Éxito! La Lambda recibió la orden.")
            print(f"Job ID: {result.get('jobId')}")
            print(f"Idioma configurado: {result.get('language', 'auto').upper()}")
            print("\n--- PASOS SIGUIENTES ---")
            print("1. Ve a AWS Batch -> Jobs")
            print("2. Busca en la cola 'Rimai-Queue'")
            print("3. Monitorea el estado: RUNNABLE → RUNNING → SUCCEEDED")
            print(f"\n⏱️ Tiempo estimado: ~30-45 min para videos de 2 horas")
        else:
            print(f"\n❌ Error {response.status_code}:", response.text)
            
    except Exception as e:
        print(f"\n❌ Error de conexión: {str(e)}")

if __name__ == "__main__":
    # Modo interactivo
    if len(sys.argv) > 1:
        if sys.argv[1] == '--help':
            print("""
Uso: python trigger_test.py [video] [email] [language]

Ejemplos:
  python trigger_test.py                                    # Usa valores por defecto
  python trigger_test.py video.mp4 user@email.com es        # Español
  python trigger_test.py video.mp4 user@email.com auto      # Detección automática
  python trigger_test.py video.mp4 user@email.com en        # Inglés

Idiomas soportados:
  auto - Detección automática (recomendado)
  es   - Español
  en   - Inglés
  fr   - Francés
  de   - Alemán
  it   - Italiano
  pt   - Portugués
  ja   - Japonés
  zh   - Chino
  ... y más de 90 idiomas
            """)
            sys.exit(0)
        
        video = sys.argv[1] if len(sys.argv) > 1 else None
        email = sys.argv[2] if len(sys.argv) > 2 else None
        language = sys.argv[3] if len(sys.argv) > 3 else None
        
        trigger_transcription(video, email, language)
    else:
        trigger_transcription()