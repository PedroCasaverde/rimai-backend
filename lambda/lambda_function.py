import json
import boto3
import os

BATCH_JOB_QUEUE = 'Rimai-Queue'
BATCH_JOB_DEFINITION = 'Rimai-Whisper-Job'
INPUT_BUCKET_NAME = 'rimai-input-pibot'
OUTPUT_BUCKET_NAME = 'rimai-output-pibot'
CONTAINER_NAME = 'rimai-container'
HF_TOKEN = os.environ.get('HF_TOKEN', '')

batch = boto3.client('batch')

def _validate_secret(event):
    """Valida que el request viene de pibot-web"""
    expected = os.environ.get('PIBOT_SECRET', '')
    if not expected:
        return True  # Si no está configurado, no bloquear (dev)
    headers = event.get('headers', {})
    # Lambda Function URL envía headers en minúsculas
    received = headers.get('x-pibot-secret', '')
    return received == expected


def _validate_zoom_url(url):
    """Valida que la URL sea de zoom.us"""
    import re
    if not url or not re.match(r'https?://[\w.-]*zoom\.us/', url):
        return False
    return True


def _analyze_zoom_url(body):
    """Analiza una URL de Zoom sin descargar el video (solo metadatos)"""
    import yt_dlp

    zoom_url = body.get('zoom_url', '')

    if not zoom_url:
        return {
            'statusCode': 400,
            'body': json.dumps({'valid': False, 'error': 'Falta zoom_url'})
        }

    if not _validate_zoom_url(zoom_url):
        return {
            'statusCode': 400,
            'body': json.dumps({'valid': False, 'error': 'El enlace no es una grabación de Zoom válida'})
        }

    print(f"🔍 Analizando URL de Zoom: {zoom_url[:80]}...")

    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
        'socket_timeout': 15,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(zoom_url, download=False)

            duration = info.get('duration', 0)
            if not duration:
                return {
                    'statusCode': 200,
                    'body': json.dumps({
                        'valid': False,
                        'error': 'No se pudo obtener la duración del video. Verifica que el enlace sea correcto.'
                    })
                }

            filesize = info.get('filesize') or info.get('filesize_approx') or 0
            size_mb = round(filesize / (1024 * 1024), 1) if filesize else None

            result = {
                'valid': True,
                'duration_seconds': duration,
                'title': info.get('title', ''),
            }
            if size_mb:
                result['size_mb'] = size_mb

            print(f"✅ URL válida: {duration}s, {result.get('title', 'sin título')}")
            return {
                'statusCode': 200,
                'body': json.dumps(result)
            }

    except yt_dlp.utils.DownloadError as e:
        error_msg = str(e).lower()
        if any(kw in error_msg for kw in ['login', 'password', 'sign in', 'authenticate', 'private']):
            msg = 'La grabación requiere login o contraseña. Descarga el video manualmente y súbelo.'
        else:
            msg = 'No se pudo acceder a la grabación. Verifica que el enlace sea correcto y público.'
        print(f"❌ Error analizando URL: {msg}")
        return {
            'statusCode': 200,
            'body': json.dumps({'valid': False, 'error': msg})
        }
    except Exception as e:
        print(f"❌ Error inesperado analizando URL: {e}")
        return {
            'statusCode': 200,
            'body': json.dumps({
                'valid': False,
                'error': 'No se pudo acceder a la grabación. Verifica que el enlace sea correcto y público.'
            })
        }


def _submit_transcription(body):
    """Procesa un job de transcripción simple (flujo original)"""
    file_name = body.get('fileName') or body.get('s3_key')
    zoom_url = body.get('zoom_url')
    user_email = body.get('email', '')
    language = body.get('language', 'auto')
    speakers = body.get('speakers')
    callback_url = body.get('callback_url', '')

    if not file_name and not zoom_url:
        return {
            'statusCode': 400,
            'body': json.dumps('Error: Falta fileName, s3_key o zoom_url')
        }

    if zoom_url and not _validate_zoom_url(zoom_url):
        return {
            'statusCode': 400,
            'body': json.dumps('Error: El enlace no es una grabación de Zoom válida')
        }

    source = zoom_url or file_name
    msg_speakers = f"speakers: {speakers}" if speakers else "speakers: AUTO"
    print(f"🎬 Trabajo transcripción: {source} ({language}, {msg_speakers})")

    env_vars = [
        {'name': 'S3_INPUT_BUCKET', 'value': INPUT_BUCKET_NAME},
        {'name': 'S3_OUTPUT_BUCKET', 'value': OUTPUT_BUCKET_NAME},
        {'name': 'FILE_NAME', 'value': file_name or ''},
        {'name': 'USER_EMAIL', 'value': user_email},
        {'name': 'LANGUAGE', 'value': language},
        {'name': 'HF_TOKEN', 'value': HF_TOKEN},
        {'name': 'JOB_TYPE', 'value': 'transcription'},
    ]

    if zoom_url:
        env_vars.append({'name': 'ZOOM_URL', 'value': zoom_url})

    if speakers:
        env_vars.append({'name': 'NUM_SPEAKERS', 'value': str(speakers)})
    if callback_url:
        env_vars.append({'name': 'CALLBACK_URL', 'value': callback_url})

    job_name = f'transcribe-zoom' if zoom_url else f'transcribe-{file_name.replace(".", "-")}'
    response = batch.submit_job(
        jobName=job_name,
        jobQueue=BATCH_JOB_QUEUE,
        jobDefinition=BATCH_JOB_DEFINITION,
        timeout={'attemptDurationSeconds': 7200},
        ecsPropertiesOverride={
            'taskProperties': [{
                'containers': [{
                    'name': CONTAINER_NAME,
                    'environment': env_vars
                }]
            }]
        }
    )

    print(f"✅ Job ID: {response['jobId']}")

    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': 'Trabajo iniciado',
            'jobId': response['jobId'],
            'job_type': 'transcription',
            'language': language,
            'speakers': speakers or 'auto'
        })
    }


def _submit_mentoria(body):
    """Procesa un job de MentorIA"""
    job_config = body.get('job_config', {})
    callback_url = body.get('callback_url', '')
    language = body.get('language', 'auto')
    speakers = body.get('speakers')

    video_keys = job_config.get('video_keys', [])
    zoom_urls = job_config.get('zoom_urls', [])
    pdf_keys = job_config.get('pdf_keys', [])
    outputs = job_config.get('outputs', [])

    if not video_keys and not zoom_urls and not pdf_keys:
        return {
            'statusCode': 400,
            'body': json.dumps('Error: MentorIA requiere al menos video_keys, zoom_urls o pdf_keys')
        }

    # Validar todas las URLs de Zoom
    for url in zoom_urls:
        if not _validate_zoom_url(url):
            return {
                'statusCode': 400,
                'body': json.dumps(f'Error: El enlace no es una grabación de Zoom válida: {url}')
            }

    if not outputs:
        return {
            'statusCode': 400,
            'body': json.dumps('Error: MentorIA requiere al menos un output (resumen, guia, banco_preguntas, alerta_examen, analogias)')
        }

    analogia_contexto = (job_config.get('analogia_contexto') or '')[:30].strip()

    disciplina = job_config.get('disciplina', '')
    usar_analogias = job_config.get('usar_analogias', False)
    print(f"🧠 Trabajo MentorIA: {len(video_keys)} videos, {len(zoom_urls)} zoom, {len(pdf_keys)} PDFs, outputs: {outputs}, disciplina: {disciplina or 'N/A'}, analogías: {usar_analogias}, contexto_analogía: {analogia_contexto or 'N/A'}")

    # Serializar job_config como JSON string para pasarlo como env var
    env_vars = [
        {'name': 'S3_INPUT_BUCKET', 'value': INPUT_BUCKET_NAME},
        {'name': 'S3_OUTPUT_BUCKET', 'value': OUTPUT_BUCKET_NAME},
        {'name': 'JOB_TYPE', 'value': 'mentoria'},
        {'name': 'JOB_CONFIG', 'value': json.dumps(job_config)},
        {'name': 'LANGUAGE', 'value': language},
        {'name': 'HF_TOKEN', 'value': HF_TOKEN},
        {'name': 'OPENAI_API_KEY', 'value': os.environ.get('OPENAI_API_KEY', '')},
    ]

    if speakers:
        env_vars.append({'name': 'NUM_SPEAKERS', 'value': str(speakers)})
    if callback_url:
        env_vars.append({'name': 'CALLBACK_URL', 'value': callback_url})

    # Generar un job name descriptivo
    job_name = f'mentoria-{len(video_keys)}v-{len(pdf_keys)}p'

    response = batch.submit_job(
        jobName=job_name,
        jobQueue=BATCH_JOB_QUEUE,
        jobDefinition=BATCH_JOB_DEFINITION,
        timeout={'attemptDurationSeconds': 7200},
        ecsPropertiesOverride={
            'taskProperties': [{
                'containers': [{
                    'name': CONTAINER_NAME,
                    'environment': env_vars
                }]
            }]
        }
    )

    job_id = response['jobId']
    print(f"✅ Job ID: {job_id}")

    # Agregar JOB_ID al env (se usa dentro del container para nombrar archivos)
    # Nota: ya fue enviado, pero el container puede usar AWS_BATCH_JOB_ID como fallback

    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': 'Trabajo MentorIA iniciado',
            'jobId': job_id,
            'job_type': 'mentoria',
            'status': 'queued'
        })
    }


def _get_request_path(event):
    """Extrae el path de la request (Lambda Function URL o API Gateway)"""
    # Lambda Function URL
    raw_path = event.get('rawPath', '')
    if raw_path:
        return raw_path
    # API Gateway v1
    path = event.get('path', '')
    if path:
        return path
    # Fallback
    return '/'


def lambda_handler(event, context):
    try:
        # Validar autenticación
        if not _validate_secret(event):
            return {
                'statusCode': 401,
                'body': json.dumps({'error': 'No autorizado'})
            }

        path = _get_request_path(event)

        if 'body' in event:
            body = json.loads(event['body'])
        else:
            body = event

        # Routing por path
        if path.rstrip('/').endswith('/analyze-url'):
            return _analyze_zoom_url(body)

        job_type = body.get('job_type', 'transcription')

        # Compatibilidad: si viene fileName sin job_type, es transcripción
        if 'fileName' in body and 'job_type' not in body:
            job_type = 'transcription'

        if job_type == 'mentoria':
            return _submit_mentoria(body)
        else:
            return _submit_transcription(body)

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"Error interno: {str(e)}")
        }