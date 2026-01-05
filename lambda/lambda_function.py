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

def lambda_handler(event, context):
    try:
        if 'body' in event:
            body = json.loads(event['body'])
        else:
            body = event

        file_name = body.get('fileName')
        user_email = body.get('email')
        language = body.get('language', 'auto')
        # NUEVO: Recibimos speakers (puede ser None)
        speakers = body.get('speakers') 

        if not file_name or not user_email:
            return {
                'statusCode': 400,
                'body': json.dumps('Error: Faltan datos (fileName o email)')
            }

        # Mensaje de log informativo
        msg_speakers = f"speakers: {speakers}" if speakers else "speakers: AUTO"
        print(f"🎬 Trabajo recibido: {file_name} ({language}, {msg_speakers})")

        # Preparamos las variables de entorno
        env_vars = [
            {'name': 'S3_INPUT_BUCKET', 'value': INPUT_BUCKET_NAME},
            {'name': 'S3_OUTPUT_BUCKET', 'value': OUTPUT_BUCKET_NAME},
            {'name': 'FILE_NAME', 'value': file_name},
            {'name': 'USER_EMAIL', 'value': user_email},
            {'name': 'LANGUAGE', 'value': language},
            {'name': 'HF_TOKEN', 'value': HF_TOKEN}
        ]

        # Solo agregamos la variable NUM_SPEAKERS si el usuario envió un número
        if speakers:
             env_vars.append({'name': 'NUM_SPEAKERS', 'value': str(speakers)})

        response = batch.submit_job(
            jobName=f'transcribe-{file_name.replace(".", "-")}',
            jobQueue=BATCH_JOB_QUEUE,
            jobDefinition=BATCH_JOB_DEFINITION,
            timeout={'attemptDurationSeconds': 7200},
            ecsPropertiesOverride={
                'taskProperties': [{
                    'containers': [{
                        'name': CONTAINER_NAME,
                        'environment': env_vars # Usamos la lista que creamos arriba
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
                'language': language,
                'speakers': speakers or 'auto'
            })
        }

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"Error interno: {str(e)}")
        }