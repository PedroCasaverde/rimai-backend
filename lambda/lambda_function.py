import json
import boto3
import os

# --- CONFIGURACIÓN (Ajusta esto si tus nombres son diferentes) ---
BATCH_JOB_QUEUE = 'Rimai-Queue'
BATCH_JOB_DEFINITION = 'Rimai-Whisper-Job'
INPUT_BUCKET_NAME = 'rimai-input-pibot' # <--- ¡CAMBIA ESTO POR TU NOMBRE REAL!
OUTPUT_BUCKET_NAME = 'rimai-output-pibot' # <--- ¡CAMBIA ESTO POR TU NOMBRE REAL!
# ---------------------------------------------------------------

batch = boto3.client('batch')

def lambda_handler(event, context):
    try:
        # 1. Recibir datos desde la Web (Frontend)
        # Si viene desde API Gateway, el cuerpo suele venir en 'body' como string
        if 'body' in event:
            body = json.loads(event['body'])
        else:
            body = event

        file_name = body.get('fileName')
        user_email = body.get('email')

        if not file_name or not user_email:
            return {
                'statusCode': 400,
                'body': json.dumps('Error: Faltan datos (fileName o email)')
            }

        print(f"🎬 Recibido trabajo para: {file_name}, usuario: {user_email}")

        # 2. Enviar la orden a AWS Batch (Encender la GPU)
        response = batch.submit_job(
            jobName=f'transcribe-{file_name.replace(".", "-")}',
            jobQueue=BATCH_JOB_QUEUE,
            jobDefinition=BATCH_JOB_DEFINITION,
            containerOverrides={
                'environment': [
                    {'name': 'S3_INPUT_BUCKET', 'value': INPUT_BUCKET_NAME},
                    {'name': 'S3_OUTPUT_BUCKET', 'value': OUTPUT_BUCKET_NAME},
                    {'name': 'FILE_NAME', 'value': file_name},
                    {'name': 'USER_EMAIL', 'value': user_email}
                ]
            }
        )

        print(f"✅ Trabajo enviado a Batch! Job ID: {response['jobId']}")

        return {
            'statusCode': 200,
            'headers': {
                "Access-Control-Allow-Origin": "*", # Importante para que funcione desde cualquier web
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            'body': json.dumps({
                'message': 'Trabajo iniciado correctamente',
                'jobId': response['jobId']
            })
        }

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"Error interno: {str(e)}")
        }