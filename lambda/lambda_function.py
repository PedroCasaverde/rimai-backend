import json
import boto3
import os

# --- CONFIGURACIÓN ---
BATCH_JOB_QUEUE = 'Rimai-Queue'
BATCH_JOB_DEFINITION = 'Rimai-Whisper-Job'
# ¡ASEGÚRATE DE QUE ESTOS NOMBRES SEAN LOS TUYOS!
INPUT_BUCKET_NAME = 'rimai-input-pibot'  # <--- Revisa si es correcto
OUTPUT_BUCKET_NAME = 'rimai-output-pibot' # <--- Revisa si es correcto
CONTAINER_NAME = 'rimai-container' # <--- Este es el nombre que pusimos en la configuración
# ---------------------

batch = boto3.client('batch')

def lambda_handler(event, context):
    try:
        # 1. Recibir datos
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

        print(f"🎬 Recibido trabajo para: {file_name}")

        # 2. Enviar orden a AWS Batch (MODO NUEVO ECS)
        response = batch.submit_job(
            jobName=f'transcribe-{file_name.replace(".", "-")}',
            jobQueue=BATCH_JOB_QUEUE,
            jobDefinition=BATCH_JOB_DEFINITION,
            # Aquí está el cambio clave para corregir tu error:
            ecsPropertiesOverride={
                'taskProperties': [
                    {
                        'containers': [
                            {
                                'name': CONTAINER_NAME,
                                'environment': [
                                    {'name': 'HF_TOKEN', 'value': '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'},
                                    {'name': 'S3_INPUT_BUCKET', 'value': INPUT_BUCKET_NAME},
                                    {'name': 'S3_OUTPUT_BUCKET', 'value': OUTPUT_BUCKET_NAME},
                                    {'name': 'FILE_NAME', 'value': file_name},
                                    {'name': 'USER_EMAIL', 'value': user_email}
                                ]
                            }
                        ]
                    }
                ]
            }
        )

        print(f"✅ Trabajo enviado! Job ID: {response['jobId']}")

        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'Trabajo iniciado', 'jobId': response['jobId']})
        }

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"Error interno: {str(e)}")
        }