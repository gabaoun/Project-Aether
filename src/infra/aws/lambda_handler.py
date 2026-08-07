import asyncio
import os
import urllib.parse

import boto3

from src.pipeline.ingestion import IngestionWorkflow
from src.utils.logger import logger

# Inicializa o cliente do S3 globalmente para reaproveitamento entre invocações a quente
s3_client = boto3.client('s3')

def handler(event, context):
    """
    Entrypoint do AWS Lambda.
    Acordado automaticamente sempre que um arquivo é feito upload no bucket S3 configurado.
    """
    logger.info("Lambda invocation started via S3 event.")
    
    # 1. Pega os dados do arquivo que gerou o evento
    try:
        record = event['Records'][0]
        bucket_name = record['s3']['bucket']['name']
        object_key = urllib.parse.unquote_plus(record['s3']['object']['key'])
        logger.info(f"Triggered by file {object_key} in bucket {bucket_name}")
    except KeyError:
        logger.error("Invalid event format.")
        return {"statusCode": 400, "body": "Invalid event format."}

    # 2. Baixa o arquivo para a pasta temporária do Lambda (/tmp)
    # A AWS garante 512MB até 10GB no /tmp para operações efêmeras.
    tmp_dir = "/tmp/ingestion_workspace"
    os.makedirs(tmp_dir, exist_ok=True)
    
    file_path = os.path.join(tmp_dir, os.path.basename(object_key))
    
    try:
        logger.info(f"Downloading {object_key} to {file_path}")
        s3_client.download_file(bucket_name, object_key, file_path)
    except Exception as e:  # noqa: BLE001 - Lambda handler boundary: must return a response, never crash
        logger.error(f"Error downloading file from S3: {e}")
        return {"statusCode": 500, "body": "S3 download failed."}

    # 3. Executa o pipeline de Ingestão Original do Project Aether
    # O pipeline não foi alterado, ele apenas vai ler da pasta temporária agora.
    try:
        logger.info("Starting Ingestion Workflow...")
        workflow = IngestionWorkflow()
        
        async def _run_workflow() -> None:
            # Passamos a pasta /tmp como input_dir
            await workflow.run(input_dir=tmp_dir)

        # Roda o pipeline assíncrono no loop do Lambda
        asyncio.run(_run_workflow())
        
        logger.info(f"Workflow completed successfully for {object_key}")
        
    except Exception as e:  # noqa: BLE001 - Lambda handler boundary: must return a response, never crash
        logger.error(f"Ingestion workflow failed: {e}")
        return {"statusCode": 500, "body": "Workflow execution failed."}
    
    # Limpa o arquivo da memória efêmera
    os.remove(file_path)
    
    return {
        "statusCode": 200,
        "body": f"Successfully ingested {object_key} into Chroma Cloud."
    }
