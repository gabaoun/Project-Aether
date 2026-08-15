"""
HTTP-triggerable variant of the ingestion Lambda, fronted by API Gateway
(see infra/terraform/main.tf). Distinct from lambda_handler.py, which is
wired to the S3 upload event (../../../template.yaml, AWS SAM) and expects
that event shape - this one expects an API Gateway HTTP API v2 payload and
lets an authenticated caller trigger ingestion for a specific S3 object
on demand, without waiting for a new upload.
"""

import asyncio
import json
import os

import boto3

from src.pipeline.ingestion import IngestionWorkflow
from src.utils.logger import logger

s3_client = boto3.client("s3")


def handler(event, context):
    """
    API Gateway HTTP API (v2) entrypoint. Expects a JSON body:
    {"bucket": "<s3-bucket>", "key": "<object-key>"}
    """
    try:
        body = json.loads(event.get("body") or "{}")
        bucket_name = body["bucket"]
        object_key = body["key"]
    except (KeyError, json.JSONDecodeError):
        return {"statusCode": 400, "body": json.dumps({"error": "Expected JSON body with bucket and key"})}

    tmp_dir = "/tmp/ingestion_workspace"
    file_path = f"{tmp_dir}/{object_key.rsplit('/', 1)[-1]}"

    try:
        os.makedirs(tmp_dir, exist_ok=True)
        logger.info(f"[API_TRIGGER] Downloading {object_key} from {bucket_name}")
        s3_client.download_file(bucket_name, object_key, file_path)
    except Exception as e:  # noqa: BLE001 - Lambda handler boundary: must return a response, never crash
        logger.error(f"[API_TRIGGER] S3 download failed: {e}")
        return {"statusCode": 500, "body": json.dumps({"error": "S3 download failed"})}

    try:
        workflow = IngestionWorkflow()
        asyncio.run(workflow.run(input_dir=tmp_dir))
    except Exception as e:  # noqa: BLE001 - Lambda handler boundary: must return a response, never crash
        logger.error(f"[API_TRIGGER] Ingestion workflow failed: {e}")
        return {"statusCode": 500, "body": json.dumps({"error": "Ingestion workflow failed"})}

    return {"statusCode": 200, "body": json.dumps({"message": f"Ingested {object_key}"})}
