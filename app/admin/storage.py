"""S3/MinIO storage helpers for the admin-npa bucket.

Separate bucket from `contracts` so admin uploads have their own lifecycle
(retention, access policy) and we can wipe it without touching user data.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import boto3
from botocore.client import Config
from config import settings

LOGGER = logging.getLogger("app.admin.storage")


def _client():
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        aws_access_key_id=settings.s3_access_key_id,
        aws_secret_access_key=settings.s3_secret_access_key,
        region_name=settings.s3_region,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def ensure_admin_bucket() -> None:
    client = _client()
    try:
        client.head_bucket(Bucket=settings.admin_npa_bucket)
    except Exception:
        client.create_bucket(Bucket=settings.admin_npa_bucket)
        LOGGER.info("Created admin NPA bucket=%s", settings.admin_npa_bucket)


def upload_bytes(key: str, data: bytes, content_type: str) -> str:
    ensure_admin_bucket()
    _client().put_object(
        Bucket=settings.admin_npa_bucket,
        Key=key,
        Body=data,
        ContentType=content_type,
    )
    return key


def upload_text(key: str, text: str, content_type: str = "text/plain; charset=utf-8") -> str:
    return upload_bytes(key, text.encode("utf-8"), content_type)


def download_bytes(key: str) -> bytes:
    response = _client().get_object(Bucket=settings.admin_npa_bucket, Key=key)
    return response["Body"].read()


def download_text(key: str) -> str:
    return download_bytes(key).decode("utf-8")


def delete_objects(keys: Iterable[str]) -> None:
    keys_list = [k for k in keys if k]
    if not keys_list:
        return
    client = _client()
    # S3 DeleteObjects caps at 1000 keys per request — one batch is always enough at our scale.
    client.delete_objects(
        Bucket=settings.admin_npa_bucket,
        Delete={"Objects": [{"Key": k} for k in keys_list], "Quiet": True},
    )


def raw_txt_key(npa_id: str) -> str:
    return f"npa/{npa_id}/raw.txt"


def chunks_json_key(npa_id: str) -> str:
    return f"npa/{npa_id}/chunks.json"
