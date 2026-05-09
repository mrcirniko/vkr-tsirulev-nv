from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.client import Config
from config import settings

LOGGER = logging.getLogger("storage")
DOCX_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def _client():
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        aws_access_key_id=settings.s3_access_key_id,
        aws_secret_access_key=settings.s3_secret_access_key,
        region_name=settings.s3_region,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def ensure_bucket() -> None:
    if not settings.s3_enabled:
        return
    client = _client()
    try:
        client.head_bucket(Bucket=settings.s3_bucket)
    except Exception:
        client.create_bucket(Bucket=settings.s3_bucket)
        LOGGER.info("Created S3 bucket=%s", settings.s3_bucket)


def object_key(case_id: str, version_number: int) -> str:
    return f"contracts/{case_id}/v{version_number}.docx"


def upload_contract_docx(path: str | Path, case_id: str, version_number: int) -> str:
    ensure_bucket()
    key = object_key(case_id, version_number)
    _client().upload_file(
        str(path),
        settings.s3_bucket,
        key,
        ExtraArgs={"ContentType": DOCX_CONTENT_TYPE},
    )
    uri = f"s3://{settings.s3_bucket}/{key}"
    LOGGER.info("Uploaded contract DOCX to %s", uri)
    return uri


def is_s3_uri(value: str | None) -> bool:
    return bool(value and value.startswith("s3://"))


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def download_object(uri: str) -> bytes:
    bucket, key = parse_s3_uri(uri)
    response = _client().get_object(Bucket=bucket, Key=key)
    return response["Body"].read()


def presigned_url(uri: str) -> str:
    bucket, key = parse_s3_uri(uri)
    url = _client().generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=settings.s3_presign_expires_seconds,
    )
    if settings.s3_endpoint_url != settings.s3_public_endpoint_url:
        url = url.replace(settings.s3_endpoint_url.rstrip("/"), settings.s3_public_endpoint_url.rstrip("/"), 1)
    return url
