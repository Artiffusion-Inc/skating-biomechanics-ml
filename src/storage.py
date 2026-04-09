"""S3-compatible object storage (Cloudflare R2) for video transfer."""

from __future__ import annotations

import logging
from pathlib import Path

import boto3
from botocore.config import Config as BotoConfig

from src.config import get_settings

logger = logging.getLogger(__name__)


def _client():
    s = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=s.cf_r2_endpoint_url,
        aws_access_key_id=s.cf_r2_access_key_id,
        aws_secret_access_key=s.cf_r2_secret_access_key,
        config=BotoConfig(signature_version="s3v4"),
        region_name="auto",
    )


def upload_file(local_path: str | Path, key: str) -> str:
    """Upload file to R2. Returns the key."""
    logger.info("Uploading %s -> s3://%s/%s", local_path, get_settings().cf_r2_bucket, key)
    _client().upload_file(str(local_path), get_settings().cf_r2_bucket, key)
    return key


def download_file(key: str, local_path: str | Path) -> str:
    """Download file from R2. Returns the local path."""
    logger.info("Downloading s3://%s/%s -> %s", get_settings().cf_r2_bucket, key, local_path)
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    _client().download_file(get_settings().cf_r2_bucket, key, str(local_path))
    return str(local_path)


def delete_object(key: str) -> None:
    """Delete object from R2."""
    _client().delete_object(Bucket=get_settings().cf_r2_bucket, Key=key)
