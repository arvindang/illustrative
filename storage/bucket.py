"""S3-compatible storage for Railway bucket."""
import os
from pathlib import Path
from typing import Optional
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


class BucketStorage:
    """S3-compatible storage client for Railway bucket."""

    def __init__(self):
        """Initialize S3 client with Railway bucket credentials."""
        self.endpoint = os.getenv("BUCKET_ENDPOINT", "https://storage.railway.app")
        self.bucket_name = os.getenv("BUCKET")
        self.access_key = os.getenv("BUCKET_ACCESS_KEY_ID")
        self.secret_key = os.getenv("BUCKET_SECRET_ACCESS_KEY")
        self.region = os.getenv("BUCKET_REGION", "auto")

        self._client = None

    @property
    def client(self):
        """Lazy-load S3 client."""
        if self._client is None:
            if not all([self.bucket_name, self.access_key, self.secret_key]):
                raise RuntimeError(
                    "Bucket not configured. Set BUCKET, BUCKET_ACCESS_KEY_ID, "
                    "and BUCKET_SECRET_ACCESS_KEY environment variables."
                )
            self._client = boto3.client(
                "s3",
                endpoint_url=self.endpoint,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name=self.region,
                config=Config(signature_version="s3v4"),
            )
        return self._client

    def is_configured(self) -> bool:
        """Check if bucket is properly configured."""
        return all([self.bucket_name, self.access_key, self.secret_key])

    def upload_file(self, file_path: str, storage_key: str, content_type: Optional[str] = None) -> str:
        """
        Upload a file to the bucket.

        Args:
            file_path: Local path to file
            storage_key: Key (path) in the bucket
            content_type: Optional MIME type

        Returns:
            The storage key
        """
        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type

        self.client.upload_file(
            str(file_path),
            self.bucket_name,
            storage_key,
            ExtraArgs=extra_args if extra_args else None,
        )
        return storage_key

    def upload_bytes(self, data: bytes, storage_key: str, content_type: Optional[str] = None) -> str:
        """
        Upload bytes to the bucket.

        Args:
            data: Bytes to upload
            storage_key: Key (path) in the bucket
            content_type: Optional MIME type

        Returns:
            The storage key
        """
        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type

        self.client.put_object(
            Bucket=self.bucket_name,
            Key=storage_key,
            Body=data,
            **extra_args,
        )
        return storage_key

    def download_file(self, storage_key: str, local_path: str) -> Path:
        """
        Download a file from the bucket.

        Args:
            storage_key: Key in the bucket
            local_path: Local path to save file

        Returns:
            Path to downloaded file
        """
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket_name, storage_key, str(local_path))
        return local_path

    def generate_presigned_url(self, storage_key: str, expiration: int = 3600) -> str:
        """
        Generate a presigned URL for downloading a file.

        Args:
            storage_key: Key in the bucket
            expiration: URL expiration time in seconds (default: 1 hour)

        Returns:
            Presigned download URL
        """
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket_name, "Key": storage_key},
            ExpiresIn=expiration,
        )

    def delete_file(self, storage_key: str) -> None:
        """
        Delete a file from the bucket.

        Args:
            storage_key: Key in the bucket
        """
        self.client.delete_object(Bucket=self.bucket_name, Key=storage_key)

    def delete_files(self, storage_keys: list[str]) -> None:
        """
        Delete multiple files from the bucket.

        Args:
            storage_keys: List of keys to delete
        """
        if not storage_keys:
            return

        objects = [{"Key": key} for key in storage_keys]
        self.client.delete_objects(
            Bucket=self.bucket_name,
            Delete={"Objects": objects},
        )

    def file_exists(self, storage_key: str) -> bool:
        """
        Check if a file exists in the bucket.

        Args:
            storage_key: Key in the bucket

        Returns:
            True if file exists
        """
        try:
            self.client.head_object(Bucket=self.bucket_name, Key=storage_key)
            return True
        except ClientError:
            return False


# Singleton instance
_storage: Optional[BucketStorage] = None


def get_storage() -> BucketStorage:
    """Get the global storage instance."""
    global _storage
    if _storage is None:
        _storage = BucketStorage()
    return _storage
