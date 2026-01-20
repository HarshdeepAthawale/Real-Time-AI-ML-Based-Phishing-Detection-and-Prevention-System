import boto3
from typing import Optional, Dict
import os
from botocore.exceptions import ClientError
import logging

logger = logging.getLogger(__name__)

class S3Uploader:
    """Utility class for uploading images to S3"""
    
    def __init__(
        self,
        bucket_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: str = 'us-east-1'
    ):
        self.bucket_name = bucket_name or os.getenv('S3_BUCKET_NAME')
        self.region_name = region_name or os.getenv('AWS_REGION', 'us-east-1')
        
        # Initialize S3 client
        if aws_access_key_id and aws_secret_access_key:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=self.region_name
            )
        else:
            # Use default credentials (from environment or IAM role)
            self.s3_client = boto3.client('s3', region_name=self.region_name)
    
    def upload_image(
        self,
        image_bytes: bytes,
        key: str,
        content_type: str = 'image/png',
        metadata: Optional[Dict] = None
    ) -> Optional[str]:
        """Upload image to S3"""
        if not self.bucket_name:
            logger.warning("S3 bucket name not configured. Skipping upload.")
            return None
        
        try:
            extra_args = {
                'ContentType': content_type
            }
            
            if metadata:
                extra_args['Metadata'] = {str(k): str(v) for k, v in metadata.items()}
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=image_bytes,
                **extra_args
            )
            
            # Generate URL
            url = f"https://{self.bucket_name}.s3.{self.region_name}.amazonaws.com/{key}"
            logger.info(f"Successfully uploaded image to S3: {url}")
            return url
            
        except ClientError as e:
            logger.error(f"Failed to upload image to S3: {e}")
            return None
    
    def upload_screenshot(
        self,
        screenshot_bytes: bytes,
        url: str,
        analysis_id: Optional[str] = None
    ) -> Optional[str]:
        """Upload screenshot with organized naming"""
        import hashlib
        from datetime import datetime
        
        # Generate key from URL and timestamp
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        
        if analysis_id:
            key = f"screenshots/{analysis_id}/{url_hash}_{timestamp}.png"
        else:
            key = f"screenshots/{url_hash}_{timestamp}.png"
        
        metadata = {
            'url': url,
            'uploaded_at': timestamp
        }
        
        if analysis_id:
            metadata['analysis_id'] = analysis_id
        
        return self.upload_image(screenshot_bytes, key, metadata=metadata)
    
    def delete_image(self, key: str) -> bool:
        """Delete image from S3"""
        if not self.bucket_name:
            return False
        
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=key)
            logger.info(f"Successfully deleted image from S3: {key}")
            return True
        except ClientError as e:
            logger.error(f"Failed to delete image from S3: {e}")
            return False
