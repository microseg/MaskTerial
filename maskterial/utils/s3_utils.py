"""
S3 utilities for MaskTerial
Handles image upload and deletion from S3
"""

import os
from typing import Optional

try:
    import boto3
    from botocore.exceptions import ClientError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    print("Warning: boto3 not installed. S3 features will not be available.")


def get_s3_client():
    """
    Get S3 client
    Configure AWS credentials via environment variables or AWS config
    """
    if not S3_AVAILABLE:
        raise ImportError("boto3 is required for S3 operations. Install with: pip install boto3")
    
    region = os.environ.get('AWS_REGION', 'us-east-1')
    
    s3_client = boto3.client(
        's3',
        region_name=region,
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
    )
    
    return s3_client


def delete_s3_object(bucket_name: str, s3_key: str) -> bool:
    """
    Delete an object from S3
    
    Args:
        bucket_name: S3 bucket name
        s3_key: S3 object key (path)
    
    Returns:
        Boolean indicating success
    """
    if not S3_AVAILABLE:
        print("S3 not available, skipping delete")
        return False
    
    try:
        s3_client = get_s3_client()
        
        # Delete the object
        s3_client.delete_object(
            Bucket=bucket_name,
            Key=s3_key
        )
        
        print(f"Successfully deleted s3://{bucket_name}/{s3_key}")
        return True
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        print(f"S3 ClientError: {error_code} - {error_message}")
        return False
    
    except Exception as e:
        print(f"Error deleting from S3: {str(e)}")
        return False


def delete_s3_objects(bucket_name: str, s3_keys: list) -> dict:
    """
    Delete multiple objects from S3
    
    Args:
        bucket_name: S3 bucket name
        s3_keys: List of S3 object keys
    
    Returns:
        Dictionary with success count and errors
    """
    if not S3_AVAILABLE:
        return {"success": 0, "errors": ["S3 not available"]}
    
    if not s3_keys:
        return {"success": 0, "errors": []}
    
    try:
        s3_client = get_s3_client()
        
        # Prepare objects for batch delete
        objects = [{'Key': key} for key in s3_keys]
        
        # Delete objects
        response = s3_client.delete_objects(
            Bucket=bucket_name,
            Delete={
                'Objects': objects,
                'Quiet': False
            }
        )
        
        deleted = response.get('Deleted', [])
        errors = response.get('Errors', [])
        
        print(f"Successfully deleted {len(deleted)} objects from S3")
        if errors:
            print(f"Failed to delete {len(errors)} objects")
        
        return {
            "success": len(deleted),
            "errors": errors
        }
        
    except Exception as e:
        print(f"Error deleting from S3: {str(e)}")
        return {"success": 0, "errors": [str(e)]}


def get_s3_bucket_from_url(image_url: str) -> Optional[tuple]:
    """
    Extract bucket name and key from S3 URL
    
    Args:
        image_url: S3 URL (e.g., https://bucket.s3.region.amazonaws.com/key or s3://bucket/key)
    
    Returns:
        Tuple of (bucket_name, key) or None if not a valid S3 URL
    """
    if not image_url:
        return None
    
    # Handle s3:// URLs
    if image_url.startswith('s3://'):
        parts = image_url[5:].split('/', 1)
        if len(parts) == 2:
            return (parts[0], parts[1])
    
    # Handle https:// URLs
    if 's3.amazonaws.com' in image_url or 's3.' in image_url:
        try:
            from urllib.parse import urlparse
            parsed = urlparse(image_url)
            
            # Format: https://bucket.s3.region.amazonaws.com/key
            if parsed.hostname:
                bucket = parsed.hostname.split('.')[0]
                key = parsed.path.lstrip('/')
                return (bucket, key)
        except Exception as e:
            print(f"Error parsing S3 URL: {str(e)}")
    
    return None


def check_s3_object_exists(bucket_name: str, s3_key: str) -> bool:
    """
    Check if an S3 object exists
    
    Args:
        bucket_name: S3 bucket name
        s3_key: S3 object key
    
    Returns:
        Boolean indicating if object exists
    """
    if not S3_AVAILABLE:
        return False
    
    try:
        s3_client = get_s3_client()
        s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        raise
    except Exception:
        return False


def generate_presigned_download_url(
    bucket_name: str,
    s3_key: str,
    expiration: int = 86400,  # 24 hours default
    custom_filename: str = None
) -> Optional[str]:
    """
    Generate a presigned URL for downloading an S3 object
    
    Args:
        bucket_name: S3 bucket name
        s3_key: S3 object key
        expiration: URL expiration time in seconds (default: 86400 = 24 hours)
        custom_filename: Optional custom filename for download
    
    Returns:
        Presigned URL string or None if failed
    """
    if not S3_AVAILABLE:
        print("S3 not available, cannot generate presigned URL")
        return None
    
    try:
        s3_client = get_s3_client()
        
        # Prepare parameters
        params = {
            'Bucket': bucket_name,
            'Key': s3_key
        }
        
        # Add custom filename for download if provided
        if custom_filename:
            params['ResponseContentDisposition'] = f'attachment; filename="{custom_filename}"'
        
        # Generate presigned URL
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params=params,
            ExpiresIn=expiration
        )
        
        print(f"Generated presigned URL for s3://{bucket_name}/{s3_key} (expires in {expiration}s)")
        return presigned_url
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        print(f"S3 ClientError generating presigned URL: {error_code} - {error_message}")
        return None
    
    except Exception as e:
        print(f"Error generating presigned URL: {str(e)}")
        return None


def generate_public_url(bucket_name: str, s3_key: str, region: str = None) -> str:
    """
    Generate a public S3 URL (works only if bucket/object is public)
    
    Args:
        bucket_name: S3 bucket name
        s3_key: S3 object key
        region: AWS region (default from env)
    
    Returns:
        Public S3 URL
    """
    if region is None:
        region = os.environ.get('AWS_REGION', 'us-east-1')
    
    # Generate URL in format: https://bucket.s3.region.amazonaws.com/key
    return f"https://{bucket_name}.s3.{region}.amazonaws.com/{s3_key}"

