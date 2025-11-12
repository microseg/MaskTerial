"""
DynamoDB utilities for MaskTerial
Handles image metadata storage and retrieval
"""

import json
import os
from datetime import datetime
from typing import Optional, Dict, List, Any
from decimal import Decimal


try:
    import boto3
    from boto3.dynamodb.conditions import Key, Attr
    from botocore.exceptions import ClientError
    DYNAMODB_AVAILABLE = True
except ImportError:
    DYNAMODB_AVAILABLE = False
    print("Warning: boto3 not installed. DynamoDB features will not be available.")


class DecimalEncoder(json.JSONEncoder):
    """Helper class to convert Decimal to int/float for JSON serialization"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return int(obj) if obj % 1 == 0 else float(obj)
        return super(DecimalEncoder, self).default(obj)


def get_dynamodb_client():
    """
    Get DynamoDB client
    Configure AWS credentials via environment variables or AWS config
    """
    if not DYNAMODB_AVAILABLE:
        raise ImportError("boto3 is required for DynamoDB operations. Install with: pip install boto3")
    
    region = os.environ.get('AWS_REGION', 'us-east-1')
    
    # Create DynamoDB resource
    dynamodb = boto3.resource(
        'dynamodb',
        region_name=region,
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
    )
    
    return dynamodb


def query_user_images(
    customer_id: str,
    limit: int = 5,
    last_evaluated_key: Optional[Dict[str, Any]] = None,
    table_name: str = None
) -> Dict[str, Any]:
    """
    Query images for a specific customer from DynamoDB
    
    Args:
        customer_id: Customer/User ID (partition key)
        limit: Number of items to return per page (default: 5)
        last_evaluated_key: Pagination token from previous query
        table_name: DynamoDB table name (default from env or 'MaskTerial-Images')
    
    Returns:
        Dictionary containing:
        - items: List of image records
        - last_evaluated_key: Token for next page (None if no more pages)
        - count: Number of items returned
        - has_more: Boolean indicating if more pages exist
    """
    if not DYNAMODB_AVAILABLE:
        return {
            "error": "DynamoDB not available",
            "items": [],
            "last_evaluated_key": None,
            "count": 0,
            "has_more": False
        }
    
    try:
        dynamodb = get_dynamodb_client()
        
        # Get table name from environment or use default
        if table_name is None:
            table_name = os.environ.get('DYNAMODB_TABLE_NAME', 'CustomerImages')
        
        table = dynamodb.Table(table_name)
        
        # Build query parameters
        query_params = {
            'KeyConditionExpression': Key('customerID').eq(customer_id),
            'FilterExpression': Attr('status').eq('active'),  # Only return active images
            'ScanIndexForward': False,  # Sort by CreatedAt DESC
            'Limit': limit
        }
        
        # Add pagination token if provided
        if last_evaluated_key:
            query_params['ExclusiveStartKey'] = last_evaluated_key
        
        # Execute query
        response = table.query(**query_params)
        
        # Extract items and convert Decimal to native types
        items = []
        for item in response.get('Items', []):
            # Skip deleted items (double check)
            if item.get('status') == 'deleted':
                continue
                
            # Convert DynamoDB item to dict with native types
            converted_item = json.loads(json.dumps(item, cls=DecimalEncoder))
            
            # Normalize field names to match frontend expectations
            normalized_item = {
                'imageID': converted_item.get('imageID'),
                'customerID': converted_item.get('customerID'),
                'image_name': converted_item.get('imageName', converted_item.get('image_name', 'Unknown')),
                'image_url': converted_item.get('imageURL', converted_item.get('image_url', '')),
                'download_url': converted_item.get('downloadURL', ''),  # Presigned download URL
                'download_url_expires_at': converted_item.get('downloadURLExpiresAt'),  # Expiration timestamp
                'CreatedAt': converted_item.get('createdAt', converted_item.get('CreatedAt', '')),
                'metadata': converted_item.get('metadata', {}),
                's3Key': converted_item.get('s3Key'),
                'status': converted_item.get('status', 'active'),
                'type': converted_item.get('type', 'UPLOADED')
            }
            
            # Convert Unix timestamp to ISO format if needed
            if isinstance(normalized_item['CreatedAt'], (int, float)):
                from datetime import datetime
                timestamp = normalized_item['CreatedAt']
                
                # 检测时间戳格式 (秒 vs 毫秒)
                # 如果时间戳 > 10000000000 (约2286年)，说明是毫秒级
                if timestamp > 10000000000:
                    timestamp = timestamp / 1000  # 转换为秒
                
                try:
                    normalized_item['CreatedAt'] = datetime.fromtimestamp(timestamp).isoformat()
                except (ValueError, OSError) as e:
                    # 如果转换失败，保留原始值
                    print(f"Warning: Invalid timestamp {timestamp}: {e}")
                    normalized_item['CreatedAt'] = str(timestamp)
            
            items.append(normalized_item)
        
        # Get pagination info
        last_key = response.get('LastEvaluatedKey')
        
        return {
            "items": items,
            "last_evaluated_key": last_key,
            "count": len(items),
            "has_more": last_key is not None,
            "customer_id": customer_id
        }
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        print(f"DynamoDB ClientError: {error_code} - {error_message}")
        
        return {
            "error": f"DynamoDB error: {error_code}",
            "error_message": error_message,
            "items": [],
            "last_evaluated_key": None,
            "count": 0,
            "has_more": False
        }
    
    except Exception as e:
        print(f"Error querying DynamoDB: {str(e)}")
        return {
            "error": "Query failed",
            "error_message": str(e),
            "items": [],
            "last_evaluated_key": None,
            "count": 0,
            "has_more": False
        }


def save_image_metadata(
    customer_id: str,
    image_id: str,
    image_data: Dict[str, Any],
    table_name: str = None
) -> bool:
    """
    Save image metadata to DynamoDB
    
    Args:
        customer_id: Customer/User ID (partition key)
        image_id: Unique image identifier (sort key)
        image_data: Dictionary containing image metadata
        table_name: DynamoDB table name
    
    Returns:
        Boolean indicating success
    """
    if not DYNAMODB_AVAILABLE:
        return False
    
    try:
        dynamodb = get_dynamodb_client()
        
        if table_name is None:
            table_name = os.environ.get('DYNAMODB_TABLE_NAME', 'MaskTerial-Images')
        
        table = dynamodb.Table(table_name)
        
        # Prepare item
        item = {
            'customerID': customer_id,
            'imageID': image_id,
            'CreatedAt': image_data.get('created_at', datetime.now().isoformat()),
            **image_data
        }
        
        # Put item
        table.put_item(Item=item)
        
        return True
        
    except Exception as e:
        print(f"Error saving to DynamoDB: {str(e)}")
        return False


def get_image_by_id(
    customer_id: str,
    image_id: str,
    table_name: str = None
) -> Optional[Dict[str, Any]]:
    """
    Get a specific image by ID
    
    Args:
        customer_id: Customer/User ID
        image_id: Image identifier
        table_name: DynamoDB table name
    
    Returns:
        Image data dictionary or None if not found
    """
    if not DYNAMODB_AVAILABLE:
        return None
    
    try:
        dynamodb = get_dynamodb_client()
        
        if table_name is None:
            table_name = os.environ.get('DYNAMODB_TABLE_NAME', 'MaskTerial-Images')
        
        table = dynamodb.Table(table_name)
        
        response = table.get_item(
            Key={
                'customerID': customer_id,
                'imageID': image_id
            }
        )
        
        item = response.get('Item')
        if item:
            return json.loads(json.dumps(item, cls=DecimalEncoder))
        
        return None
        
    except Exception as e:
        print(f"Error getting image from DynamoDB: {str(e)}")
        return None


def convert_floats_to_decimal(obj):
    """
    Recursively convert all float values to Decimal for DynamoDB compatibility
    
    Args:
        obj: Object to convert (dict, list, or primitive)
    
    Returns:
        Converted object with Decimal instead of float
    """
    if isinstance(obj, list):
        return [convert_floats_to_decimal(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_floats_to_decimal(value) for key, value in obj.items()}
    elif isinstance(obj, float):
        return Decimal(str(obj))
    else:
        return obj


def update_image_metadata(
    customer_id: str,
    image_id: str,
    update_data: Dict[str, Any],
    table_name: str = None
) -> bool:
    """
    Update image metadata in DynamoDB
    
    Args:
        customer_id: Customer/User ID
        image_id: Image identifier
        update_data: Dictionary of fields to update
        table_name: DynamoDB table name
    
    Returns:
        Boolean indicating success
    """
    if not DYNAMODB_AVAILABLE:
        return False
    
    if not update_data:
        return True  # Nothing to update
    
    try:
        dynamodb = get_dynamodb_client()
        
        if table_name is None:
            table_name = os.environ.get('DYNAMODB_TABLE_NAME', 'CustomerImages')
        
        table = dynamodb.Table(table_name)
        
        # Convert floats to Decimal for DynamoDB
        update_data = convert_floats_to_decimal(update_data)
        
        # Build update expression
        update_expression_parts = []
        expression_attribute_values = {}
        expression_attribute_names = {}
        
        for key, value in update_data.items():
            # Skip primary keys and None values
            if key in ['customerID', 'imageID'] or value is None:
                continue
            
            # Handle reserved keywords by using attribute names
            attr_name = f"#{key}"
            attr_value = f":{key}"
            
            update_expression_parts.append(f"{attr_name} = {attr_value}")
            expression_attribute_values[attr_value] = value
            expression_attribute_names[attr_name] = key
        
        if not update_expression_parts:
            return True  # Nothing to update
        
        update_expression = "SET " + ", ".join(update_expression_parts)
        
        # Update item
        table.update_item(
            Key={
                'customerID': customer_id,
                'imageID': image_id
            },
            UpdateExpression=update_expression,
            ExpressionAttributeValues=expression_attribute_values,
            ExpressionAttributeNames=expression_attribute_names
        )
        
        return True
        
    except Exception as e:
        print(f"Error updating DynamoDB: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def delete_image(
    customer_id: str,
    image_id: str,
    table_name: str = None
) -> bool:
    """
    Delete an image record from DynamoDB
    
    Args:
        customer_id: Customer/User ID
        image_id: Image identifier
        table_name: DynamoDB table name
    
    Returns:
        Boolean indicating success
    """
    if not DYNAMODB_AVAILABLE:
        return False
    
    try:
        dynamodb = get_dynamodb_client()
        
        if table_name is None:
            table_name = os.environ.get('DYNAMODB_TABLE_NAME', 'MaskTerial-Images')
        
        table = dynamodb.Table(table_name)
        
        table.delete_item(
            Key={
                'customerID': customer_id,
                'imageID': image_id
            }
        )
        
        return True
        
    except Exception as e:
        print(f"Error deleting from DynamoDB: {str(e)}")
        return False

