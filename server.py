import io
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
import uuid
from typing import Literal, Optional
from datetime import datetime, timedelta

from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from PIL import Image
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from maskterial.modeling.classification_models import AMM_head, GMM_head
from maskterial.maskterial import MaskTerial
from maskterial.modeling.segmentation_models import M2F_model
from maskterial.utils.inference_server_utils import (
    ServerState,
    check_available_models,
    convert_coco_polygon_to_rle,
    read_image,
    update_server_state_and_predictor,
)
from maskterial.utils.user_utils import sanitize_user_id, log_user_action
from maskterial.utils.dynamodb_utils import (
    query_user_images,
    save_image_metadata,
    get_image_by_id,
    update_image_metadata,
    delete_image
)
from maskterial.utils.s3_utils import (
    delete_s3_object, 
    get_s3_bucket_from_url,
    generate_presigned_download_url,
    generate_public_url
)

# Working with lower precision improves performance marginally
torch.set_float32_matmul_precision("medium")


app = FastAPI(
    title="MaskTerial API",
    description="A Foundation Model for Automated 2D Material Flake Detection",
    version="1.0.0",
    openapi_version="3.0.2"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

server_state: ServerState | None = None
predictor: MaskTerial | None = None
currently_training = False
file_dir = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cls_model_dir = os.path.join(file_dir, "data", "models", "classification_models")
seg_model_dir = os.path.join(file_dir, "data", "models", "segmentation_models")
pp_model_dir = os.path.join(file_dir, "data", "models", "postprocessing_models")

pretrained_m2f_path = os.path.join(
    file_dir,
    "data",
    "models",
    "segmentation_models",
    "M2F",
    "Synthetic_Data",
    "model_final.pth",
)

available_cls_models = check_available_models(cls_model_dir)
available_seg_models = check_available_models(seg_model_dir)
available_pp_models = check_available_models(pp_model_dir)

frontend_dist_dir = Path(file_dir) / "maskterial-train-frontend" / "dist"
frontend_index_path = frontend_dist_dir / "index.html"

# Get AWS configuration from environment variables
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME', 'matsight-customer-images')
S3_REGION = os.environ.get('AWS_REGION', 'us-east-1')
DYNAMODB_TABLE_NAME = os.environ.get('DYNAMODB_TABLE_NAME', 'CustomerImages')

# Initialize AWS clients
s3_client = boto3.client('s3', region_name=S3_REGION)
dynamodb = boto3.resource('dynamodb', region_name=S3_REGION)
customer_images_table = dynamodb.Table(DYNAMODB_TABLE_NAME)


def get_user_id_from_request(
    user_id_form: Optional[str] = None,
    user_id_header: Optional[str] = None
) -> str:
    """
    Extract and sanitize user_id from request
    Priority: Form data > Header > Default
    """
    user_id = user_id_form or user_id_header or "test_user"
    return sanitize_user_id(user_id)


def _server_state_payload() -> dict[str, object]:
    if server_state is not None:
        return server_state.to_dict()
    return {"message": "No Models are loaded, try to run inference on the /predict endpoint"}


def _status_message() -> str:
    if currently_training:
        return "Currently training, try again later"
    return "Ready for inference"


if frontend_index_path.exists():
    assets_path = frontend_dist_dir / "assets"
    if assets_path.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_path), name="frontend-assets")

    @app.get("/", response_class=HTMLResponse)
    async def serve_frontend_root() -> FileResponse:
        return FileResponse(frontend_index_path)

    @app.get("/{full_path:path}", response_class=HTMLResponse)
    async def serve_frontend_routes(full_path: str) -> FileResponse:
        if full_path.startswith("api"):
            raise HTTPException(status_code=404, detail="Not Found")
        candidate = frontend_dist_dir / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(frontend_index_path)
else:

    @app.get("/")
    async def serve_frontend_root() -> dict[str, object]:
        return _server_state_payload()


@app.get("/api/server-state")
async def api_server_state() -> dict[str, object]:
    return _server_state_payload()


@app.get("/status")
async def check_status() -> str:
    return _status_message()


@app.get("/api/status")
async def check_status_api() -> str:
    return _status_message()


@app.get("/available_models")
async def get_models():
    global available_cls_models, available_seg_models, available_pp_models
    available_cls_models = check_available_models(cls_model_dir)
    available_seg_models = check_available_models(seg_model_dir)
    available_pp_models = check_available_models(pp_model_dir)

    return {
        "available_models": {
            "classification_models": available_cls_models,
            "segmentation_models": available_seg_models,
            "postprocessing_models": available_pp_models,
        },
    }


@app.get("/api/available_models")
async def get_models_api():
    return await get_models()


@app.post("/upload_image")
async def upload_image(
    image_file: UploadFile = File(...),
    image_name: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
):
    """
    Upload an image to S3 and store metadata in DynamoDB
    
    Args:
        image_file: The image file to upload
        image_name: Custom name for the image (optional, defaults to original filename)
        user_id: User ID (from form data or header)
    
    Returns:
        Dictionary containing imageID and upload details
    """
    # Get and sanitize user_id
    customer_id = get_user_id_from_request(user_id, x_user_id)
    
    # Generate unique image ID
    image_id = str(uuid.uuid4())
    
    # Get image name (use provided name or original filename)
    if image_name is None or image_name.strip() == "":
        image_name = image_file.filename
    
    # Read and validate image file
    try:
        image_data = await image_file.read()
        
        # Validate it's a valid image
        image = Image.open(io.BytesIO(image_data))
        image_format = image.format.lower() if image.format else 'jpg'
        
        # Convert to JPEG if needed
        if image_format not in ['jpg', 'jpeg']:
            # Convert to RGB if needed (handles PNG with transparency, etc.)
            if image.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                background.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Save as JPEG
            output_buffer = io.BytesIO()
            image.save(output_buffer, format='JPEG', quality=95)
            image_data = output_buffer.getvalue()
            image_format = 'jpg'
    
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image file: {str(e)}"
        )
    
    # Construct S3 key
    s3_key = f"{customer_id}/uploaded/{image_id}_original.jpg"
    
    # Upload to S3
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            Body=image_data,
            ContentType='image/jpeg',
            Metadata={
                'customer-id': customer_id,
                'image-id': image_id,
                'original-name': image_name
            }
        )
    except ClientError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload image to S3: {str(e)}"
        )
    
    # Generate presigned URL for viewing (7天有效，足够长)
    # AWS presigned URL最长7天，我们使用最大值
    presigned_view_url = generate_presigned_download_url(
        bucket_name=S3_BUCKET_NAME,
        s3_key=s3_key,
        expiration=7 * 24 * 60 * 60,  # 7 days (AWS maximum for presigned URLs)
        custom_filename=None  # Don't force download, allow viewing in browser
    )
    
    # Generate presigned URL for downloading (7天有效)
    presigned_download_url = generate_presigned_download_url(
        bucket_name=S3_BUCKET_NAME,
        s3_key=s3_key,
        expiration=7 * 24 * 60 * 60,  # 7 days
        custom_filename=image_name  # Force download with original filename
    )
    
    # Use presigned URL as primary URL (private bucket compatible)
    download_url = presigned_view_url or presigned_download_url
    
    # Calculate URL expiration timestamp (7天后)
    url_expires_at = int((datetime.now() + timedelta(days=7)).timestamp())
    
    # Prepare DynamoDB record
    current_timestamp = int(datetime.now().timestamp())
    # Set record expiration to 90 days from now (configurable)
    expires_at = int((datetime.now() + timedelta(days=90)).timestamp())
    
    dynamodb_item = {
        'customerID': customer_id,
        'imageID': image_id,
        'imageName': image_name,
        'createdAt': current_timestamp,
        'type': 'UPLOADED',
        's3Key': s3_key,
        'imageURL': presigned_view_url,  # Presigned URL for viewing
        'downloadURL': presigned_download_url,  # Presigned URL for downloading (7天有效)
        'downloadURLExpiresAt': url_expires_at,  # URL expiration timestamp
        'status': 'active',
        'metadata': {},  # Empty for now, will be filled after inference
        'expiresAt': expires_at
    }
    
    # Store in DynamoDB
    try:
        customer_images_table.put_item(Item=dynamodb_item)
    except ClientError as e:
        # If DynamoDB fails, try to delete the S3 object
        try:
            s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        except:
            pass
        raise HTTPException(
            status_code=500,
            detail=f"Failed to store image metadata in DynamoDB: {str(e)}"
        )
    
    # Log user action
    log_user_action(
        file_dir, 
        customer_id, 
        "upload_image", 
        {
            "image_id": image_id,
            "image_name": image_name,
            "s3_key": s3_key,
            "download_url_generated": presigned_download_url is not None
        }
    )
    
    return {
        "success": True,
        "imageID": image_id,
        "customerID": customer_id,
        "imageName": image_name,
        "s3Key": s3_key,
        "imageURL": presigned_view_url,  # Presigned URL for viewing (7天有效)
        "downloadURL": presigned_download_url,  # Presigned URL for downloading (7天有效)
        "downloadURLExpiresAt": url_expires_at,  # URL过期时间
        "createdAt": current_timestamp,
        "bucket": S3_BUCKET_NAME,
        "key": s3_key,
        "region": S3_REGION,
        "message": "Image uploaded successfully"
    }


@app.get("/get_user_images")
async def get_user_images(
    user_id: Optional[str] = None,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    status: str = "active",
    limit: int = 100
):
    """
    Get list of images for a specific user from DynamoDB
    
    Args:
        user_id: User ID (from query param or header)
        status: Filter by status (default: "active")
        limit: Maximum number of images to return (default: 100)
    
    Returns:
        List of image metadata
    """
    # Get and sanitize user_id
    customer_id = get_user_id_from_request(user_id, x_user_id)
    
    try:
        # Query DynamoDB for user's images
        response = customer_images_table.query(
            KeyConditionExpression='customerID = :customer_id',
            ExpressionAttributeValues={
                ':customer_id': customer_id
            },
            ScanIndexForward=False,  # Sort by sort key (imageID) in descending order
            Limit=limit
        )
        
        images = response.get('Items', [])
        
        # Filter by status if specified
        if status:
            images = [img for img in images if img.get('status') == status]
        
        # Generate presigned URLs for each image
        image_list = []
        for img in images:
            # Generate presigned URL (valid for 1 hour)
            try:
                presigned_url = s3_client.generate_presigned_url(
                    'get_object',
                    Params={
                        'Bucket': S3_BUCKET_NAME,
                        'Key': img['s3Key']
                    },
                    ExpiresIn=3600  # URL valid for 1 hour
                )
            except ClientError:
                presigned_url = None
            
            image_list.append({
                'imageID': img['imageID'],
                'imageName': img.get('imageName', 'Untitled'),
                'createdAt': img['createdAt'],
                'type': img.get('type', 'UPLOADED'),
                'status': img.get('status', 'active'),
                'url': presigned_url,
                's3Key': img['s3Key'],
                'metadata': img.get('metadata', {})
            })
        
        return {
            'success': True,
            'customerID': customer_id,
            'count': len(image_list),
            'images': image_list
        }
        
    except ClientError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve images from DynamoDB: {str(e)}"
        )


@app.post("/upload/amm")
async def upload_amm(
    model_name: str = Form(...),
    metadata_file: UploadFile = File(...),
    loc_file: UploadFile = File(...),
    cov_file: UploadFile = File(...),
    weights_file: UploadFile = File(...),
):

    available_models = check_available_models(cls_model_dir)
    if model_name in available_models["AMM"]:
        raise HTTPException(
            status_code=400,
            detail=f"Model {model_name} already exists, please choose a different name",
        )

    new_model_path = os.path.join(cls_model_dir, "AMM", model_name)

    try:
        os.makedirs(new_model_path)

        loc_path = os.path.join(new_model_path, "loc.npy")
        cov_path = os.path.join(new_model_path, "cov.npy")
        model_path = os.path.join(new_model_path, "model.pth")
        meta_path = os.path.join(new_model_path, "meta_data.json")

        with open(meta_path, "wb") as f:
            f.write(metadata_file.file.read())

        with open(loc_path, "wb") as f:
            f.write(loc_file.file.read())

        with open(cov_path, "wb") as f:
            f.write(cov_file.file.read())

        with open(model_path, "wb") as f:
            f.write(weights_file.file.read())

        # Create a new model instance to check if the model is valid
        AMM_head.from_pretrained(new_model_path)

    except Exception as e:
        shutil.rmtree(new_model_path)
        raise HTTPException(status_code=500, detail=f"Failed to create model: {str(e)}")

    return f"Model {model_name} uploaded successfully"


@app.post("/upload/gmm")
async def upload_gmm(
    model_name: str = Form(...),
    contrast_file: UploadFile = File(...),
):

    available_models = check_available_models(cls_model_dir)
    if model_name in available_models["GMM"]:
        raise HTTPException(
            status_code=400,
            detail=f"Model {model_name} already exists, please choose a different name",
        )

    new_model_path = os.path.join(cls_model_dir, "GMM", model_name)

    try:
        os.makedirs(new_model_path)
        contrast_dict_path = os.path.join(new_model_path, "contrast_dict.json")
        with open(contrast_dict_path, "wb") as f:
            f.write(contrast_file.file.read())

        GMM_head.from_pretrained(new_model_path)

    except Exception as e:
        shutil.rmtree(new_model_path)
        raise HTTPException(status_code=500, detail=f"Failed to create model: {str(e)}")

    return f"Model {model_name} uploaded successfully"


@app.post("/upload/m2f")
async def upload_M2F(
    model_name: str = Form(...),
    model_file: UploadFile = File(...),
    config_file: UploadFile = File(None),
):

    available_models = check_available_models(seg_model_dir)
    if model_name in available_models["M2F"]:
        raise HTTPException(
            status_code=400,
            detail=f"Model {model_name} already exists, please choose a different name",
        )
    new_model_path = os.path.join(seg_model_dir, "M2F", model_name)

    try:
        os.makedirs(new_model_path)

        config_path = os.path.join(new_model_path, "config.yaml")
        with open(config_path, "wb") as f:
            f.write(config_file.file.read())

        model_path = os.path.join(new_model_path, "model.pth")
        with open(model_path, "wb") as f:
            f.write(model_file.file.read())

        M2F_model.from_pretrained(new_model_path)

    except Exception as e:
        shutil.rmtree(new_model_path)
        raise HTTPException(status_code=500, detail=f"Failed to create model: {str(e)}")

    return f"Model {model_name} uploaded successfully"


@app.post("/delete_model")
async def delete_model(
    model_class: Literal["segmentation", "classification"] = Form(...),
    model_type: str = Form(...),
    model_name: str = Form(...),
):
    model_dir = None
    if model_class == "segmentation":
        model_dir = seg_model_dir
    elif model_class == "classification":
        model_dir = cls_model_dir

    available_models = check_available_models(model_dir)

    # check if the model exists in the available models [model_type][model_name]
    if model_type not in available_models:
        raise HTTPException(
            status_code=404,
            detail=f"Model type {model_type} not found for {model_class}",
        )
    if model_name not in available_models[model_type]:
        raise HTTPException(
            status_code=404,
            detail=f"Model name {model_name} not found in {model_type} for {model_class}",
        )

    model_path = os.path.join(model_dir, model_type, model_name)

    try:
        shutil.rmtree(model_path)
        return f"Model {model_name} deleted successfully"

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")


@app.get("/download_model")
async def download_model(
    model_class: Literal["segmentation", "classification", "postprocessing"],
    model_type: str,
    model_name: str,
):
    model_dir = None
    if model_class == "segmentation":
        model_dir = seg_model_dir
    elif model_class == "classification":
        model_dir = cls_model_dir
    elif model_class == "postprocessing":
        model_dir = pp_model_dir

    available_models = check_available_models(model_dir)

    # check if the model exists in the available models [model_type][model_name]
    if model_type not in available_models:
        raise HTTPException(
            status_code=404,
            detail=f"Model type {model_type} not found for {model_class}",
        )
    if model_name not in available_models[model_type]:
        raise HTTPException(
            status_code=404,
            detail=f"Model name {model_name} not found in {model_type} for {model_class}",
        )

    model_path = os.path.join(model_dir, model_type, model_name)

    # Create an in-memory ZIP file
    zip_buffer = io.BytesIO()
    try:
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for root, _, files in os.walk(model_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, model_path)
                    zip_file.write(file_path, arcname)

        # Ensure buffer's pointer is at the beginning
        zip_buffer.seek(0)

        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename={model_class}-{model_type}-{model_name}.zip"
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to create zip file: {str(e)}"
        )


@app.post("/predict")
async def predict(
    files: list[UploadFile] = File(...),
    segmentation_model: str | None = Form(None),
    classification_model: str | None = Form(None),
    postprocessing_model: str | None = Form(None),
    score_threshold: float = Form(0.0),
    min_class_occupancy: float = Form(0.0),
    size_threshold: int = Form(300),
    return_bbox: bool = Form(False),
    user_id: Optional[str] = Form(None),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
):
    global server_state, predictor

    # Get and sanitize user_id
    current_user_id = get_user_id_from_request(user_id, x_user_id)

    if currently_training:
        return "Currently training, try again later"

    new_server_state = ServerState(
        seg_model_name=segmentation_model,
        cls_model_name=classification_model,
        pp_model_name=postprocessing_model,
        score_threshold=score_threshold,
        min_class_occupancy=min_class_occupancy,
        size_threshold=size_threshold,
        cls_model_dir=cls_model_dir,
        seg_model_dir=seg_model_dir,
        pp_model_dir=pp_model_dir,
        device=device,
    )

    if server_state != new_server_state:
        server_state, predictor = update_server_state_and_predictor(new_server_state)

    start = time.time()
    img = read_image(files[0])
    print(f"[User: {current_user_id}] Read Time:", round(time.time() - start, 3), "s")

    start = time.time()
    result = predictor.predict(img)
    print(f"[User: {current_user_id}] Pred Time:", round(time.time() - start, 3), "s")

    # Log user action
    log_user_action(
        file_dir, 
        current_user_id, 
        "inference", 
        {
            "segmentation_model": segmentation_model,
            "classification_model": classification_model,
            "num_results": len(result)
        }
    )

    result_dict = [r.to_dict(return_bbox=return_bbox) for r in result]
    return result_dict


@app.post("/train/m2f")
async def train(
    model_name: str = Form(...),
    dataset_file: UploadFile = File(...),
    config_file: UploadFile = File(None),
):
    global currently_training, server_state

    available_models = check_available_models(seg_model_dir)
    if model_name in available_models["M2F"]:
        raise HTTPException(
            status_code=400,
            detail=f"Model {model_name} already exists, please choose a different name",
        )
    new_model_path = os.path.join(seg_model_dir, "M2F", model_name)
    data_dir = os.path.join(new_model_path, "data")
    image_dir = os.path.join(data_dir, "images")
    ann_path = os.path.join(data_dir, "result.json")
    RLE_ann_path = os.path.join(data_dir, "result_RLE.json")

    if config_file is None:
        config_path = os.path.join("configs", "M2F", "base_config.yaml")
    else:
        config_path = os.path.join(new_model_path, "train_config.yaml")

    try:
        currently_training = True
        os.makedirs(new_model_path)
        os.makedirs(data_dir)

        with zipfile.ZipFile(dataset_file.file, "r") as zip_ref:
            zip_ref.extractall(data_dir)

        # we now need to convert the polygon annotations to RLE format
        with open(ann_path, "r") as f:
            coco_ann = convert_coco_polygon_to_rle(json.load(f))
            # we also need to strip the first images\/ from the image path
            for ann in coco_ann["images"]:
                ann["file_name"] = ann["file_name"][7:]

        with open(RLE_ann_path, "w") as f:
            json.dump(coco_ann, f)

        if config_file is not None:
            with open(config_path, "wb") as f:
                f.write(config_file.file.read())

        os.environ["WANDB_ACTIVE"] = "0"

        torch.cuda.empty_cache()
        server_state = None

        cmd = [
            sys.executable,
            "-u",
            "train_segmentation_model.py",
            "--config-file",
            config_path,
            "--train-image-root",
            image_dir,
            "--train-annotation-path",
            RLE_ann_path,
            "--dist-url",
            "auto",
            "OUTPUT_DIR",
            new_model_path,
            "MODEL.WEIGHTS",
            pretrained_m2f_path,
        ]

        def stream_process():
            global currently_training
            yield "Training started...\r"
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,  # Merge stdout and stderr
                    bufsize=1,  # Line-buffered
                    text=True,
                )
                for line in iter(process.stdout.readline, ""):
                    print(line, end="")
                    yield line
            except Exception as e:
                raise Exception(str(e))
            finally:
                process.stdout.close()
                process.wait()
                currently_training = False
                torch.cuda.empty_cache()

        return StreamingResponse(stream_process(), media_type="text/event-stream")

    except Exception as e:
        shutil.rmtree(new_model_path)
        currently_training = False
        raise HTTPException(status_code=500, detail=f"Failed to train model: {str(e)}")


# ============================================
# DynamoDB Image Gallery API Endpoints
# ============================================

@app.get("/images")
@app.get("/api/images")
async def get_user_images(
    limit: int = 5,
    last_key: Optional[str] = None,
    user_id: Optional[str] = None,
    status: Optional[str] = None,  # For backward compatibility
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
):
    """
    Get images for a specific user from DynamoDB with pagination
    Auto-refreshes expired presigned URLs
    
    Query Parameters:
    - limit: Number of items per page (default: 5)
    - last_key: Base64-encoded pagination token from previous request
    - user_id: User ID (can also be provided via X-User-ID header)
    
    Returns:
    - items: List of image records with refreshed URLs if needed
    - last_evaluated_key: Pagination token for next page (base64-encoded)
    - count: Number of items in current page
    - has_more: Boolean indicating if more pages exist
    """
    # Get and sanitize user_id
    current_user_id = get_user_id_from_request(user_id, x_user_id)
    
    # Decode pagination token if provided
    last_evaluated_key = None
    if last_key:
        try:
            import base64
            decoded_key = base64.b64decode(last_key).decode('utf-8')
            last_evaluated_key = json.loads(decoded_key)
        except Exception as e:
            print(f"Error decoding pagination token: {str(e)}")
            # Continue with None - will start from beginning
    
    # Query DynamoDB
    result = query_user_images(
        customer_id=current_user_id,
        limit=limit,
        last_evaluated_key=last_evaluated_key
    )
    
    # Check for errors
    if "error" in result:
        raise HTTPException(
            status_code=500,
            detail=result.get("error_message", "Failed to query images")
        )
    
    # Auto-refresh expired URLs
    current_time = int(datetime.now().timestamp())
    refreshed_items = []
    
    for item in result["items"]:
        # Check if URL is expired or expiring soon (within 1 day)
        url_expires_at = item.get('download_url_expires_at')
        
        # If no expiration time or URL is missing, needs refresh
        needs_refresh = (
            not item.get('image_url') or 
            not item.get('download_url') or
            url_expires_at is None or
            url_expires_at < (current_time + 86400)  # Expiring within 24 hours
        )
        
        if needs_refresh and item.get('s3Key'):
            print(f"Auto-refreshing URL for image {item['imageID']} (expires at {url_expires_at})")
            
            # Generate new presigned URLs
            new_view_url = generate_presigned_download_url(
                bucket_name=S3_BUCKET_NAME,
                s3_key=item['s3Key'],
                expiration=7 * 24 * 60 * 60,
                custom_filename=None
            )
            
            new_download_url = generate_presigned_download_url(
                bucket_name=S3_BUCKET_NAME,
                s3_key=item['s3Key'],
                expiration=7 * 24 * 60 * 60,
                custom_filename=item.get('image_name', 'download.jpg')
            )
            
            new_expires_at = int((datetime.now() + timedelta(days=7)).timestamp())
            
            # Update item with new URLs
            if new_view_url:
                item['image_url'] = new_view_url
                item['download_url'] = new_download_url or new_view_url
                item['download_url_expires_at'] = new_expires_at
                
                # Update DynamoDB
                try:
                    customer_images_table.update_item(
                        Key={
                            'customerID': item['customerID'],
                            'imageID': item['imageID']
                        },
                        UpdateExpression='SET imageURL = :view_url, downloadURL = :dl_url, downloadURLExpiresAt = :exp',
                        ExpressionAttributeValues={
                            ':view_url': new_view_url,
                            ':dl_url': new_download_url or new_view_url,
                            ':exp': new_expires_at
                        }
                    )
                    print(f"✓ URL refreshed for {item['imageID']}")
                except Exception as e:
                    print(f"✗ Failed to update DynamoDB: {str(e)}")
        
        refreshed_items.append(item)
    
    # Encode pagination token for response
    encoded_last_key = None
    if result.get("last_evaluated_key"):
        import base64
        json_key = json.dumps(result["last_evaluated_key"])
        encoded_last_key = base64.b64encode(json_key.encode('utf-8')).decode('utf-8')
    
    return {
        "items": refreshed_items,
        "last_evaluated_key": encoded_last_key,
        "count": len(refreshed_items),
        "has_more": result["has_more"],
        "customer_id": current_user_id,
        "page_size": limit
    }


@app.get("/images/{image_id}/download")
@app.get("/api/images/{image_id}/download")
async def download_image_file(
    image_id: str,
    user_id: Optional[str] = None,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
):
    """
    Download image file from S3 (proxy endpoint to avoid CORS)
    
    Path Parameters:
    - image_id: Unique image identifier
    
    Returns:
    - Image file as streaming response
    """
    # Get and sanitize user_id
    current_user_id = get_user_id_from_request(user_id, x_user_id)
    
    # Get image from DynamoDB
    image_data = get_image_by_id(
        customer_id=current_user_id,
        image_id=image_id
    )
    
    if not image_data:
        raise HTTPException(
            status_code=404,
            detail=f"Image {image_id} not found for user {current_user_id}"
        )
    
    # Get S3 key
    s3_key = image_data.get('s3Key')
    if not s3_key:
        raise HTTPException(
            status_code=404,
            detail="Image S3 key not found"
        )
    
    # Download from S3
    try:
        response = s3_client.get_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key
        )
        
        # Get image data
        image_bytes = response['Body'].read()
        
        # Return as streaming response
        return StreamingResponse(
            io.BytesIO(image_bytes),
            media_type=response.get('ContentType', 'image/jpeg'),
            headers={
                'Content-Disposition': f'inline; filename="{image_data.get("image_name", "image.jpg")}"',
                'Access-Control-Allow-Origin': '*',
                'Cache-Control': 'public, max-age=31536000'
            }
        )
        
    except ClientError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download image from S3: {str(e)}"
        )


@app.get("/images/{image_id}")
@app.get("/api/images/{image_id}")
async def get_image_details(
    image_id: str,
    user_id: Optional[str] = None,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
):
    """
    Get details of a specific image by ID
    
    Path Parameters:
    - image_id: Unique image identifier
    
    Returns:
    - Image metadata and details
    """
    # Get and sanitize user_id
    current_user_id = get_user_id_from_request(user_id, x_user_id)
    
    # Get image from DynamoDB
    image_data = get_image_by_id(
        customer_id=current_user_id,
        image_id=image_id
    )
    
    if not image_data:
        raise HTTPException(
            status_code=404,
            detail=f"Image {image_id} not found for user {current_user_id}"
        )
    
    return image_data


@app.post("/images")
@app.post("/api/images")
async def save_image(
    image_id: Optional[str] = Form(None),
    image_name: str = Form(...),
    image_url: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
):
    """
    Save image metadata to DynamoDB
    
    Form Parameters:
    - image_id: Optional image ID (will be generated if not provided)
    - image_name: Name of the image
    - image_url: Optional URL where image is stored
    - metadata: Optional JSON string with additional metadata
    - user_id: User ID
    
    Returns:
    - Success message with image_id
    """
    # Get and sanitize user_id
    current_user_id = get_user_id_from_request(user_id, x_user_id)
    
    # Generate image_id if not provided
    if not image_id:
        image_id = f"img_{uuid.uuid4().hex[:16]}"
    
    # Parse metadata if provided
    additional_metadata = {}
    if metadata:
        try:
            additional_metadata = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400,
                detail="Invalid metadata JSON format"
            )
    
    # Prepare image data
    image_data = {
        "created_at": datetime.now().isoformat(),
        "image_name": image_name,
        "image_url": image_url or "",
        **additional_metadata
    }
    
    # Save to DynamoDB
    success = save_image_metadata(
        customer_id=current_user_id,
        image_id=image_id,
        image_data=image_data
    )
    
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to save image metadata to DynamoDB"
        )
    
    # Log action
    log_user_action(
        file_dir,
        current_user_id,
        "image_saved",
        {
            "image_id": image_id,
            "image_name": image_name
        }
    )
    
    return {
        "success": True,
        "message": "Image metadata saved successfully",
        "image_id": image_id,
        "customer_id": current_user_id
    }


@app.put("/images/{image_id}")
@app.put("/api/images/{image_id}")
@app.patch("/images/{image_id}")
@app.patch("/api/images/{image_id}")
async def update_image(
    image_id: str,
    image_name: Optional[str] = Form(None),
    status: Optional[str] = Form(None),
    type: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
):
    """
    Update image metadata in DynamoDB
    
    Path Parameters:
    - image_id: Unique image identifier
    
    Form Parameters:
    - image_name: New name for the image
    - status: New status (active | deleted)
    - type: Image type (UPLOADED | PROCESSED)
    - metadata: JSON string with metadata updates
    - user_id: User ID
    
    Returns:
    - Success message with updated fields
    """
    # Get and sanitize user_id
    current_user_id = get_user_id_from_request(user_id, x_user_id)
    
    # Validate status
    if status is not None and status not in ['active', 'deleted']:
        raise HTTPException(
            status_code=400,
            detail="Status must be either 'active' or 'deleted'"
        )
    
    # Validate type
    if type is not None and type not in ['UPLOADED', 'PROCESSED']:
        raise HTTPException(
            status_code=400,
            detail="Type must be either 'UPLOADED' or 'PROCESSED'"
        )
    
    # Prepare update data
    update_data = {}
    
    if image_name is not None:
        update_data['imageName'] = image_name
    
    if status is not None:
        update_data['status'] = status
    
    if type is not None:
        update_data['type'] = type
    
    if metadata is not None:
        try:
            metadata_dict = json.loads(metadata)
            update_data['metadata'] = metadata_dict
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400,
                detail="Invalid metadata JSON format"
            )
    
    if not update_data:
        raise HTTPException(
            status_code=400,
            detail="No update data provided"
        )
    
    # Update in DynamoDB
    success = update_image_metadata(
        customer_id=current_user_id,
        image_id=image_id,
        update_data=update_data
    )
    
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to update image metadata in DynamoDB"
        )
    
    # Log action
    log_user_action(
        file_dir,
        current_user_id,
        "image_updated",
        {
            "image_id": image_id,
            "updated_fields": list(update_data.keys())
        }
    )
    
    return {
        "success": True,
        "message": "Image metadata updated successfully",
        "image_id": image_id,
        "updated_fields": list(update_data.keys())
    }


@app.post("/images/{image_id}/refresh-download-url")
@app.post("/api/images/{image_id}/refresh-download-url")
async def refresh_download_url(
    image_id: str,
    expiration: int = 7,  # Days
    user_id: Optional[str] = Form(None),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
):
    """
    Refresh the presigned download URL for an image
    
    Path Parameters:
    - image_id: Unique image identifier
    
    Form Parameters:
    - expiration: URL expiration in days (default: 7)
    
    Returns:
    - New presigned download URL
    """
    # Get and sanitize user_id
    current_user_id = get_user_id_from_request(user_id, x_user_id)
    
    # Get image details from DynamoDB
    image_data = get_image_by_id(
        customer_id=current_user_id,
        image_id=image_id
    )
    
    if not image_data:
        raise HTTPException(
            status_code=404,
            detail=f"Image {image_id} not found"
        )
    
    # Get S3 info
    s3_bucket = os.environ.get('S3_BUCKET_NAME')
    s3_key = image_data.get('s3Key')
    
    if not s3_bucket or not s3_key:
        raise HTTPException(
            status_code=400,
            detail="Image does not have S3 information"
        )
    
    # Generate new presigned URL
    expiration_seconds = expiration * 24 * 60 * 60
    presigned_url = generate_presigned_download_url(
        bucket_name=s3_bucket,
        s3_key=s3_key,
        expiration=expiration_seconds,
        custom_filename=image_data.get('image_name', 'download.jpg')
    )
    
    if not presigned_url:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate presigned URL"
        )
    
    # Calculate new expiration timestamp
    url_expires_at = int((datetime.now() + timedelta(days=expiration)).timestamp())
    
    # Update DynamoDB with new URL
    try:
        from boto3.dynamodb.conditions import Key
        dynamodb = boto3.resource('dynamodb', region_name=os.environ.get('AWS_REGION', 'us-east-1'))
        table_name = os.environ.get('DYNAMODB_TABLE_NAME', 'CustomerImages')
        table = dynamodb.Table(table_name)
        
        table.update_item(
            Key={
                'customerID': current_user_id,
                'imageID': image_id
            },
            UpdateExpression='SET downloadURL = :url, downloadURLExpiresAt = :exp',
            ExpressionAttributeValues={
                ':url': presigned_url,
                ':exp': url_expires_at
            }
        )
    except Exception as e:
        print(f"Warning: Failed to update DynamoDB with new URL: {str(e)}")
        # Continue anyway, URL is still valid
    
    return {
        "success": True,
        "imageID": image_id,
        "downloadURL": presigned_url,
        "downloadURLExpiresAt": url_expires_at,
        "expiresIn": f"{expiration} days",
        "message": "Download URL refreshed successfully"
    }


@app.delete("/images/{image_id}")
@app.delete("/api/images/{image_id}")
async def delete_image_endpoint(
    image_id: str,
    user_id: Optional[str] = None,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
):
    """
    Delete an image: removes from both S3 and DynamoDB
    
    Path Parameters:
    - image_id: Unique image identifier
    
    Returns:
    - Success message with details about what was deleted
    """
    # Get and sanitize user_id
    current_user_id = get_user_id_from_request(user_id, x_user_id)
    
    # First, get image details to find S3 location
    image_data = get_image_by_id(
        customer_id=current_user_id,
        image_id=image_id
    )
    
    deletion_results = {
        "s3_deleted": False,
        "dynamodb_deleted": False,
        "errors": []
    }
    
    # Delete from S3 if we have S3 information
    if image_data:
        s3_key = image_data.get('s3Key')
        
        print(f"Attempting to delete S3 object: bucket={S3_BUCKET_NAME}, key={s3_key}")
        
        # Delete from S3 if we have the key
        if s3_key:
            try:
                # Use global S3_BUCKET_NAME and s3_client
                s3_client.delete_object(
                    Bucket=S3_BUCKET_NAME,
                    Key=s3_key
                )
                deletion_results["s3_deleted"] = True
                print(f"✓ Successfully deleted S3 object: s3://{S3_BUCKET_NAME}/{s3_key}")
            except ClientError as e:
                error_msg = f"S3 ClientError: {e.response['Error']['Code']} - {e.response['Error']['Message']}"
                print(f"✗ {error_msg}")
                deletion_results["errors"].append(error_msg)
            except Exception as e:
                error_msg = f"S3 deletion error: {str(e)}"
                print(f"✗ {error_msg}")
                deletion_results["errors"].append(error_msg)
        else:
            print(f"✗ No s3Key found for image {image_id}, skipping S3 deletion")
            deletion_results["errors"].append("No S3 key found in image data")
    else:
        print(f"✗ No image data found for {image_id}")
        deletion_results["errors"].append("Image not found in DynamoDB")
    
    # Delete from DynamoDB
    try:
        dynamodb_success = delete_image(
            customer_id=current_user_id,
            image_id=image_id
        )
        deletion_results["dynamodb_deleted"] = dynamodb_success
        
        if not dynamodb_success:
            deletion_results["errors"].append("Failed to delete from DynamoDB")
    except Exception as e:
        print(f"Error deleting from DynamoDB: {str(e)}")
        deletion_results["errors"].append(f"DynamoDB deletion error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete image from DynamoDB: {str(e)}"
        )
    
    # Log action
    log_user_action(
        file_dir,
        current_user_id,
        "image_deleted",
        {
            "image_id": image_id,
            "s3_deleted": deletion_results["s3_deleted"],
            "dynamodb_deleted": deletion_results["dynamodb_deleted"]
        }
    )
    
    # Determine overall success
    overall_success = deletion_results["dynamodb_deleted"]
    
    return {
        "success": overall_success,
        "message": f"Image {image_id} deleted" + (" (DynamoDB only)" if not deletion_results["s3_deleted"] else " (S3 and DynamoDB)"),
        "image_id": image_id,
        "details": deletion_results
    }


