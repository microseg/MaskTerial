# MaskTerial Deployment Fixes Documentation

## Overview

This document outlines the fixes applied to the MaskTerial repository to resolve deployment issues and ensure proper functionality of the Docker-based inference server.

## Issues Identified

### 1. FastAPI Swagger UI Error
**Problem**: Accessing `http://localhost:8000/api/docs` resulted in the error:
```
Unable to render this definition
The provided definition does not specify a valid version field.
```

**Root Cause**: The FastAPI application was initialized without proper metadata configuration.

### 2. Nginx Routing Configuration Issue
**Problem**: Swagger UI could not access the OpenAPI JSON specification.

**Root Cause**: Missing nginx route configuration for `/openapi.json` endpoint.

## Fixes Applied

### 1. FastAPI Configuration Fix (`server.py`)

**Original Code**:
```python
app = FastAPI()
```

**Fixed Code**:
```python
app = FastAPI(
    title="MaskTerial API",
    description="A Foundation Model for Automated 2D Material Flake Detection",
    version="1.0.0",
    openapi_version="3.0.2"
)
```

**Purpose**: 
- Provides proper API metadata for Swagger UI
- Ensures OpenAPI specification includes required version field
- Improves API documentation quality

### 2. Nginx Configuration Fix (`etc/nginx.conf`)

**Original Configuration**:
```nginx
location /api/ {
    proxy_pass http://backend:8000/;
}

location / {
    root   /usr/share/nginx/html;
    try_files $uri $uri/ /index.html;
}
```

**Fixed Configuration**:
```nginx
location /api/ {
    proxy_pass http://backend:8000/;
}

# Serve OpenAPI JSON directly
location /openapi.json {
    proxy_pass http://backend:8000/openapi.json;
}

location / {
    root   /usr/share/nginx/html;
    try_files $uri $uri/ /index.html;
}
```

**Purpose**:
- Enables Swagger UI to access OpenAPI JSON specification
- Provides direct route for `/openapi.json` endpoint
- Maintains existing API and frontend routing

### 3. Data Directory Structure Creation

**Created Structure**:
```
data/
├── models/
│   ├── segmentation_models/M2F/
│   │   ├── Synthetic_Data/
│   │   └── GrapheneH/
│   ├── classification_models/
│   │   ├── AMM/GrapheneH/
│   │   └── GMM/
│   └── postprocessing_models/
```

**Purpose**:
- Provides storage location for pretrained models
- Organizes models by type and material
- Enables proper model management

## Model Downloads

### Downloaded Models

1. **Base Segmentation Model**:
   - **Model**: Mask2Former (Synthetic Data)
   - **Location**: `data/models/segmentation_models/M2F/Synthetic_Data/`
   - **Purpose**: Foundation model for all other models

2. **Example Material Models**:
   - **Segmentation**: GrapheneH Mask2Former model
   - **Classification**: GrapheneH AMM model
   - **Purpose**: Demonstration and testing

## Deployment Instructions

### CPU Deployment

1. **Start Services**:
   ```bash
   docker-compose -f docker-compose.cpu.yml up --build
   ```

2. **Access Points**:
   - **Web Interface**: `http://localhost:8000/`
   - **API Documentation**: `http://localhost:8000/api/docs`
   - **OpenAPI JSON**: `http://localhost:8000/openapi.json`

### GPU Deployment

1. **Start Services**:
   ```bash
   docker-compose -f docker-compose.cuda.yml up --build
   ```

2. **Requirements**:
   - NVIDIA Docker support
   - CUDA-compatible GPU

## Verification

### API Documentation Access
- ✅ `http://localhost:8000/api/docs` - Swagger UI loads correctly
- ✅ `http://localhost:8000/openapi.json` - OpenAPI specification accessible
- ✅ `http://localhost:8000/` - Web interface functional

### Model Availability
- ✅ Base segmentation model loaded
- ✅ Example material models available
- ✅ Model management endpoints functional

## Technical Details

### FastAPI Configuration
- **OpenAPI Version**: 3.0.2
- **API Version**: 1.0.0
- **Title**: MaskTerial API
- **Description**: A Foundation Model for Automated 2D Material Flake Detection

### Nginx Routing
- **API Routes**: `/api/*` → Backend service
- **OpenAPI Route**: `/openapi.json` → Backend service
- **Frontend Routes**: `/*` → React application

### Docker Services
1. **Backend**: FastAPI server (Python)
2. **Frontend Builder**: React application builder (Node.js)
3. **Nginx**: Reverse proxy server

## Root Cause Analysis

### Why These Issues Existed in Original Repository

1. **Development Environment Focus**: 
   - Developers likely tested directly against FastAPI without nginx proxy
   - Incomplete testing of full Docker deployment

2. **Priority Misalignment**:
   - Focus on core ML functionality over infrastructure
   - API documentation treated as secondary feature

3. **Common Deployment Pitfall**:
   - Microservice architecture with reverse proxy
   - Containerized deployment complexity

## Impact

### Before Fixes
- ❌ API documentation inaccessible
- ❌ Swagger UI error messages
- ❌ Poor developer experience

### After Fixes
- ✅ Complete API documentation access
- ✅ Interactive Swagger UI functionality
- ✅ Professional API interface
- ✅ Improved developer experience

## Maintenance Notes

### Future Considerations
1. **API Versioning**: Consider implementing proper API versioning strategy
2. **Documentation**: Maintain API documentation as code evolves
3. **Testing**: Include full Docker deployment in CI/CD pipeline
4. **Monitoring**: Add health checks for all services

### Dependencies
- FastAPI version compatibility
- Nginx configuration maintenance
- Docker service orchestration

## Conclusion

These fixes resolve critical deployment issues in the MaskTerial repository, ensuring proper functionality of the Docker-based inference server. The changes are minimal but essential for a professional deployment experience.

The fixes address both the immediate technical issues and provide a foundation for future development and maintenance of the MaskTerial API infrastructure.
