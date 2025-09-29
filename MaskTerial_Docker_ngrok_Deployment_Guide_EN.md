# MaskTerial Docker + ngrok Deployment Guide

## 📋 **Overview**

This document provides detailed instructions for deploying MaskTerial in a local Docker environment and enabling public access through ngrok.

## 🎯 **Objectives**

- Run MaskTerial in a local Docker environment
- Enable public access through ngrok
- Support model upload, download, and inference functionality
- Achieve 24/7 stable operation

## 🔧 **System Requirements**

### **Hardware Requirements**
- **CPU**: 2+ cores
- **Memory**: 8GB+ RAM
- **Storage**: 20GB+ available space
- **Network**: Stable internet connection

### **Software Requirements**
- **Operating System**: Windows 10/11, macOS, Linux
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **ngrok**: 3.0+

## 📦 **Installation Steps**

### **1. Install Docker**

#### **Windows**
```powershell
# Download Docker Desktop
# Visit: https://www.docker.com/products/docker-desktop
# Install and start Docker Desktop
```

#### **macOS**
```bash
# Install using Homebrew
brew install --cask docker

# Or download Docker Desktop
# Visit: https://www.docker.com/products/docker-desktop
```

#### **Linux (Ubuntu)**
```bash
# Update package index
sudo apt update

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### **2. Install ngrok**

#### **Download ngrok**
```bash
# Windows
curl -o ngrok.zip https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-windows-amd64.zip
Expand-Archive -Path ngrok.zip -DestinationPath . -Force

# macOS
brew install ngrok/ngrok/ngrok

# Linux
curl -o ngrok.zip https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.zip
unzip ngrok.zip
```

#### **Register ngrok Account**
1. Visit: https://dashboard.ngrok.com/signup
2. Create an account
3. Get your authtoken

#### **Configure ngrok**
```bash
# Set authtoken
ngrok config add-authtoken YOUR_AUTHTOKEN
```

## 🚀 **Deployment Steps**

### **1. Clone the Project**
```bash
git clone https://github.com/yourusername/MaskTerial.git
cd MaskTerial
```

### **2. Build Docker Images**
```bash
# Build all services
docker compose -f docker-compose.cuda.yml build

# Or use CPU version
docker compose -f docker-compose.cpu.yml build
```

### **3. Start Services**
```bash
# Start all services
docker compose -f docker-compose.cuda.yml up -d

# Check service status
docker compose -f docker-compose.cuda.yml ps
```

### **4. Configure ngrok**

#### **Get ngrok URL**
```bash
# Start ngrok
ngrok http 8000

# Record the generated URL, e.g.:
# https://abc123.ngrok-free.dev
```

#### **Update Frontend Configuration**
```yaml
# Modify docker-compose.cuda.yml
frontend_builder:
  environment:
    - VITE_AVAILABLE_MODELS_URL=https://abc123.ngrok-free.dev/api/available_models
    - VITE_INFERENCE_URL=https://abc123.ngrok-free.dev/api/predict
    - VITE_DOWNLOAD_MODEL_URL=https://abc123.ngrok-free.dev/api/download_model
    - VITE_GMM_UPLOAD_URL=https://abc123.ngrok-free.dev/api/upload/gmm
    - VITE_AMM_UPLOAD_URL=https://abc123.ngrok-free.dev/api/upload/amm
    - VITE_M2F_UPLOAD_URL=https://abc123.ngrok-free.dev/api/upload/m2f
    - VITE_DELETE_MODEL_URL=https://abc123.ngrok-free.dev/api/delete_model
    - VITE_M2F_TRAIN_URL=https://abc123.ngrok-free.dev/api/train/m2f
```

#### **Rebuild Frontend**
```bash
# Rebuild frontend
docker compose -f docker-compose.cuda.yml up --build frontend_builder -d

# Restart nginx
docker compose -f docker-compose.cuda.yml restart nginx
```

## 🔧 **Configuration Details**

### **Docker Compose Configuration**

#### **Service Architecture**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   frontend      │    │     backend     │    │     nginx       │
│   (React/Vite)  │    │   (FastAPI)     │    │  (Reverse Proxy)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │     ngrok       │
                    │  (Tunnel)       │
                    └─────────────────┘
```

#### **Port Mapping**
- **Local 8000 port** → **ngrok tunnel** → **Public access**
- **Docker internal 8000 port** → **Local 8000 port**
- **Nginx 80 port** → **Docker internal 8000 port**

### **Environment Variables**

| Variable Name | Purpose | Example Value |
|---------------|---------|---------------|
| `VITE_AVAILABLE_MODELS_URL` | Get available models list | `https://abc123.ngrok-free.dev/api/available_models` |
| `VITE_INFERENCE_URL` | Model inference API | `https://abc123.ngrok-free.dev/api/predict` |
| `VITE_DOWNLOAD_MODEL_URL` | Download model API | `https://abc123.ngrok-free.dev/api/download_model` |
| `VITE_GMM_UPLOAD_URL` | Upload GMM model | `https://abc123.ngrok-free.dev/api/upload/gmm` |
| `VITE_AMM_UPLOAD_URL` | Upload AMM model | `https://abc123.ngrok-free.dev/api/upload/amm` |
| `VITE_M2F_UPLOAD_URL` | Upload M2F model | `https://abc123.ngrok-free.dev/api/upload/m2f` |
| `VITE_DELETE_MODEL_URL` | Delete model API | `https://abc123.ngrok-free.dev/api/delete_model` |
| `VITE_M2F_TRAIN_URL` | Train M2F model | `https://abc123.ngrok-free.dev/api/train/m2f` |

## 🔍 **Deployment Verification**

### **1. Check Service Status**
```bash
# Check Docker services
docker compose -f docker-compose.cuda.yml ps

# Check ngrok status
curl http://localhost:4040/api/tunnels
```

### **2. Test APIs**
```bash
# Test local API
curl http://localhost:8000/api/available_models

# Test ngrok API
curl https://abc123.ngrok-free.dev/api/available_models
```

### **3. Test Frontend**
1. Visit: `https://abc123.ngrok-free.dev`
2. Check if model list is displayed
3. Test upload functionality
4. Test inference functionality

## ⚙️ **System Configuration**

### **Windows Power Settings**
```
Turn off display: 5 minutes
Put computer to sleep: Never
Put computer to hibernation: Never
Turn off hard disk: Never
```

### **Firewall Settings**
```powershell
# Allow port 8000
netsh advfirewall firewall add rule name="MaskTerial Port 8000" dir=in action=allow protocol=TCP localport=8000
```

### **Auto-start Script**
```batch
@echo off
cd /d C:\path\to\MaskTerial

echo Starting Docker services...
docker compose -f docker-compose.cuda.yml up -d

echo Waiting for Docker to start...
timeout /t 30 /nobreak >nul

echo Starting ngrok...
start "ngrok" ngrok.exe http 8000

echo All services started!
pause
```

## 🔧 **Maintenance Operations**

### **Common Commands**

#### **Docker Operations**
```bash
# View service status
docker compose -f docker-compose.cuda.yml ps

# View logs
docker compose -f docker-compose.cuda.yml logs backend
docker compose -f docker-compose.cuda.yml logs nginx

# Restart services
docker compose -f docker-compose.cuda.yml restart

# Stop services
docker compose -f docker-compose.cuda.yml down

# Update services
docker compose -f docker-compose.cuda.yml up -d --build
```

#### **ngrok Operations**
```bash
# Start ngrok
ngrok http 8000

# Check ngrok status
curl http://localhost:4040/api/tunnels

# Stop ngrok
# Press Ctrl+C
```

### **Monitoring Script**
```batch
@echo off
:loop
echo %date% %time% - Checking services...

REM Check Docker
docker compose -f docker-compose.cuda.yml ps | findstr "Up" >nul
if errorlevel 1 (
    echo Docker services down, restarting...
    docker compose -f docker-compose.cuda.yml up -d
)

REM Check ngrok
tasklist | findstr "ngrok.exe" >nul
if errorlevel 1 (
    echo ngrok down, restarting...
    start "ngrok" ngrok.exe http 8000
)

timeout /t 300 /nobreak >nul
goto loop
```

## 🚨 **Troubleshooting**

### **Common Issues**

#### **1. Docker Services Won't Start**
```bash
# Check Docker status
docker --version
docker compose --version

# Restart Docker service
sudo systemctl restart docker
```

#### **2. ngrok Connection Failed**
```bash
# Check network connection
ping 8.8.8.8

# Check ngrok configuration
ngrok config check

# Reconfigure authtoken
ngrok config add-authtoken YOUR_AUTHTOKEN
```

#### **3. Frontend Can't Access API**
- Check environment variable configuration
- Verify ngrok URL is correct
- Rebuild frontend

#### **4. Model Upload Failed**
- Check disk space
- Check file permissions
- View Docker logs

### **Log Viewing**
```bash
# View all service logs
docker compose -f docker-compose.cuda.yml logs

# View specific service logs
docker compose -f docker-compose.cuda.yml logs backend
docker compose -f docker-compose.cuda.yml logs nginx
docker compose -f docker-compose.cuda.yml logs frontend_builder
```

## 📊 **Performance Optimization**

### **Docker Optimization**
```yaml
# docker-compose.cuda.yml
services:
  backend:
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
```

### **System Optimization**
- Use SSD storage
- Increase memory capacity
- Optimize network connection
- Regularly clean Docker images

## 🔒 **Security Considerations**

### **Network Security**
- Use HTTPS (automatically provided by ngrok)
- Configure firewall rules
- Regularly update Docker images

### **Data Security**
- Regularly backup data directory
- Use strong passwords
- Limit access permissions

## 📈 **Scaling Options**

### **Upgrade to Cloudflare Tunnel**
```bash
# Install Cloudflare Tunnel
curl -L --output cloudflared.exe https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe

# Login to Cloudflare
cloudflared.exe tunnel login

# Create tunnel
cloudflared.exe tunnel create maskterial
```

### **Upgrade to AWS Deployment**
- Use EC2 instances
- Configure load balancer
- Set up auto-scaling
- Use RDS database

## 📝 **Summary**

With this guide, you can:

1. ✅ Deploy MaskTerial in a local Docker environment
2. ✅ Enable public access through ngrok
3. ✅ Configure 24/7 stable operation
4. ✅ Support all model operation functionality
5. ✅ Achieve centralized model management

**Access URL**: `https://your-ngrok-url.ngrok-free.dev`

**Maintenance Points**:
- Keep Docker and ngrok running
- Regularly check service status
- Backup important data
- Monitor system performance

---

*Document Version: 1.0*  
*Last Updated: 2024*  
*Maintainer: MaskTerial Team*
