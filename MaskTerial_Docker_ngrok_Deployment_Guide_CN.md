# MaskTerial Docker + ngrok 部署文档

## 📋 **概述**

本文档详细说明如何在本地环境中使用Docker部署MaskTerial，并通过ngrok实现公网访问。

## 🎯 **目标**

- 在本地Docker环境中运行MaskTerial
- 通过ngrok实现公网访问
- 支持模型上传、下载、推理等功能
- 实现24/7稳定运行

## 🔧 **系统要求**

### **硬件要求**
- **CPU**: 2核心以上
- **内存**: 8GB以上
- **存储**: 20GB以上可用空间
- **网络**: 稳定的互联网连接

### **软件要求**
- **操作系统**: Windows 10/11, macOS, Linux
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **ngrok**: 3.0+

## 📦 **安装步骤**

### **1. 安装Docker**

#### **Windows**
```powershell
# 下载Docker Desktop
# 访问: https://www.docker.com/products/docker-desktop
# 安装并启动Docker Desktop
```

#### **macOS**
```bash
# 使用Homebrew安装
brew install --cask docker

# 或下载Docker Desktop
# 访问: https://www.docker.com/products/docker-desktop
```

#### **Linux (Ubuntu)**
```bash
# 更新包索引
sudo apt update

# 安装Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 安装Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### **2. 安装ngrok**

#### **下载ngrok**
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

#### **注册ngrok账号**
1. 访问: https://dashboard.ngrok.com/signup
2. 注册账号
3. 获取authtoken

#### **配置ngrok**
```bash
# 设置authtoken
ngrok config add-authtoken YOUR_AUTHTOKEN
```

## 🚀 **部署步骤**

### **1. 克隆项目**
```bash
git clone https://github.com/yourusername/MaskTerial.git
cd MaskTerial
```

### **2. 构建Docker镜像**
```bash
# 构建所有服务
docker compose -f docker-compose.cuda.yml build

# 或使用CPU版本
docker compose -f docker-compose.cpu.yml build
```

### **3. 启动服务**
```bash
# 启动所有服务
docker compose -f docker-compose.cuda.yml up -d

# 检查服务状态
docker compose -f docker-compose.cuda.yml ps
```

### **4. 配置ngrok**

#### **获取ngrok URL**
```bash
# 启动ngrok
ngrok http 8000

# 记录生成的URL，例如:
# https://abc123.ngrok-free.dev
```

#### **更新前端配置**
```yaml
# 修改 docker-compose.cuda.yml
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

#### **重新构建前端**
```bash
# 重新构建前端
docker compose -f docker-compose.cuda.yml up --build frontend_builder -d

# 重启nginx
docker compose -f docker-compose.cuda.yml restart nginx
```

## 🔧 **配置说明**

### **Docker Compose配置**

#### **服务架构**
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

#### **端口映射**
- **本地8000端口** → **ngrok隧道** → **公网访问**
- **Docker内部8000端口** → **本地8000端口**
- **Nginx 80端口** → **Docker内部8000端口**

### **环境变量说明**

| 变量名 | 作用 | 示例值 |
|--------|------|--------|
| `VITE_AVAILABLE_MODELS_URL` | 获取可用模型列表 | `https://abc123.ngrok-free.dev/api/available_models` |
| `VITE_INFERENCE_URL` | 模型推理API | `https://abc123.ngrok-free.dev/api/predict` |
| `VITE_DOWNLOAD_MODEL_URL` | 下载模型API | `https://abc123.ngrok-free.dev/api/download_model` |
| `VITE_GMM_UPLOAD_URL` | 上传GMM模型 | `https://abc123.ngrok-free.dev/api/upload/gmm` |
| `VITE_AMM_UPLOAD_URL` | 上传AMM模型 | `https://abc123.ngrok-free.dev/api/upload/amm` |
| `VITE_M2F_UPLOAD_URL` | 上传M2F模型 | `https://abc123.ngrok-free.dev/api/upload/m2f` |
| `VITE_DELETE_MODEL_URL` | 删除模型API | `https://abc123.ngrok-free.dev/api/delete_model` |
| `VITE_M2F_TRAIN_URL` | 训练M2F模型 | `https://abc123.ngrok-free.dev/api/train/m2f` |

## 🔍 **验证部署**

### **1. 检查服务状态**
```bash
# 检查Docker服务
docker compose -f docker-compose.cuda.yml ps

# 检查ngrok状态
curl http://localhost:4040/api/tunnels
```

### **2. 测试API**
```bash
# 测试本地API
curl http://localhost:8000/api/available_models

# 测试ngrok API
curl https://abc123.ngrok-free.dev/api/available_models
```

### **3. 测试前端**
1. 访问: `https://abc123.ngrok-free.dev`
2. 检查模型列表是否显示
3. 测试上传功能
4. 测试推理功能

## ⚙️ **系统配置**

### **Windows电源设置**
```
关闭显示器: 5分钟
使计算机进入睡眠状态: 从不
使计算机进入休眠状态: 从不
关闭硬盘: 从不
```

### **防火墙设置**
```powershell
# 允许8000端口
netsh advfirewall firewall add rule name="MaskTerial Port 8000" dir=in action=allow protocol=TCP localport=8000
```

### **自动启动脚本**
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

## 🔧 **维护操作**

### **常用命令**

#### **Docker操作**
```bash
# 查看服务状态
docker compose -f docker-compose.cuda.yml ps

# 查看日志
docker compose -f docker-compose.cuda.yml logs backend
docker compose -f docker-compose.cuda.yml logs nginx

# 重启服务
docker compose -f docker-compose.cuda.yml restart

# 停止服务
docker compose -f docker-compose.cuda.yml down

# 更新服务
docker compose -f docker-compose.cuda.yml up -d --build
```

#### **ngrok操作**
```bash
# 启动ngrok
ngrok http 8000

# 查看ngrok状态
curl http://localhost:4040/api/tunnels

# 停止ngrok
# 按 Ctrl+C
```

### **监控脚本**
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

## 🚨 **故障排除**

### **常见问题**

#### **1. Docker服务无法启动**
```bash
# 检查Docker状态
docker --version
docker compose --version

# 重启Docker服务
sudo systemctl restart docker
```

#### **2. ngrok连接失败**
```bash
# 检查网络连接
ping 8.8.8.8

# 检查ngrok配置
ngrok config check

# 重新配置authtoken
ngrok config add-authtoken YOUR_AUTHTOKEN
```

#### **3. 前端无法访问API**
- 检查环境变量配置
- 确认ngrok URL正确
- 重新构建前端

#### **4. 模型上传失败**
- 检查磁盘空间
- 检查文件权限
- 查看Docker日志

### **日志查看**
```bash
# 查看所有服务日志
docker compose -f docker-compose.cuda.yml logs

# 查看特定服务日志
docker compose -f docker-compose.cuda.yml logs backend
docker compose -f docker-compose.cuda.yml logs nginx
docker compose -f docker-compose.cuda.yml logs frontend_builder
```

## 📊 **性能优化**

### **Docker优化**
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

### **系统优化**
- 使用SSD存储
- 增加内存容量
- 优化网络连接
- 定期清理Docker镜像

## 🔒 **安全考虑**

### **网络安全**
- 使用HTTPS（ngrok自动提供）
- 配置防火墙规则
- 定期更新Docker镜像

### **数据安全**
- 定期备份数据目录
- 使用强密码
- 限制访问权限

## 📈 **扩展方案**

### **升级到Cloudflare Tunnel**
```bash
# 安装Cloudflare Tunnel
curl -L --output cloudflared.exe https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe

# 登录Cloudflare
cloudflared.exe tunnel login

# 创建隧道
cloudflared.exe tunnel create maskterial
```

### **升级到AWS部署**
- 使用EC2实例
- 配置负载均衡
- 设置自动扩展
- 使用RDS数据库

## 📝 **总结**

通过本文档，你可以：

1. ✅ 在本地Docker环境中部署MaskTerial
2. ✅ 使用ngrok实现公网访问
3. ✅ 配置24/7稳定运行
4. ✅ 支持所有模型操作功能
5. ✅ 实现集中式模型管理

**访问地址**: `https://your-ngrok-url.ngrok-free.dev`

**维护要点**:
- 保持Docker和ngrok运行
- 定期检查服务状态
- 备份重要数据
- 监控系统性能

---

*文档版本: 1.0*  
*最后更新: 2024年*  
*维护者: MaskTerial团队*
