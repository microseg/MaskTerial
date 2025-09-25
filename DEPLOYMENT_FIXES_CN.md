# MaskTerial 部署修复文档

## 概述

本文档详细记录了针对MaskTerial仓库的部署问题修复，确保基于Docker的推理服务器能够正常运行。

## 问题识别

### 1. FastAPI Swagger UI 错误
**问题描述**: 访问 `http://localhost:8000/api/docs` 时出现错误：
```
无法渲染此定义
提供的定义未指定有效的版本字段。
```

**根本原因**: FastAPI应用程序初始化时缺少必要的元数据配置。

### 2. Nginx 路由配置问题
**问题描述**: Swagger UI 无法访问 OpenAPI JSON 规范文件。

**根本原因**: nginx 配置中缺少 `/openapi.json` 端点的路由规则。

## 修复方案

### 1. FastAPI 配置修复 (`server.py`)

**原始代码**:
```python
app = FastAPI()
```

**修复后代码**:
```python
app = FastAPI(
    title="MaskTerial API",
    description="A Foundation Model for Automated 2D Material Flake Detection",
    version="1.0.0",
    openapi_version="3.0.2"
)
```

**修复目的**: 
- 为 Swagger UI 提供完整的 API 元数据
- 确保 OpenAPI 规范包含必需的版本字段
- 提升 API 文档质量

### 2. Nginx 配置修复 (`etc/nginx.conf`)

**原始配置**:
```nginx
location /api/ {
    proxy_pass http://backend:8000/;
}

location / {
    root   /usr/share/nginx/html;
    try_files $uri $uri/ /index.html;
}
```

**修复后配置**:
```nginx
location /api/ {
    proxy_pass http://backend:8000/;
}

# 直接提供 OpenAPI JSON 服务
location /openapi.json {
    proxy_pass http://backend:8000/openapi.json;
}

location / {
    root   /usr/share/nginx/html;
    try_files $uri $uri/ /index.html;
}
```

**修复目的**:
- 使 Swagger UI 能够访问 OpenAPI JSON 规范
- 为 `/openapi.json` 端点提供直接路由
- 保持现有的 API 和前端路由不变

### 3. 数据目录结构创建

**创建的目录结构**:
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

**创建目的**:
- 为预训练模型提供存储位置
- 按类型和材料组织模型
- 实现正确的模型管理

## 模型下载

### 已下载的模型

1. **基础分割模型**:
   - **模型**: Mask2Former (合成数据)
   - **位置**: `data/models/segmentation_models/M2F/Synthetic_Data/`
   - **用途**: 所有其他模型的基础模型

2. **示例材料模型**:
   - **分割模型**: GrapheneH Mask2Former 模型
   - **分类模型**: GrapheneH AMM 模型
   - **用途**: 演示和测试

## 部署说明

### CPU 部署

1. **启动服务**:
   ```bash
   docker-compose -f docker-compose.cpu.yml up --build
   ```

2. **访问地址**:
   - **Web 界面**: `http://localhost:8000/`
   - **API 文档**: `http://localhost:8000/api/docs`
   - **OpenAPI JSON**: `http://localhost:8000/openapi.json`

### GPU 部署

1. **启动服务**:
   ```bash
   docker-compose -f docker-compose.cuda.yml up --build
   ```

2. **系统要求**:
   - NVIDIA Docker 支持
   - CUDA 兼容的 GPU

## 验证测试

### API 文档访问
- ✅ `http://localhost:8000/api/docs` - Swagger UI 正常加载
- ✅ `http://localhost:8000/openapi.json` - OpenAPI 规范可访问
- ✅ `http://localhost:8000/` - Web 界面功能正常

### 模型可用性
- ✅ 基础分割模型已加载
- ✅ 示例材料模型可用
- ✅ 模型管理端点功能正常

## 技术细节

### FastAPI 配置
- **OpenAPI 版本**: 3.0.2
- **API 版本**: 1.0.0
- **标题**: MaskTerial API
- **描述**: 用于自动化2D材料薄片检测的基础模型

### Nginx 路由
- **API 路由**: `/api/*` → 后端服务
- **OpenAPI 路由**: `/openapi.json` → 后端服务
- **前端路由**: `/*` → React 应用

### Docker 服务
1. **后端**: FastAPI 服务器 (Python)
2. **前端构建器**: React 应用构建器 (Node.js)
3. **Nginx**: 反向代理服务器

## 根本原因分析

### 为什么原始仓库存在这些问题

1. **开发环境导向**: 
   - 开发者可能直接测试 FastAPI 而不通过 nginx 代理
   - 未完整测试 Docker 部署环境

2. **优先级错位**:
   - 重点关注核心 ML 功能而非基础设施
   - API 文档被视为次要功能

3. **常见部署陷阱**:
   - 微服务架构配合反向代理
   - 容器化部署的复杂性

## 影响评估

### 修复前
- ❌ API 文档无法访问
- ❌ Swagger UI 错误信息
- ❌ 开发者体验差

### 修复后
- ✅ 完整的 API 文档访问
- ✅ 交互式 Swagger UI 功能
- ✅ 专业的 API 界面
- ✅ 改善的开发者体验

## 维护说明

### 未来考虑事项
1. **API 版本控制**: 考虑实施适当的 API 版本控制策略
2. **文档维护**: 随着代码演进维护 API 文档
3. **测试**: 在 CI/CD 流水线中包含完整的 Docker 部署测试
4. **监控**: 为所有服务添加健康检查

### 依赖关系
- FastAPI 版本兼容性
- Nginx 配置维护
- Docker 服务编排

## API 功能说明

### 主要 API 端点

1. **模型管理**:
   - `GET /api/available_models` - 查看可用模型
   - `POST /api/upload/amm` - 上传 AMM 模型
   - `POST /api/upload/gmm` - 上传 GMM 模型
   - `POST /api/upload/m2f` - 上传 M2F 模型
   - `GET /api/download_model` - 下载模型
   - `POST /api/delete_model` - 删除模型

2. **图像推理**:
   - `POST /api/predict` - 图像检测和分类

3. **模型训练**:
   - `POST /api/train/m2f` - 训练 M2F 模型

4. **系统状态**:
   - `GET /api/` - 检查服务器状态
   - `GET /api/status` - 检查训练状态

### API 网页功能
- **交互式测试**: 直接在网页上测试每个 API 端点
- **参数输入**: 通过表单输入参数
- **文件上传**: 直接上传图像文件进行测试
- **代码生成**: 自动生成各种编程语言的调用代码
- **响应示例**: 详细的请求/响应数据结构

## 结论

这些修复解决了 MaskTerial 仓库中的关键部署问题，确保了基于 Docker 的推理服务器的正常运行。修改虽然简单但对专业部署体验至关重要。

修复不仅解决了当前的技术问题，还为 MaskTerial API 基础设施的未来开发和维护奠定了基础。

## 使用建议

1. **开发环境**: 建议在开发时也使用完整的 Docker 环境进行测试
2. **API 文档**: 定期更新 API 文档以保持与代码同步
3. **模型管理**: 利用 API 端点进行模型的生命周期管理
4. **批量处理**: 通过 API 实现自动化批量图像处理工作流
