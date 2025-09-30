# Maskterial AWS CDK Deployment

这个文件夹包含将Maskterial部署到AWS的所有CDK配置文件和脚本。

## 📁 **文件结构**

```
cdk/
├── bin/
│   └── maskterial-budget.ts          # CDK应用入口
├── lib/
│   └── maskterial-budget-stack.ts    # 主CDK栈配置
├── Dockerfile.aws                    # 优化的Docker镜像
├── package.json                      # 项目依赖
├── cdk.json                          # CDK配置
└── README_Budget_Deployment.md       # 详细部署指南
```

## 🚀 **快速开始**

### **1. 进入cdk目录**
```bash
cd cdk
```

### **2. 安装依赖**
```bash
npm install
```

### **3. 配置AWS**
```bash
aws configure
```

### **4. 初始化CDK**
```bash
cdk bootstrap
```

### **5. 部署**
```bash
cdk deploy
```

## 💰 **成本**

- **月费用**: $15-25
- **架构**: ECS Fargate + S3 + CloudFront
- **优化**: 单AZ，最小资源配置

## 📖 **详细文档**

查看 `README_Budget_Deployment.md` 获取完整的部署指南和配置说明。

## 🔧 **常用命令**

```bash
# 部署
cdk deploy

# 查看差异
cdk diff

# 销毁资源
cdk destroy

# 查看输出
cdk outputs
```

## ⚠️ **注意事项**

- 确保AWS CLI已配置
- 确保有足够的AWS权限
- 部署前请仔细阅读成本说明
