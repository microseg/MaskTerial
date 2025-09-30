# Maskterial AWS CDK 预算部署指南

## 💰 **成本优化配置**

这个CDK配置专门为省钱优化，月费用约 **$15-25**。

### **成本优化策略**
- ✅ **单AZ部署**：只用一个可用区
- ✅ **无NAT Gateway**：使用公网子网
- ✅ **最小ECS配置**：1 vCPU, 1GB RAM
- ✅ **单实例运行**：不自动扩展
- ✅ **最便宜CloudFront**：只用最便宜区域
- ✅ **短期日志保留**：只保留1周日志

## 🚀 **快速部署**

### **1. 初始化项目**
```bash
# 安装CDK
npm install -g aws-cdk

# 创建项目目录
mkdir maskterial-budget-cdk
cd maskterial-budget-cdk

# 复制文件
cp ../lib/maskterial-budget-stack.ts ./lib/
cp ../bin/maskterial-budget.ts ./bin/
cp ../Dockerfile.aws ./
cp ../package.json ./
cp ../cdk.json ./
```

### **2. 安装依赖**
```bash
npm install
```

### **3. 配置AWS**
```bash
# 配置AWS CLI
aws configure

# 初始化CDK
cdk bootstrap
```

### **4. 部署**
```bash
# 构建
npm run build

# 部署
cdk deploy

# 查看输出
cdk outputs
```

## 📊 **成本明细**

| 服务 | 配置 | 月费用 |
|------|------|--------|
| **ECS Fargate** | 1 vCPU, 1GB RAM | $8-12 |
| **ALB** | 标准负载均衡器 | $16 |
| **CloudFront** | 最便宜配置 | $1-2 |
| **S3** | 标准存储 | $1-3 |
| **总计** | | **$15-25** |

## 🔧 **配置说明**

### **ECS配置**
```typescript
// 最小配置
memoryLimitMiB: 1024, // 1GB内存
cpu: 512, // 0.5 vCPU
desiredCount: 1, // 只运行1个实例
```

### **VPC配置**
```typescript
// 单AZ省钱配置
maxAzs: 1, // 只用一个AZ
natGateways: 0, // 不用NAT Gateway
```

### **CloudFront配置**
```typescript
// 最便宜配置
priceClass: cloudfront.PriceClass.PRICE_CLASS_100, // 只用最便宜区域
```

## 🎯 **部署后操作**

### **1. 获取访问URL**
```bash
# 获取CloudFront URL
aws cloudformation describe-stacks \
  --stack-name MaskterialBudgetStack \
  --query 'Stacks[0].Outputs[?OutputKey==`CloudFrontURL`].OutputValue' \
  --output text
```

### **2. 配置域名（可选）**
```bash
# 在Route 53或DNS提供商添加CNAME记录
# 指向CloudFront域名
```

### **3. 监控成本**
```bash
# 设置成本告警
aws budgets create-budget \
  --account-id YOUR_ACCOUNT_ID \
  --budget '{
    "BudgetName": "Maskterial-Budget",
    "BudgetLimit": {
      "Amount": "30",
      "Unit": "USD"
    },
    "TimeUnit": "MONTHLY",
    "BudgetType": "COST"
  }'
```

## 🔍 **监控和维护**

### **查看服务状态**
```bash
# 查看ECS服务
aws ecs describe-services \
  --cluster maskterial-cluster \
  --services maskterial-service

# 查看CloudFront分布
aws cloudfront get-distribution \
  --id YOUR_DISTRIBUTION_ID
```

### **查看日志**
```bash
# 查看ECS日志
aws logs describe-log-groups \
  --log-group-name-prefix /aws/ecs/maskterial
```

## ⚠️ **注意事项**

### **性能限制**
- **单实例**：高并发时可能性能不足
- **单AZ**：可用性较低
- **最小内存**：大模型可能内存不足

### **扩展建议**
如果使用量增加，可以：
1. **增加内存**：1024MB → 2048MB
2. **增加CPU**：512 → 1024
3. **启用自动扩展**：min: 1, max: 3
4. **多AZ部署**：提高可用性

## 🚨 **故障排除**

### **常见问题**
1. **内存不足**：增加ECS内存配置
2. **CPU不足**：增加ECS CPU配置
3. **网络问题**：检查安全组配置
4. **存储问题**：检查S3权限

### **成本控制**
```bash
# 设置成本告警
aws budgets create-budget-notification \
  --account-id YOUR_ACCOUNT_ID \
  --budget-name Maskterial-Budget \
  --notification '{
    "NotificationType": "ACTUAL",
    "ComparisonOperator": "GREATER_THAN",
    "Threshold": 80,
    "ThresholdType": "PERCENTAGE"
  }'
```

## 📈 **升级路径**

### **阶段1：基础版（当前）**
- 成本：$15-25/月
- 适用：个人使用，低并发

### **阶段2：标准版**
- 成本：$40-60/月
- 配置：2 vCPU, 4GB RAM, 多AZ

### **阶段3：企业版**
- 成本：$100-200/月
- 配置：自动扩展，RDS，监控

## 🎉 **总结**

这个CDK配置提供了：
- ✅ **最低成本**：$15-25/月
- ✅ **快速部署**：一键部署
- ✅ **生产就绪**：包含所有必要组件
- ✅ **易于扩展**：可以随时升级

**开始使用**：
```bash
cdk deploy
```

**访问应用**：使用CloudFront URL即可访问Maskterial！
