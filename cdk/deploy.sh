#!/bin/bash

# Maskterial AWS CDK 部署脚本

echo "🚀 开始部署 Maskterial 到 AWS..."

# 检查是否在cdk目录
if [ ! -f "cdk.json" ]; then
    echo "❌ 请在cdk目录中运行此脚本"
    exit 1
fi

# 检查AWS CLI
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI 未安装，请先安装 AWS CLI"
    exit 1
fi

# 检查CDK
if ! command -v cdk &> /dev/null; then
    echo "❌ AWS CDK 未安装，正在安装..."
    npm install -g aws-cdk
fi

# 检查Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js 未安装，请先安装 Node.js"
    exit 1
fi

echo "✅ 环境检查完成"

# 安装依赖
echo "📦 安装依赖..."
npm install

# 构建项目
echo "🔨 构建项目..."
npm run build

# 检查AWS配置
echo "🔍 检查AWS配置..."
aws sts get-caller-identity > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ AWS CLI 未配置，请运行 'aws configure'"
    exit 1
fi

# 初始化CDK（如果需要）
echo "🚀 初始化CDK..."
cdk bootstrap

# 显示部署计划
echo "📋 显示部署计划..."
cdk diff

# 确认部署
read -p "是否继续部署？(y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ 部署已取消"
    exit 1
fi

# 部署
echo "🚀 开始部署..."
cdk deploy

# 显示输出
echo "📊 部署完成，显示输出..."
cdk outputs

echo "✅ 部署完成！"
echo "💰 预计月费用: $15-25"
echo "🌐 访问应用: 查看上面的CloudFront URL"
