@echo off
REM Maskterial AWS CDK 部署脚本 (Windows)

echo 🚀 开始部署 Maskterial 到 AWS...

REM 检查是否在cdk目录
if not exist "cdk.json" (
    echo ❌ 请在cdk目录中运行此脚本
    pause
    exit /b 1
)

REM 检查AWS CLI
where aws >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ AWS CLI 未安装，请先安装 AWS CLI
    pause
    exit /b 1
)

REM 检查CDK
where cdk >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ AWS CDK 未安装，正在安装...
    npm install -g aws-cdk
)

REM 检查Node.js
where node >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Node.js 未安装，请先安装 Node.js
    pause
    exit /b 1
)

echo ✅ 环境检查完成

REM 安装依赖
echo 📦 安装依赖...
npm install

REM 构建项目
echo 🔨 构建项目...
npm run build

REM 检查AWS配置
echo 🔍 检查AWS配置...
aws sts get-caller-identity >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ AWS CLI 未配置，请运行 'aws configure'
    pause
    exit /b 1
)

REM 初始化CDK
echo 🚀 初始化CDK...
cdk bootstrap

REM 显示部署计划
echo 📋 显示部署计划...
cdk diff

REM 确认部署
set /p confirm="是否继续部署？(y/N): "
if /i not "%confirm%"=="y" (
    echo ❌ 部署已取消
    pause
    exit /b 1
)

REM 部署
echo 🚀 开始部署...
cdk deploy

REM 显示输出
echo 📊 部署完成，显示输出...
cdk outputs

echo ✅ 部署完成！
echo 💰 预计月费用: $15-25
echo 🌐 访问应用: 查看上面的CloudFront URL
pause
