#!/bin/bash

echo "🚀 CPU专用部署脚本"

# 检查Docker是否安装
if ! command -v docker &> /dev/null; then
    echo "❌ Docker未安装，请先安装Docker"
    exit 1
fi

# 检查Docker Compose是否安装
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose未安装，请先安装Docker Compose"
    exit 1
fi

echo "✅ 检测到CPU环境，使用CPU优化版本"

# 构建镜像
echo "📦 构建CPU优化镜像..."
docker-compose build mlsj-cpu

# 启动服务
echo "🚀 启动CPU优化服务..."
docker-compose up -d mlsj-cpu

# 等待服务启动
echo "⏳ 等待服务启动..."
sleep 15

# 检查服务状态
if docker-compose ps | grep -q "Up"; then
    echo "✅ CPU版本部署成功！"
    echo "🌐 访问地址: http://localhost:8000"
    echo "📚 API文档: http://localhost:8000/docs"
    echo ""
    echo "📊 查看服务状态: docker-compose ps"
    echo "📋 查看日志: docker-compose logs -f mlsj-cpu"
    echo "🛑 停止服务: docker-compose down"
else
    echo "❌ 服务启动失败，请检查日志: docker-compose logs mlsj-cpu"
    exit 1
fi
