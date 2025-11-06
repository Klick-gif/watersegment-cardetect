#!/bin/bash

echo "🐧 Linux服务器CPU专用部署脚本"

# 安装Docker（如果未安装）
if ! command -v docker &> /dev/null; then
    echo "📥 安装Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
    echo "✅ Docker安装完成"
fi

# 安装Docker Compose（如果未安装）
if ! command -v docker-compose &> /dev/null; then
    echo "📥 安装Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo "✅ Docker Compose安装完成"
fi

# 重启Docker服务
echo "🔄 重启Docker服务..."
sudo systemctl restart docker

# 等待Docker服务启动
sleep 5

# 执行CPU专用部署
./deploy-cpu.sh
