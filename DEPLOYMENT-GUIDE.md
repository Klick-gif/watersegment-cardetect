# 积水识别系统 - CPU专用部署指南

## 部署方式选择

### 方式一：一键部署（推荐）
```bash
# Linux服务器
chmod +x deploy-cpu-linux.sh
./deploy-cpu-linux.sh

# Windows本地
双击运行 deploy-cpu-windows.bat
```

### 方式二：手动部署
```bash
# 1. 构建镜像
docker-compose build mlsj-cpu

# 2. 启动服务
docker-compose up -d mlsj-cpu

# 3. 查看状态
docker-compose ps

# 4. 查看日志
docker-compose logs -f mlsj-cpu
```

## 访问方式

- **Web界面**: http://localhost:8000
- **API文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/health

## 管理命令

```bash
# 停止服务
docker-compose down

# 重启服务
docker-compose restart mlsj-cpu

# 更新服务（重新构建）
docker-compose up -d --build mlsj-cpu

# 查看资源使用
docker stats mlsj-cpu
```

## 性能优化说明

1. **CPU专用**: 强制使用CPU，禁用GPU相关功能
2. **内存限制**: 容器内存限制为4GB
3. **线程优化**: 限制PyTorch线程数以优化CPU性能
4. **轻量镜像**: 使用slim基础镜像减少镜像大小

## 故障排除

### 常见问题

1. **端口占用**: 修改docker-compose.yml中的端口映射
2. **内存不足**: 增加docker-compose.yml中的内存限制
3. **模型加载慢**: 首次加载需要时间，后续会缓存

### 日志查看
```bash
# 实时查看日志
docker-compose logs -f mlsj-cpu

# 查看最近100行日志
docker-compose logs --tail=100 mlsj-cpu
```

## 文件结构说明

- `uploads/`: 用户上传的图片
- `results/`: 预测结果图片
- `detect/`: 检测模型文件
- `segment/`: 分割模型文件
- `segNet/`: 分割网络模型文件