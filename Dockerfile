# 使用官方 Python 3.10 镜像作为基础
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 复制依赖文件并安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码、模型和脚本
COPY . /app

# 暴露 FastAPI 运行的端口
EXPOSE 8080

# 启动服务
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]