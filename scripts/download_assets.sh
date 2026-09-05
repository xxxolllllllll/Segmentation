#!/usr/bin/env bash

# 发生错误时立即停止运行
set -e

echo "=== 1. 检查并安装依赖库 ==="
python3 -c "import huggingface_hub" 2>/dev/null || pip install huggingface_hub

echo "=== 2. 创建本地存放目录 ==="
mkdir -p weights data

echo "=== 3. 下载模型权重到 ./weights/ ==="
python3 -c "
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='AIzero2hero123/paper_repo_weights',
    repo_type='model',
    local_dir='./weights',
    ignore_patterns=['.git*', 'README.md'] # 过滤掉不需要的 Git 配置和说明文件
)
"

echo "=== 4. 下载数据集到 ./data/ ==="
python3 -c "
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='AIzero2hero123/paper_repo_data',
    repo_type='dataset',
    local_dir='./data',
    ignore_patterns=['.git*', 'README.md'] # 过滤掉不需要的 Git 配置和说明文件
)
"

echo "=== 所有资源下载成功！==="
