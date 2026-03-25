#!/bin/bash

# STFT数据集批量训练启动脚本
# 使用100个随机种子，计算Top5平均准确率

echo "=========================================="
echo "STFT批量训练脚本"
echo "数据集: MMFI-STFT, Wiar-STFT"
echo "种子数: 100"
echo "=========================================="
echo ""

# 使用conda环境的Python
PYTHON=/data1/rsl/anaconda3/envs/consense/bin/python

# 检查是否要后台运行
if [ "$1" == "background" ] || [ "$1" == "bg" ]; then
    echo "以后台模式运行..."
    nohup $PYTHON batch_train_stft.py > batch_stft.log 2>&1 &
    PID=$!
    echo "进程ID: $PID"
    echo "日志文件: batch_stft.log"
    echo ""
    echo "查看进度: tail -f batch_stft.log"
    echo "停止运行: kill $PID"
else
    echo "以前台模式运行..."
    echo "提示: 如需后台运行，使用: ./run_batch_stft.sh background"
    echo ""
    $PYTHON batch_train_stft.py
fi
