#!/bin/bash
# MMFI按人划分100次训练 - 后台运行脚本

echo "=========================================="
echo "MMFI按人划分批量训练 - 方案A"
echo "=========================================="
echo "配置: configs/mmfi_by_person_stft.json"
echo "训练次数: 100次"
echo "=========================================="
echo ""

# 创建日志目录
mkdir -p batch_results

# 生成日志文件名
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="batch_results/mmfi_by_person_100runs_${TIMESTAMP}.log"

echo "日志文件: $LOG_FILE"
echo ""

# 确认是否继续
read -p "确认开始训练? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "已取消"
    exit 1
fi

echo "开始训练..."
echo "可以使用以下命令查看进度:"
echo "  tail -f $LOG_FILE"
echo ""

# 后台运行
nohup python3 batch_train_mmfi_by_person.py > "$LOG_FILE" 2>&1 &

PID=$!
echo "训练已在后台启动!"
echo "进程ID: $PID"
echo "日志文件: $LOG_FILE"
echo ""
echo "查看实时日志:"
echo "  tail -f $LOG_FILE"
echo ""
echo "检查进程状态:"
echo "  ps -p $PID"
echo ""
echo "停止训练:"
echo "  kill $PID"
