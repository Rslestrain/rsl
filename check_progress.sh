#!/bin/bash

# 批量训练进度查看脚本

echo "============================================================"
echo "STFT 批量训练进度监控"
echo "============================================================"
echo ""

# 查找最新的日志文件
LOG_FILE=$(ls -t batch_results/batch_stft_*.log 2>/dev/null | head -1)

if [ -z "$LOG_FILE" ]; then
    echo "❌ 未找到日志文件"
    echo "请先运行: ./run_batch_stft.sh background"
    exit 1
fi

echo "📄 日志文件: $LOG_FILE"
echo ""

# 检查进程是否运行
PID=$(ps aux | grep "python batch_train_stft.py" | grep -v grep | awk '{print $2}')

if [ -n "$PID" ]; then
    echo "✓ 进程运行中 (PID: $PID)"

    # 显示CPU和内存使用
    CPU=$(ps aux | grep $PID | grep -v grep | awk '{print $3}')
    MEM=$(ps aux | grep $PID | grep -v grep | awk '{print $4}')
    echo "  CPU: ${CPU}%  |  内存: ${MEM}%"
else
    echo "⚠ 进程未运行（可能已完成或出错）"
fi

echo ""
echo "------------------------------------------------------------"
echo "当前进度:"
echo "------------------------------------------------------------"

# 统计已完成的种子数
TOTAL_SEEDS=$(grep -c "Seed [0-9]*/100:" "$LOG_FILE" 2>/dev/null || echo "0")
COMPLETED=$(grep -c "Accuracy:" "$LOG_FILE" 2>/dev/null || echo "0")

echo "已运行种子: $TOTAL_SEEDS / 100"
echo "已完成评估: $COMPLETED / 100"

if [ "$COMPLETED" -gt 0 ]; then
    # 计算进度百分比
    PROGRESS=$((COMPLETED * 100 / 100))
    echo "进度: ${PROGRESS}%"

    echo ""
    echo "最近5次准确率:"
    tail -100 "$LOG_FILE" | grep "Accuracy:" | tail -5
fi

echo ""
echo "------------------------------------------------------------"
echo "实时查看: tail -f $LOG_FILE"
echo "停止运行: kill $PID"
echo "============================================================"
