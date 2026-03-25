#!/bin/bash
# 自动监控Wiar训练完成后启动MMFI训练

WIAR_PID=2777717
MMFI_SCRIPT="batch_train_mmfi_stft.py"
LOG_DIR="batch_results"

echo "============================================================"
echo "自动启动脚本 - 等待Wiar完成后启动MMFI"
echo "============================================================"
echo "Wiar PID: $WIAR_PID"
echo "启动时间: $(date)"
echo ""

# 检查Wiar进程是否存在
if ! ps -p $WIAR_PID > /dev/null 2>&1; then
    echo "⚠ Wiar进程不存在，立即启动MMFI训练"
    /data1/rsl/anaconda3/envs/consense/bin/python $MMFI_SCRIPT > $LOG_DIR/mmfi_stft_100runs_$(date +%Y%m%d_%H%M%S).log 2>&1 &
    NEW_PID=$!
    echo "✓ MMFI训练已启动 (PID: $NEW_PID)"
    exit 0
fi

# 等待Wiar完成
echo "等待Wiar训练完成..."
while ps -p $WIAR_PID > /dev/null 2>&1; do
    # 显示当前进度
    COMPLETED=$(grep -c "Accuracy:" $LOG_DIR/wiar_stft_100runs_20251121_162600.log 2>/dev/null || echo "0")
    echo "[$(date +%H:%M:%S)] Wiar进度: $COMPLETED/100"

    sleep 60  # 每分钟检查一次
done

echo ""
echo "✓ Wiar训练已完成！"
echo ""

# 等待GPU内存释放
echo "等待GPU内存释放（30秒）..."
sleep 30

# 启动MMFI训练
echo "启动MMFI训练..."
/data1/rsl/anaconda3/envs/consense/bin/python $MMFI_SCRIPT > $LOG_DIR/mmfi_stft_100runs_$(date +%Y%m%d_%H%M%S).log 2>&1 &
NEW_PID=$!

echo ""
echo "============================================================"
echo "✓ MMFI训练已启动"
echo "============================================================"
echo "PID: $NEW_PID"
echo "启动时间: $(date)"
echo "日志文件: $LOG_DIR/mmfi_stft_100runs_$(date +%Y%m%d_%H%M%S).log"
echo ""
echo "监控命令:"
echo "  tail -f $LOG_DIR/mmfi_stft_100runs_*.log"
echo "  ps aux | grep $NEW_PID"
echo "============================================================"
