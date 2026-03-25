#!/bin/bash
# MMFI STFT批量训练监控脚本

echo "========================================"
echo "MMFI STFT (无拼接) 批量训练监控"
echo "========================================"
echo ""

# 检查批量训练进程
BATCH_PID=$(ps aux | grep "batch_train_mmfi_stft.py" | grep -v grep | awk '{print $2}')

if [ -z "$BATCH_PID" ]; then
    echo "❌ 批量训练未运行"
    exit 1
else
    echo "✅ 批量训练运行中 (PID: $BATCH_PID)"
fi

echo ""
echo "----------------------------------------"
echo "进度统计"
echo "----------------------------------------"

# 统计已完成的训练
LOG_DIR="logs/mmfi_stft/12_3_sip/lorasub_drs/Adam"
TOTAL_LOGS=$(ls -1 $LOG_DIR/*.log 2>/dev/null | wc -l)
echo "已完成训练: $TOTAL_LOGS / 100"

# 计算完成百分比
PERCENT=$((TOTAL_LOGS))
echo "完成进度: ${PERCENT}%"

echo ""
echo "----------------------------------------"
echo "最近完成的5个训练"
echo "----------------------------------------"

ls -lt $LOG_DIR/*.log 2>/dev/null | head -6 | tail -5 | while read line; do
    LOG_FILE=$(echo $line | awk '{print $9}')
    SEED=$(basename $LOG_FILE .log)

    # 提取准确率
    if [ -f "$LOG_FILE" ]; then
        ACC=$(grep "Average Accuracy:" $LOG_FILE | tail -1 | awk '{print $3}')
        TIMESTAMP=$(echo $line | awk '{print $6, $7}')

        if [ ! -z "$ACC" ]; then
            printf "种子 %5s: %6.2f%% (完成于 %s)\n" "$SEED" "$ACC" "$TIMESTAMP"
        fi
    fi
done

echo ""
echo "----------------------------------------"
echo "当前训练进度"
echo "----------------------------------------"

# 显示最后20行batch日志
tail -20 batch_mmfi_stft.log 2>/dev/null | grep -E "Seed|Accuracy|Statistics"

echo ""
echo "----------------------------------------"
echo "GPU使用情况"
echo "----------------------------------------"

nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1

echo ""
echo "========================================"
echo "监控命令:"
echo "  实时日志: tail -f batch_mmfi_stft.log"
echo "  循环监控: watch -n 30 ./monitor_mmfi_stft.sh"
echo "========================================"
