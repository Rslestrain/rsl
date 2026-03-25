#!/bin/bash
# 监控MMFI 100次训练进度

LOG_FILE="batch_results/mmfi_by_person_100runs_20251122_151903.log"
PID=2742110

echo "=========================================="
echo "MMFI 100次训练监控面板"
echo "=========================================="
echo ""

# 检查进程状态
if ps -p $PID > /dev/null 2>&1; then
    ELAPSED=$(ps -p $PID -o etime= | tr -d ' ')
    echo "✓ 训练进程运行中"
    echo "  进程ID: $PID"
    echo "  运行时间: $ELAPSED"
else
    echo "✗ 训练进程已停止"
    exit 1
fi

echo ""
echo "日志文件: $LOG_FILE"
echo ""

# 统计已完成的训练数
if [ -f "$LOG_FILE" ]; then
    COMPLETED=$(grep -c "✓ 种子" "$LOG_FILE" 2>/dev/null || echo "0")
    FAILED=$(grep -c "✗ 种子" "$LOG_FILE" 2>/dev/null || echo "0")
    TOTAL=100

    echo "进度统计:"
    echo "  已完成: $COMPLETED/100"
    echo "  失败数: $FAILED"
    echo "  剩余: $((TOTAL - COMPLETED - FAILED))"

    if [ $COMPLETED -gt 0 ]; then
        # 计算预计完成时间
        ELAPSED_SEC=$(ps -p $PID -o etimes= | tr -d ' ')
        if [ ! -z "$ELAPSED_SEC" ] && [ $COMPLETED -gt 0 ]; then
            AVG_TIME=$((ELAPSED_SEC / COMPLETED))
            REMAINING=$((TOTAL - COMPLETED - FAILED))
            ETA_SEC=$((AVG_TIME * REMAINING))
            ETA_HOURS=$((ETA_SEC / 3600))
            ETA_MINS=$(((ETA_SEC % 3600) / 60))

            echo "  平均用时: ${AVG_TIME}秒/次"
            echo "  预计剩余: ${ETA_HOURS}小时${ETA_MINS}分钟"
        fi
    fi

    echo ""

    # 显示当前统计
    CURRENT_STATS=$(grep "当前统计:" "$LOG_FILE" 2>/dev/null | tail -1)
    if [ ! -z "$CURRENT_STATS" ]; then
        echo "最新统计:"
        echo "  $CURRENT_STATS"
    fi

    echo ""
    echo "------------------------------------------"
    echo "最近10行日志:"
    echo "------------------------------------------"
    tail -10 "$LOG_FILE" 2>/dev/null || echo "日志正在写入中..."
else
    echo "⚠️ 日志文件尚未创建"
fi

echo ""
echo "=========================================="
echo "监控命令:"
echo "=========================================="
echo "实时日志: tail -f $LOG_FILE"
echo "查看进程: ps -p $PID -o pid,etime,cmd"
echo "停止训练: kill $PID"
echo "再次查看: ./monitor_training.sh"
