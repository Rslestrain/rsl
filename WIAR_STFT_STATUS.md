# Wiar STFT 100次批量训练状态

## ✅ 当前运行状态

- **状态**: 🟢 运行中
- **进程ID**: 2777717
- **数据集**: Wiar STFT
- **种子数**: 100
- **开始时间**: 2025-11-21 16:26
- **日志文件**: `batch_results/wiar_stft_100runs_20251121_162650.log`

## 📊 实时进度

当前第 1/100 个种子（1824）正在训练：
- Task 2 完成：Average Accuracy 90.84%
- Task 3 进行中：Learning on 12-14

## ⏱️ 预计时间

- 单次训练：5-8分钟
- 100次总计：**8-13小时**
- 预计完成：明天凌晨 0:00 - 5:00

## 🔍 查看进度命令

```bash
# 查看最新日志
tail -f batch_results/wiar_stft_100runs_20251121_162650.log

# 查看当前种子的训练日志
tail -f logs/wiar_stft/8_2_sip/lorasub_drs/Adam/1824.log

# 查看已完成种子数
ls -d logs/wiar_stft/8_2_sip/lorasub_drs/Adam/*/ | wc -l

# 查看进程状态
ps aux | grep 2777717 | grep -v grep
```

## 📁 结果文件

训练完成后生成：
- `batch_results_stft_YYYYMMDD_HHMMSS.json` - 完整结果
- 包含100次准确率和Top5统计

## 🎯 预期输出

```
WIAR_STFT:
  Runs: 100
  Mean: XX.XX% ± X.XX%
  Top5 Mean: XX.XX% ± X.XX%  ← 汇报给师兄的结果
  Top5 Seeds: [seed1, seed2, seed3, seed4, seed5]
```

## 🛑 停止训练（如需）

```bash
kill 2777717
```

---

**更新时间**: 2025-11-21 16:28
**状态**: 运行正常，第1个种子训练中
