# 100次随机种子运行 + Top5平均

## 师兄要求

> "用随机种子跑100次，调5次最好的取平均"

## 快速使用

### 1. 后台运行100次训练（推荐）
```bash
./run_batch_stft.sh background
```

### 2. 查看实时进度
```bash
tail -f batch_stft.log
```

### 3. 等待完成，查看Top5结果
训练完成后会自动显示：
```
MMFI_STFT:
  Top5 Mean: 0.6752 ± 0.0098
  Top5 Seeds: [8234, 4521, 7893, 2341, 9876]

WIAR_STFT:
  Top5 Mean: 0.9234 ± 0.0078
  Top5 Seeds: [1234, 5678, 9012, 3456, 7890]
```

## 核心实现

### 1. 生成100个随机种子
```python
seeds = generate_random_seeds(100)
# [123, 456, 789, ..., 9876] (100个)
```

### 2. 自动运行所有种子
每个种子独立训练，保存日志到：
- `logs/mmfi_stft/12_3_sip/lorasub_drs/Adam/{seed}.log`
- `logs/wiar_stft/8_2_sip/lorasub_drs/Adam/{seed}.log`

### 3. 自动计算Top5
```python
# 排序后取最大的5个
top5 = np.sort(accuracies)[-5:]
top5_mean = np.mean(top5)
top5_std = np.std(top5)
```

## 完整流程图

```
开始
  ↓
生成100个随机种子 [0-9999]
  ↓
对每个种子:
  ├─ 创建临时配置文件
  ├─ 运行完整训练
  ├─ 从日志解析准确率
  └─ 记录结果
  ↓
收集所有准确率
  ↓
排序 → 取Top5
  ↓
计算Top5平均值和标准差
  ↓
输出最终结果 + 保存JSON
  ↓
结束
```

## 文件说明

| 文件 | 用途 |
|------|------|
| `batch_train_stft.py` | 主脚本（100次训练+Top5统计） |
| `run_batch_stft.sh` | 一键启动脚本 |
| `test_batch_stft.py` | 测试脚本（2次训练验证） |
| `BATCH_TRAIN_STFT_GUIDE.md` | 详细使用指南 |
| `batch_results_stft_*.json` | 结果文件（自动生成） |

## 使用方式对比

### 方式1：单次运行（原方法）
```bash
python main.py --config configs/mmfi_short_stft.json
```
- 只运行1次
- 随机性影响大
- 无统计信息

### 方式2：100次运行 + Top5平均（新方法）
```bash
./run_batch_stft.sh background
```
- 运行100次
- **Top5平均准确率**（师兄要求）
- 完整统计数据
- 结果可重复

## 时间估算

### MMFI STFT
- 单次: 10-15分钟
- 100次: 17-25小时

### Wiar STFT
- 单次: 5-8分钟
- 100次: 8-13小时

### 建议
使用 `nohup` 后台运行，避免终端断开。

## 结果文件示例

`batch_results_stft_20251121_143022.json`:
```json
{
  "mmfi_stft": {
    "statistics": {
      "mean": 0.6231,
      "std": 0.0345,
      "count": 100,
      "top5_mean": 0.6752,
      "top5_std": 0.0098,
      "top5_values": [0.6701, 0.6732, 0.6754, 0.6789, 0.6842]
    }
  }
}
```

## 向师兄汇报时

提供以下信息：

1. **Top5平均准确率**（最重要）：
   ```
   MMFI-STFT: 67.52% ± 0.98%
   Wiar-STFT: 92.34% ± 0.78%
   ```

2. **100次统计**：
   ```
   MMFI-STFT: 62.31% ± 3.45% (100次)
   Wiar-STFT: 89.08% ± 2.56% (100次)
   ```

3. **最好的5个种子**：方便后续重现

4. **完整JSON结果**：包含所有详细数据

## 测试验证

运行测试脚本（2个种子）：
```bash
/data1/rsl/anaconda3/envs/consense/bin/python test_batch_stft.py
```

验证通过后再运行完整100次。

## 常见问题

**Q: 可以中途停止吗？**
A: 可以。已完成的结果会保存，可以修改脚本从未完成的种子继续。

**Q: 可以只运行特定数据集吗？**
A: 可以。编辑 `batch_train_stft.py` 第194行：
```python
datasets = ["mmfi_stft"]  # 只跑MMFI
datasets = ["wiar_stft"]  # 只跑Wiar
```

**Q: 如何查看中间结果？**
A: 实时查看：`tail -f batch_stft.log`
   或查看已完成的日志文件。

## 原理说明

这个方案完全遵循原有代码的设计：

1. **原有 trainer.py 第19行**：
   ```python
   for seed in seed_list:
       args['seed'] = seed
       _train(args)
   ```
   支持遍历多个种子。

2. **原有 batch_train.py**：
   - 第40行：`generate_random_seeds(n=50)`
   - 第148行：`top5 = np.sort(acc_array)[-5:]`
   已有50次运行和Top5统计。

3. **我们的改进**：
   - 种子数量：50 → 100
   - 数据集：原始数据 → STFT版本
   - 其余逻辑完全保持一致

## 总结

✅ 完全按照师兄要求实现
✅ 100个随机种子
✅ 自动计算Top5平均
✅ 完整结果保存
✅ 一键运行

使用命令：
```bash
./run_batch_stft.sh background
tail -f batch_stft.log
```

等待完成后查看Top5结果即可！
