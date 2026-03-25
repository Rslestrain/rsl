# STFT数据集批量训练指南

## 概述

`batch_train_stft.py` 是用于 MMFI-STFT 和 Wiar-STFT 数据集的批量训练脚本，支持：
- **100个随机种子** 的自动化训练
- **Top5平均准确率** 的自动计算
- GPU内存管理和错误处理
- 详细的结果统计和保存

## 师兄的要求

> "用随机种子跑100次，调5次最好的取平均"

本脚本完全按照这个要求实现：
1. 生成100个不同的随机种子（固定随机数生成器种子42，确保可重复）
2. 对每个种子运行完整训练
3. 自动找出准确率最高的5次
4. 计算Top5的平均值和标准差
5. 保存完整结果到JSON文件

## 快速开始

### 方法1：直接运行（推荐）

```bash
cd /data1/rsl/LoRA-Sub-DRS-master
/data1/rsl/anaconda3/envs/consense/bin/python batch_train_stft.py
```

### 方法2：使用运行脚本

```bash
./run_batch_stft.sh
```

### 方法3：后台运行

```bash
nohup /data1/rsl/anaconda3/envs/consense/bin/python batch_train_stft.py > batch_stft.log 2>&1 &
```

查看进度：
```bash
tail -f batch_stft.log
```

## 数据集选择

在 `batch_train_stft.py` 第194行，可以选择要运行的数据集：

```python
# 选项1：同时运行两个数据集（默认）
datasets = ["mmfi_stft", "wiar_stft"]

# 选项2：只运行MMFI STFT
datasets = ["mmfi_stft"]

# 选项3：只运行Wiar STFT
datasets = ["wiar_stft"]
```

## 输出说明

### 1. 实时输出

训练过程中会显示：
```
============================================================
Processing dataset: mmfi_stft
============================================================
Generated 100 random seeds: [123, 456, 789, ...]

[mmfi_stft] Seed 1/100: 123
Accuracy: 0.6231

[mmfi_stft] Seed 2/100: 456
Accuracy: 0.6354

...
```

### 2. 统计结果

每个数据集完成后显示：
```
mmfi_stft Statistics:
  Total Runs: 100
  Mean Accuracy: 0.6231 ± 0.0345
  Min Accuracy: 0.5523
  Max Accuracy: 0.6842
  Top5 Mean: 0.6752 ± 0.0098
  Top5 Values: ['0.6701', '0.6732', '0.6754', '0.6789', '0.6842']
```

### 3. 最终汇总

所有数据集完成后显示：
```
============================================================
Final Summary
============================================================

MMFI_STFT:
  Runs: 100
  Mean: 0.6231 ± 0.0345
  Top5 Mean: 0.6752 ± 0.0098
  Top5 Seeds: [8234, 4521, 7893, 2341, 9876]

WIAR_STFT:
  Runs: 100
  Mean: 0.8908 ± 0.0256
  Top5 Mean: 0.9234 ± 0.0078
  Top5 Seeds: [1234, 5678, 9012, 3456, 7890]
```

### 4. 结果文件

生成 `batch_results_stft_YYYYMMDD_HHMMSS.json`：
```json
{
  "mmfi_stft": {
    "results": [
      {"seed": 123, "accuracy": 0.6231, "status": "success"},
      {"seed": 456, "accuracy": 0.6354, "status": "success"},
      ...
    ],
    "statistics": {
      "mean": 0.6231,
      "std": 0.0345,
      "count": 100,
      "min": 0.5523,
      "max": 0.6842,
      "top5_mean": 0.6752,
      "top5_std": 0.0098,
      "top5_values": [0.6701, 0.6732, 0.6754, 0.6789, 0.6842]
    }
  },
  "wiar_stft": { ... }
}
```

## 时间估算

### 单次训练时间
- **MMFI STFT**: 约10-15分钟
- **Wiar STFT**: 约5-8分钟

### 100次训练总时间
- **MMFI STFT**: 约17-25小时
- **Wiar STFT**: 约8-13小时
- **两者一起**: 约25-38小时

建议使用 `nohup` 后台运行，避免终端断开导致中断。

## 错误处理

脚本自动处理以下情况：

### 1. GPU内存不足
```
检测到CUDA内存不足错误，跳过此种子并继续下一个
GPU内存不足，强制清理内存...
```
- 自动清理GPU内存
- 跳过失败的种子
- 继续运行下一个

### 2. 训练超时
- 单个训练超时限制：1小时
- 超时后自动跳过，继续下一个

### 3. 日志解析失败
- 记录为 `log_parse_failed` 状态
- 不影响其他种子运行

## 日志位置

训练日志保存在：
- **MMFI STFT**: `logs/mmfi_stft/12_3_sip/lorasub_drs/Adam/{seed}.log`
- **Wiar STFT**: `logs/wiar_stft/8_2_sip/lorasub_drs/Adam/{seed}.log`

临时配置文件：
- `temp_config/temp_{dataset}_{seed}.json` （训练完成后自动删除）

## 高级用法

### 修改种子数量

编辑 `batch_train_stft.py` 第210行：
```python
# 生成100个随机种子
seeds = generate_random_seeds(100)

# 或改为其他数量
seeds = generate_random_seeds(50)  # 50个种子
```

### 修改随机种子范围

编辑 `batch_train_stft.py` 第44行：
```python
def generate_random_seeds(n=100, seed_range=(0, 10000)):
    # 修改为其他范围
    # seed_range=(0, 100000)
```

### 只运行特定种子

可以手动指定种子列表：
```python
# 在 main() 函数中，替换第210行
seeds = [0, 1, 42, 100, 999]  # 只运行这些种子
```

## Top5计算逻辑

代码在 `calculate_statistics()` 函数（第145-158行）：
```python
# 排序后取最大的5个值
top5 = np.sort(acc_array)[-5:]
stats['top5_mean'] = np.mean(top5)
stats['top5_std'] = np.std(top5)
stats['top5_values'] = top5.tolist()
```

这完全符合师兄的要求："调5次最好的取平均"

## 对比原有方法

### 原有 batch_train.py
- 50个随机种子
- 支持 wiar, mmfi, xrf 原始数据集

### 新的 batch_train_stft.py
- **100个随机种子**（师兄要求）
- 支持 mmfi_stft, wiar_stft
- **自动计算Top5平均**（师兄要求）
- 完全兼容原有的统计方式

## 问题排查

### 问题1：CUDA out of memory

**解决方案**：
- 脚本会自动跳过并继续
- 可以减少batch_size（编辑配置文件）
- 或在 GPU 空闲时运行

### 问题2：训练很慢

**原因**：正常，100次训练需要较长时间

**建议**：
```bash
# 使用 nohup 后台运行
nohup /data1/rsl/anaconda3/envs/consense/bin/python batch_train_stft.py > batch_stft.log 2>&1 &

# 查看进度
tail -f batch_stft.log

# 或只运行部分数据集
# 编辑 batch_train_stft.py 第194行
datasets = ["mmfi_stft"]  # 只跑MMFI
```

### 问题3：如何查看中间结果

训练过程中，每完成一个种子就会显示准确率：
```bash
[mmfi_stft] Seed 50/100: 5234
Accuracy: 0.6234
```

可以随时查看已完成种子的日志：
```bash
ls logs/mmfi_stft/12_3_sip/lorasub_drs/Adam/*.log
```

## 与师兄讨论

当你向师兄汇报时，可以提供：

1. **Top5平均准确率**（最重要）：
   ```
   MMFI-STFT Top5 Mean: 67.52% ± 0.98%
   Wiar-STFT Top5 Mean: 92.34% ± 0.78%
   ```

2. **Top5对应的种子**：方便后续重现最好的结果

3. **完整的统计结果**：
   - 100次的平均值和标准差
   - 最小值和最大值
   - Top5的详细数值

4. **结果JSON文件**：`batch_results_stft_YYYYMMDD_HHMMSS.json`

## 总结

本脚本完全按照师兄的要求实现：
- ✅ 100个随机种子
- ✅ 自动找出最好的5次
- ✅ 计算Top5平均值
- ✅ 完整的结果保存
- ✅ 自动化错误处理

使用建议：
1. 后台运行（`nohup`）
2. 定期查看进度（`tail -f`）
3. 等待完成后查看最终的Top5结果
4. 向师兄汇报Top5平均准确率
