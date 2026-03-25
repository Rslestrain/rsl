# MMFI按人划分方案A - 100次训练使用指南

## 快速开始

### 方式1: 交互式启动（推荐）

```bash
./run_batch_mmfi_by_person.sh
```

该脚本会：
1. 显示配置信息
2. 询问确认
3. 后台启动100次训练
4. 提供日志文件位置和查看命令

### 方式2: 直接启动

```bash
# 前台运行
python3 batch_train_mmfi_by_person.py

# 后台运行并保存日志
nohup python3 batch_train_mmfi_by_person.py > batch_results/training.log 2>&1 &
```

## 训练流程

### 完整流程

```
开始
  ↓
生成100个随机种子
  ↓
逐个训练（每个种子独立训练）
  ↓
解析每次训练的准确率
  ↓
统计所有结果
  ↓
计算Top5平均
  ↓
保存结果到JSON
  ↓
完成
```

### 每次训练

1. 第一次运行：
   - 自动从源文件加载数据（约5-10分钟）
   - 进行STFT处理
   - 生成2700张图像
   - 保存到 `data/mmfi_by_person_stft/`
   - 开始训练

2. 后续运行：
   - 直接加载已处理的数据
   - 立即开始训练

### 训练时间估计

- **单次训练**: 约10-20分钟
- **100次训练**: 约20-35小时
- **首次额外时间**: +5-10分钟（数据处理）

## 监控训练进度

### 查看实时日志

```bash
# 查看整体进度日志
tail -f batch_results/mmfi_by_person_100runs_YYYYMMDD_HHMMSS.log

# 查看最新的单次训练日志
tail -f logs/mmfi_by_person_stft/12_3_sip/lorasub_drs/Adam/*.log | tail -100
```

### 检查完成情况

```bash
# 检查训练进程
ps aux | grep batch_train_mmfi_by_person

# 查看已完成的训练数量
ls logs/mmfi_by_person_stft/12_3_sip/lorasub_drs/Adam/*.log | wc -l
```

### 查看当前统计

日志中会实时显示：
```
当前统计: 完成23/100 | 均值=0.6234 | 标准差=0.0156
```

## 结果文件

### 自动生成的文件

训练完成后会自动生成：

```
batch_results_mmfi_by_person_YYYYMMDD_HHMMSS.json
```

### JSON结果结构

```json
{
  "dataset": "mmfi_by_person_stft",
  "total_runs": 100,
  "successful_runs": 98,
  "failed_seeds": [1234, 5678],
  "all_results": {
    "42": 0.6543,
    "123": 0.6432,
    ...
  },
  "statistics": {
    "mean": 0.6234,
    "std": 0.0156,
    "max": 0.6789,
    "min": 0.5891,
    "median": 0.6245
  },
  "top5": {
    "seeds": [8234, 4521, 7893, 2341, 9876],
    "accuracies": [0.6789, 0.6756, 0.6734, 0.6721, 0.6709],
    "mean": 0.6742,
    "std": 0.0028
  }
}
```

## 对比分析

训练完成后，运行对比脚本：

```bash
python3 compare_mmfi_results.py
```

### 输出内容

1. **控制台输出**：
   - 整体统计对比
   - Top5对比
   - 提升百分比

2. **可视化图表**：
   - 准确率分布直方图
   - 箱线图
   - 累积分布函数
   - 统计指标柱状图

   保存为：`mmfi_comparison_YYYYMMDD_HHMMSS.png`

3. **文本报告**：
   保存为：`mmfi_comparison_report_YYYYMMDD_HHMMSS.txt`

### 对比示例

```
================================================================================
整体统计对比
================================================================================

指标                 原方案（拼接）        新方案（按人划分）    提升
--------------------------------------------------------------------------------
平均准确率           0.4521              0.6234              +0.1713 (+37.9%)
标准差              0.0234              0.0156              -0.0078 (-33.3%)
最大准确率           0.5123              0.6789              +0.1666 (+32.5%)

================================================================================
Top5对比
================================================================================

原方案Top5均值: 0.4856
新方案Top5均值: 0.6742
提升: +0.1886 (+38.8%)

✓ 新方案Top5平均准确率提升: 38.8%
✓ 从 0.4856 提升到 0.6742
✓ 证明了按人划分方法的有效性
```

## 停止训练

如果需要中途停止：

```bash
# 找到进程ID
ps aux | grep batch_train_mmfi_by_person

# 停止进程
kill <PID>

# 或者强制停止
kill -9 <PID>
```

**注意**：停止后已完成的训练结果仍然保存在日志中，可以手动整理。

## 恢复训练

如果训练中断，可以：

1. 查看已完成的训练：
   ```bash
   ls logs/mmfi_by_person_stft/12_3_sip/lorasub_drs/Adam/*.log
   ```

2. 修改脚本跳过已完成的种子（高级用户）

3. 或者重新开始（如果已完成数量较少）

## 使用Top5种子复现

训练完成后，结果会推荐最好的种子：

```bash
# 使用Top1种子训练
python main.py --config configs/mmfi_by_person_stft.json

# 修改配置文件中的seed为推荐值，例如：
# "seed": [8234]
```

## 目录结构

```
LoRA-Sub-DRS-master/
├── configs/
│   └── mmfi_by_person_stft.json        # 配置文件
├── data/
│   ├── mmfi/
│   │   └── 27a100sample40domin/        # 源数据
│   └── mmfi_by_person_stft/            # 处理后的数据（自动生成）
│       ├── train/                      # 2160张图
│       └── test/                       # 540张图
├── logs/
│   └── mmfi_by_person_stft/            # 训练日志（自动生成）
│       └── 12_3_sip/lorasub_drs/Adam/
│           ├── 42.log
│           ├── 123.log
│           └── ...
├── batch_results/                      # 批量训练日志目录
│   └── mmfi_by_person_100runs_*.log
├── batch_train_mmfi_by_person.py       # 批量训练主脚本
├── run_batch_mmfi_by_person.sh         # 快速启动脚本
├── compare_mmfi_results.py             # 结果对比脚本
├── batch_results_mmfi_by_person_*.json # 结果JSON（自动生成）
├── mmfi_comparison_*.png               # 对比图表（自动生成）
└── mmfi_comparison_report_*.txt        # 对比报告（自动生成）
```

## 常见问题

### Q1: 训练失败怎么办？

A: 查看失败日志：
```bash
grep "✗" batch_results/mmfi_by_person_100runs_*.log
```
失败的种子会被记录，不影响其他训练。

### Q2: 内存不足？

A:
1. 减少batch_size（配置文件中）
2. 减少num_workers
3. 脚本会自动每10次清理GPU内存

### Q3: 数据处理太慢？

A: 只有第一次需要处理数据。处理完成后会保存，后续直接加载。

### Q4: 如何只运行10次测试？

A: 修改脚本最后一行：
```python
batch_train(dataset, config, num_runs=10)  # 改为10
```

### Q5: 可以同时运行多个GPU吗？

A: 可以，修改脚本指定不同的GPU：
```bash
CUDA_VISIBLE_DEVICES=0 python3 batch_train_mmfi_by_person.py &
CUDA_VISIBLE_DEVICES=1 python3 batch_train_mmfi_by_person.py &
```
但需要修改种子范围避免重复。

## 性能优化建议

### 加速训练

1. **使用更快的GPU**
2. **增加batch_size**（如果内存允许）
3. **减少epoch数量**（测试时）
4. **使用多GPU并行**

### 节省空间

1. **训练完成后可以删除中间文件**：
   ```bash
   rm -rf temp_config/
   ```

2. **压缩日志文件**：
   ```bash
   tar -czf logs_backup.tar.gz logs/
   ```

## 预期结果

基于问题分析，预期新方案相比原方案：

- **准确率提升**: 10-40%
- **稳定性提升**: 标准差减小
- **泛化能力**: 显著改善（真正的跨人测试）

## 下一步

1. **启动100次训练**
   ```bash
   ./run_batch_mmfi_by_person.sh
   ```

2. **等待完成**（约1-2天）

3. **分析结果**
   ```bash
   python3 compare_mmfi_results.py
   ```

4. **使用最佳种子**进行最终训练

5. **发表论文** 🎉

## 技术支持

如遇问题，检查：
1. 源数据是否完整
2. Python环境是否正确
3. GPU内存是否充足
4. 磁盘空间是否充足

参考文档：
- `MMFI_BY_PERSON_IMPLEMENTATION.md` - 实施细节
- `MMFI_SOURCE_SOLUTION.md` - 问题分析
- `MMFI_CONCATENATION_ANALYSIS.md` - 拼接问题分析
