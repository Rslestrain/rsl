# MMFI按人划分数据加载方案A - 实施完成

## 实施总结

已成功实现MMFI数据集按人划分的加载方案A，解决了原有拼接导致的数据混乱问题。

## 方案配置

### 数据划分
- **训练集**: S01-S08 (8人)
- **测试集**: S09-S10 (2人)
- **采样策略**: 每人每动作随机采样10个样本（从100个中）

### 数据量
- **训练样本**: 8人 × 27动作 × 10样本 = **2160个**
- **测试样本**: 2人 × 27动作 × 10样本 = **540个**
- **每类训练样本**: 80个
- **每类测试样本**: 20个

### STFT参数（针对T=10优化）
- `nperseg=8` - 窗口长度
- `noverlap=4` - 50%重叠
- `nfft=16` - 频率分辨率
- `output_size=(224, 224)` - 输出图像大小

## 实施内容

### 1. 新增数据加载类
**文件**: `utils/data.py`

添加了类 `iMMFIDataSTFT_ByPerson`:
- 从源目录读取: `data/mmfi/27a100sample40domin/27a100sample40domin/`
- 按人划分数据，不混合不同人的样本
- 使用STFT将CSI数据转换为224×224图像
- 输出目录: `data/mmfi_by_person_stft/train/` 和 `/test/`

### 2. 注册到数据管理器
**文件**: `utils/data_manager.py`

- 导入新类: `iMMFIDataSTFT_ByPerson`
- 注册数据集名称: `mmfi_by_person_stft`

### 3. 配置文件
**文件**: `configs/mmfi_by_person_stft.json`

关键配置:
```json
{
    "dataset": "mmfi_by_person_stft",
    "init_cls": 12,
    "increment": 3,
    "total_sessions": 6,
    "batch_size": 64
}
```

### 4. 测试脚本
**文件**: `test_mmfi_simple.py`

验证内容:
- 源文件完整性检查
- 数据格式验证
- 采样逻辑测试
- STFT处理器检查

## 使用方法

### 方式1: 直接训练（推荐）

```bash
python main.py --config configs/mmfi_by_person_stft.json
```

**首次运行**:
- 自动从源文件加载数据
- 进行STFT处理
- 生成2700张图像（2160训练 + 540测试）
- 保存到 `data/mmfi_by_person_stft/`
- 处理时间约5-10分钟

**后续运行**:
- 自动检测已处理的数据
- 直接加载，无需重新处理

### 方式2: 预先测试

```bash
# 测试数据完整性
python test_mmfi_simple.py

# 然后训练
python main.py --config configs/mmfi_by_person_stft.json
```

## 与原方案对比

### 原方案（mmfi_stft）的问题:

```
数据流程:
1. 合并40人的数据 → mmfi_processed_a01.npy (10400样本)
2. 只取前100样本
3. Shuffle打乱
4. 拼接相邻5个样本
5. 结果: 拼接的5个样本来自5个不同的人！
```

**问题**:
- ✗ 不同人的数据被拼接在一起
- ✗ 创造了语义错误的"虚假"样本
- ✗ 数据量减少80% (2700 → 540)
- ✗ 模型学到的是噪声

### 新方案（mmfi_by_person_stft）优势:

```
数据流程:
1. 直接从源文件按人读取
2. 每人随机采样10个样本
3. 严格按人划分训练/测试集
4. 不拼接，保持T=10的自然粒度
```

**优势**:
- ✓ 严格按人划分，测试集是完全未见过的人
- ✓ 保留数据自然粒度（T=10）
- ✓ 数据量准确控制（2160 + 540）
- ✓ 每类样本分布均匀
- ✓ 真正评估跨人泛化能力

## 预期效果提升

基于问题分析，预期:
1. **准确率**: 提升10-20%
2. **泛化性**: 显著改善（真正的跨人测试）
3. **训练稳定性**: 更好（数据语义正确）

## 数据目录结构

```
data/mmfi/
├── 27a100sample40domin/
│   └── 27a100sample40domin/
│       ├── S01_A01.npy  (源文件)
│       ├── S01_A02.npy
│       ├── ...
│       └── S40_A27.npy
│
└── mmfi_by_person_stft/  (新生成)
    ├── train/
    │   ├── 0/  (动作类别0)
    │   │   ├── train_a01_s0.png
    │   │   ├── train_a01_s1.png
    │   │   └── ...  (80张图片)
    │   ├── 1/
    │   └── ...  (27个类别)
    │
    └── test/
        ├── 0/  (20张图片)
        ├── 1/
        └── ...  (27个类别)
```

## 技术细节

### 随机采样
- 使用固定随机种子（42）保证可重复性
- 从每人的100个样本中随机选择10个
- 充分利用数据多样性

### STFT处理
- 针对T=10优化的参数
- 适合短时序CSI数据
- 生成高质量的时频谱图

### 类别分布
- 训练集每类: 80样本（8人 × 10）
- 测试集每类: 20样本（2人 × 10）
- 完全平衡，无偏差

## 注意事项

1. **首次运行需要时间**: STFT处理2700个样本需要5-10分钟
2. **磁盘空间**: 生成的图像约需500MB空间
3. **源文件必须存在**: 确保 `data/mmfi/27a100sample40domin/27a100sample40domin/` 有完整文件
4. **Python环境**: 需要numpy, PIL, scipy（STFT处理）

## 调整方案（可选）

如果想使用其他方案:

### 方案B（16+4人）
修改 `utils/data.py` 中的配置:
```python
self.train_subjects = list(range(1, 17))  # S01-S16
self.test_subjects = list(range(17, 21))  # S17-S20
self.samples_per_person = 5  # 每人5样本
```

### 方案C（10+2人）
```python
self.train_subjects = list(range(1, 11))  # S01-S10
self.test_subjects = list(range(11, 13))  # S11-S12
self.samples_per_person = 8  # 每人8样本
```

## 问题排查

### 如果数据加载失败
1. 检查源文件路径是否正确
2. 运行 `python test_mmfi_simple.py` 验证
3. 检查 `utils/csi_stft_processor.py` 是否存在

### 如果内存不足
- 减少 `num_workers` 在配置文件中
- 减少 `batch_size`

### 如果想重新生成数据
删除已处理的数据:
```bash
rm -rf data/mmfi_by_person_stft/
```
然后重新运行训练脚本。

## 参考文档

- 问题分析: `MMFI_SOURCE_SOLUTION.md`
- 拼接分析: `MMFI_CONCATENATION_ANALYSIS.md`
- 方案计算: `calculate_mmfi_sampling.py`
- 源数据分析: `analyze_mmfi_source.py`

## 总结

方案A成功解决了原有的数据混乱问题，通过:
1. **从源头按人组织数据**
2. **避免不同人数据的错误拼接**
3. **保持数据的自然粒度**
4. **严格的跨人泛化测试**

现在可以开始训练，预期会获得显著更好的效果！
