# MMFI数据源头分析与解决方案

## 数据源结构

### 原始数据组织
```
/data/mmfi/27a100sample40domin/27a100sample40domin/
├── S01_A01.npy  (100样本, shape: 100×3×114×10)
├── S01_A02.npy  (100样本)
├── ...
├── S01_A27.npy  (100样本)
├── S02_A01.npy  (100样本)
├── ...
└── S40_A27.npy  (100样本)

总计: 40人 × 27动作 × 100样本 = 108,000个样本
```

**关键信息**:
- **SXX**: Subject (人), 编号01-40
- **AYY**: Action (动作), 编号01-27
- 每个文件 = 1个特定的人做1个特定动作的所有样本

### 现有处理后的数据
```
/data/mmfi/
├── mmfi_processed_a01.npy  (10,400样本)
├── mmfi_processed_a02.npy  (10,400样本)
├── ...
└── mmfi_processed_a27.npy  (10,400样本)
```

**处理逻辑**:
- 将40个人的同一动作合并
- 预期: 40人 × 100样本 = 4,000样本
- 实际: 10,400样本（2.6倍，可能有数据增强）

## 问题根源分析

### 问题1: 拼接了不同人的数据 ⭐⭐⭐⭐⭐ (致命问题)

**现有流程**:
```python
# 1. 加载mmfi_processed_a01.npy (10400个样本，40个人混合)
data = np.load("mmfi_processed_a01.npy")  # shape: (10400, 3, 114, 10)

# 2. 只取前100个
data = data[:100]

# 3. shuffle划分训练/测试集
np.random.shuffle(indices)

# 4. 时域拼接：每5个样本拼接
concatenated_sample = np.concatenate([
    data[0],   # 可能是S17的第28个样本
    data[1],   # 可能是S05的第91个样本
    data[2],   # 可能是S33的第12个样本
    data[3],   # 可能是S22的第67个样本
    data[4],   # 可能是S08的第45个样本
], axis=2)  # 拼接成一个"假"的长样本
```

**问题**:
- ✗ 5个样本来自**5个不同的人**做同一个动作
- ✗ 每个人的动作模式、速度、幅度都不同
- ✗ 拼接创造了**不存在的、语义错误的**样本
- ✗ 相当于把"张三挥手的前2秒 + 李四挥手的2-4秒 + 王五挥手的4-6秒"拼在一起
- ✗ 这样的数据对模型完全是噪声！

### 问题2: 数据量减少80%

```
原始: 10,400样本 → 取100样本 → 拼接5个 → 20样本
训练集: 27类 × 20样本 × 0.8 = 432样本
```

### 问题3: 即使同一个人的样本也不完全连续

测试S01_A01内部的连续性:
- 连续样本差异 / 随机样本差异 = 0.82
- 说明同一个人的100个样本也不是严格连续的时间切片
- 可能是多次采集的结果

## Gemini判断的准确性

**Gemini说法**: "需要确认数据是否是连续切片，如果是shuffle过的绝对不能拼接"

**准确度评估**: ✓✓✓✓ (非常准确)

**但需要补充**:
- Gemini警告的是"shuffle"问题
- **实际更严重**: 不仅shuffle了，还**混合了不同的人**
- 这比单纯shuffle更致命

## 从源头的解决方案

### 方案1: 按人组织数据 + 按人拼接 ⭐⭐⭐⭐⭐ (强烈推荐)

```python
# 数据处理流程
train_data = []
test_data = []

# 按人划分: 前32人训练，后8人测试
train_subjects = range(1, 33)  # S01-S32
test_subjects = range(33, 41)  # S33-S40

for action in range(1, 28):  # A01-A27
    # 训练集
    for subject in train_subjects:
        data = load(f"S{subject:02d}_A{action:02d}.npy")  # 100样本

        # 在同一个人内拼接
        for i in range(0, 95, 5):  # 不重叠拼接
            concat_sample = np.concatenate([
                data[i], data[i+1], data[i+2], data[i+3], data[i+4]
            ], axis=2)
            train_data.append((concat_sample, action-1))

    # 测试集（同样的逻辑）
    for subject in test_subjects:
        # ... 同上
```

**优点**:
- ✓ 拼接的样本来自**同一个人**
- ✓ 保持语义一致性
- ✓ 训练集32人，测试集8人（真正的跨人泛化测试）
- ✓ 总数据量: 27类 × 40人 × 20拼接样本 = 21,600个样本

### 方案2: 不拼接 + 按人划分 ⭐⭐⭐⭐⭐ (最保险)

```python
# 直接使用原始T=10的样本，不拼接
train_subjects = range(1, 33)
test_subjects = range(33, 41)

for action in range(1, 28):
    for subject in train_subjects:
        data = load(f"S{subject:02d}_A{action:02d}.npy")  # 100样本
        for sample in data:
            train_data.append((sample, action-1))
```

**优点**:
- ✓ 保留所有数据: 27类 × 40人 × 100样本 = 108,000个样本
- ✓ 保持数据自然粒度
- ✓ 按人划分确保泛化性

### 方案3: 滑动窗口拼接（同一个人内） ⭐⭐⭐⭐

```python
# 在同一个人内用滑动窗口
data = load(f"S{subject:02d}_A{action:02d}.npy")  # 100样本

for i in range(96):  # 0-95
    concat_sample = np.concatenate([
        data[i], data[i+1], data[i+2], data[i+3], data[i+4]
    ], axis=2)
    # 得到96个拼接样本（而不是20个）
```

**优点**:
- ✓ 数据量: 27类 × 40人 × 96样本 = 103,680个样本
- ✓ 同一个人内拼接
- ✓ 数据利用率高

## 推荐实施方案

### 第一优先: 方案2（不拼接）
**理由**:
1. 保留全部数据
2. 避免引入任何拼接的不确定性
3. T=10可能已经是最优粒度
4. 可以针对T=10优化STFT参数

### 第二优先: 方案1（按人拼接）
**理由**:
1. 如果确实需要更长的时间序列
2. 保证拼接的语义正确性
3. 数据量虽然减少但仍然充足(21,600个)

### 不推荐: 当前的拼接方式
**理由**:
1. 混合不同人的数据
2. 创造虚假样本
3. 数据量骤减
4. 效果差是必然的

## 实施步骤

### Step 1: 创建新的数据加载类

```python
class iMMFIDataSTFT_ByPerson(iData):
    """
    从源头按人组织MMFI数据
    """
    def download_data(self):
        source_dir = "data/mmfi/27a100sample40domin/27a100sample40domin/"

        # 按人划分
        train_subjects = range(1, 33)
        test_subjects = range(33, 41)

        # 选择方案2: 不拼接
        train_data, train_labels = self._load_data(
            source_dir, train_subjects, concat=False
        )
        test_data, test_labels = self._load_data(
            source_dir, test_subjects, concat=False
        )

        # 或选择方案1: 按人拼接
        # train_data, train_labels = self._load_data(
        #     source_dir, train_subjects, concat=True, concat_num=5
        # )
```

### Step 2: 调整STFT参数

如果用方案2（T=10）:
```python
stft_processor = CSISTFTProcessor(
    nperseg=8,
    noverlap=4,
    nfft=16,
    output_size=(224, 224),
    use_log_scale=True
)
```

如果用方案1（T=50）:
```python
stft_processor = CSISTFTProcessor(
    nperseg=16,
    noverlap=8,
    nfft=32,
    output_size=(224, 224),
    use_log_scale=True
)
```

### Step 3: 更新配置文件

确保训练/测试按人划分，而不是随机shuffle

## 预期效果提升

基于上述改进，预期:
1. **准确率提升**: 10-20%（避免虚假样本）
2. **泛化能力**: 显著提升（真正的跨人测试）
3. **数据利用**: 大幅改善（保留更多数据）

## 总结

**Gemini的判断非常正确！** 但实际问题比他说的更严重：

- Gemini: "数据可能被shuffle，不适合拼接"
- 实际: "数据不仅shuffle，还混合了40个不同的人"

**从源头解决**:
1. 回到原始的 SXX_AYY.npy 文件
2. 按人组织数据
3. 只在同一个人内拼接（如果需要）
4. 训练/测试按人划分

这样才能获得有意义的、语义正确的数据！
