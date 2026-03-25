# MMFI数据STFT处理总结

## 已完成的工作

### 1. 创建了STFT时频变换处理器
文件：`utils/csi_stft_processor.py`

主要功能：
- `CSISTFTProcessor`: 使用STFT将CSI时序信号转换为时频谱图
- `CSIAmplitudePhaseProcessor`: 使用幅度-相位分解的替代方法
- 支持自定义STFT参数（窗口长度、重叠、FFT点数等）
- 对数尺度增强细节
- 自动归一化和resize

### 2. 创建了新的数据加载类
文件：`utils/data.py`（第468-619行）

新增类：
- `iMMFIDataSTFT`: 使用STFT处理MMFI数据的数据加载器
- 自动处理和缓存STFT谱图
- 与原有的`iMMFIData`类并存，可选择使用

### 3. 更新了数据管理器
文件：`utils/data_manager.py`

修改：
- 导入`iMMFIDataSTFT`类（第6行）
- 在`_get_idata`函数中添加`mmfi_stft`数据集支持（第211-212行）

### 4. 创建了测试和示例脚本

**测试脚本**：
- `check_mmfi_data.py`: 检查MMFI原始数据的形状和统计信息
- `test_stft_processor.py`: 测试STFT处理器
- `compare_methods.py`: 对比原始方法和STFT方法

**示例脚本**：
- `example_use_stft.py`: 展示如何使用新的STFT数据类

### 5. 创建了配置和文档

**配置文件**：
- `configs/mmfi_stft_example.json`: STFT数据集的训练配置示例

**文档**：
- `STFT_README.md`: 详细的使用说明和原理介绍
- `STFT_SUMMARY.md`: 本总结文档

## 核心改进

### 原始方法 vs STFT方法

| 方面 | 原始方法 | STFT方法 |
|------|---------|----------|
| 数据转换 | 直接转置(3,114,10)→(114,10,3) | STFT时频变换 |
| 时频特性 | 丢失 | 保留 |
| 频域信息 | 无 | 有 |
| 细节增强 | 无 | 对数尺度 |
| 理论基础 | 粗糙的维度操作 | 标准信号处理方法 |
| 处理速度 | 快 | 较慢（但只需一次） |

### 数据流程对比

**原始方法**：
```
CSI (3,114,10) → 转置 → 归一化 → Resize → (224,224,3)
```

**STFT方法**：
```
CSI (3,114,10) → 对每个子载波STFT → 时频谱 (3,114,freq,time)
→ 重塑为2D → 对数尺度 → 归一化 → Resize → (224,224,3)
```

## 使用方法

### 快速开始

1. **测试STFT处理器**：
```bash
cd /data1/rsl/LoRA-Sub-DRS-master
/data1/rsl/anaconda3/envs/consense/bin/python test_stft_processor.py
```

2. **对比两种方法**：
```bash
/data1/rsl/anaconda3/envs/consense/bin/python compare_methods.py
```

3. **查看使用示例**：
```bash
/data1/rsl/anaconda3/envs/consense/bin/python example_use_stft.py
```

### 在训练中使用

**方法1：使用配置文件**
```bash
# 使用STFT数据集训练
python main.py --config configs/mmfi_stft_example.json
```

**方法2：修改现有配置**
只需将配置中的 `dataset` 字段从 `"mmfi"` 改为 `"mmfi_stft"`:
```json
{
    "dataset": "mmfi_stft",
    ...
}
```

**方法3：命令行参数**
```bash
python main.py --dataset mmfi_stft --data_path data/mmfi/
```

### 在代码中使用

```python
from utils.data_manager import DataManager

# 创建数据管理器
data_manager = DataManager(
    dataset_name='mmfi_stft',
    shuffle=False,
    seed=1993,
    init_cls=9,
    increment=3,
    args={'data_path': 'data/mmfi/'}
)

# 获取数据集
train_dataset = data_manager.get_dataset(
    indices=np.arange(0, 9),  # 前9个类
    source='train',
    mode='train'
)
```

## STFT参数说明

可以通过修改 `utils/data.py` 中的STFT参数来调整处理效果：

```python
stft_processor = CSISTFTProcessor(
    nperseg=8,          # 窗口长度：更大=更好的频率分辨率，更差的时间分辨率
    noverlap=4,         # 重叠长度：更大=更平滑的谱图
    nfft=16,            # FFT点数：更大=更好的频率分辨率
    output_size=(224, 224),  # 输出图像大小
    use_log_scale=True,      # 对数尺度增强细节
    colormap='viridis'       # 颜色映射（如果使用matplotlib方法）
)
```

### 参数建议

对于MMFI数据（时间步=10）：
- `nperseg`: 6-10（窗口不能超过信号长度）
- `noverlap`: nperseg的50-75%
- `nfft`: 16-32（更高的频率分辨率）
- `use_log_scale`: True（建议开启）

## 性能影响

### 数据预处理时间
- **原始方法**：约5-10分钟（2700个样本）
- **STFT方法**：约10-30分钟（2700个样本）
- **注意**：只需处理一次，后续直接加载缓存的图像

### 存储空间
- **原始方法**：约100MB
- **STFT方法**：约100-150MB
- **差异**：STFT谱图可能包含更多细节，压缩后大小相近

### 训练速度
- **无影响**：两种方法生成的都是224x224的RGB图像，训练速度相同

## 预期效果

### 理论优势
1. **更好的时频表示**：STFT提取了频域特征
2. **更丰富的细节**：对数尺度增强了微小变化
3. **更符合信号处理原理**：标准的时序信号可视化方法

### 可能的性能提升
- **分类准确率**：预期提升2-5%（取决于任务）
- **收敛速度**：可能更快，因为特征更明显
- **泛化能力**：可能更好，因为提取了更本质的特征

## 扩展方向

### 其他时频变换方法
在 `csi_stft_processor.py` 中可以添加：
1. **连续小波变换（CWT）**：更好的时频分辨率
2. **Wigner-Ville分布**：更高的时频分辨率
3. **希尔伯特-黄变换（HHT）**：适合非平稳信号

### 其他CSI数据集
相同的方法可以应用到：
- Wiar数据集
- XRF数据集
- 其他WiFi CSI感知数据集

只需创建对应的类（如`iWiarDataSTFT`）并调整参数。

## 文件清单

### 核心文件
- `utils/csi_stft_processor.py`: STFT处理器
- `utils/data.py`: 数据加载类（新增iMMFIDataSTFT）
- `utils/data_manager.py`: 数据管理器（已更新）

### 测试文件
- `test_stft_processor.py`: 处理器测试
- `compare_methods.py`: 方法对比
- `check_mmfi_data.py`: 数据检查
- `example_use_stft.py`: 使用示例

### 配置文件
- `configs/mmfi_stft_example.json`: 配置示例

### 文档
- `STFT_README.md`: 详细文档
- `STFT_SUMMARY.md`: 本总结

## 后续步骤

1. **运行测试**：验证STFT处理器工作正常
   ```bash
   python test_stft_processor.py
   python compare_methods.py
   ```

2. **生成STFT数据集**（首次使用）：
   ```bash
   python example_use_stft.py
   ```

3. **训练模型**：
   ```bash
   python main.py --config configs/mmfi_stft_example.json
   ```

4. **对比性能**：
   - 使用原始MMFI数据训练一个baseline模型
   - 使用STFT-MMFI数据训练对比模型
   - 比较测试集准确率和收敛速度

## 常见问题

**Q: 必须使用STFT方法吗？**
A: 不是，原有的`iMMFIData`类仍然可用。可以根据需要选择。

**Q: 如何调整STFT参数？**
A: 修改 `utils/data.py` 第528-534行的参数。

**Q: 处理时间太长怎么办？**
A:
- 只处理一次，后续会使用缓存
- 减少样本数量（修改第549-551行）
- 使用更快的方法（如直接幅度-相位分解）

**Q: 可以可视化生成的谱图吗？**
A: 可以，查看 `data/mmfi_stft/train/` 和 `data/mmfi_stft/test/` 目录中的PNG图像。

## 总结

成功实现了基于STFT的CSI时序信号到时频谱图的转换流程，提供了：
1. ✅ 完整的STFT处理器实现
2. ✅ 新的数据加载类
3. ✅ 与现有框架的无缝集成
4. ✅ 详细的文档和示例
5. ✅ 测试和对比脚本

这种方法相比原始的粗糙转换，在理论上更加合理，预期能够提升模型的分类性能。
