# MMFI CSI数据的STFT时频变换处理

## 概述

本项目实现了将MMFI CSI时序信号通过STFT（短时傅里叶变换）转换为时频谱图的功能，用于ViT（Vision Transformer）图像分类。

## 数据流程

### 原始粗糙方法
```
CSI数据 (3, 114, 10)
  → 转置 (114, 10, 3)
  → 归一化到0-255
  → Resize到224x224
  → ViT输入
```

**问题**：
- 直接将维度重排，丢失了时序信息
- 没有提取时频特性
- 仅通过简单的resize改变尺寸，可能引入伪影

### 改进的STFT方法
```
CSI数据 (3, 114, 10)
  → 对每个子载波的时间序列进行STFT
  → 生成时频谱图 (3, 114, freq_bins, time_bins)
  → 重塑为2D图像 (114, freq_bins×time_bins, 3)
  → 归一化并使用对数尺度增强
  → Resize到224x224
  → ViT输入
```

**优势**：
1. 保留了信号的时频特性
2. 通过STFT提取频域信息
3. 对数尺度增强细节
4. 更符合信号处理的标准做法

## 使用方法

### 1. 基本使用

```python
from utils.data import iMMFIDataSTFT

# 创建数据集实例（使用STFT处理）
args = {'data_path': 'data/mmfi/'}
dataset = iMMFIDataSTFT(args)
dataset.download_data()
```

### 2. 自定义STFT参数

```python
from utils.csi_stft_processor import CSISTFTProcessor

# 创建自定义处理器
processor = CSISTFTProcessor(
    nperseg=8,          # STFT窗口长度
    noverlap=4,         # 窗口重叠长度
    nfft=16,            # FFT点数
    output_size=(224, 224),  # 输出图像大小
    use_log_scale=True,      # 使用对数尺度
    colormap='viridis'       # 颜色映射
)

# 处理单个样本
import numpy as np
csi_data = np.load('data/mmfi/mmfi_processed_a01.npy')[0]  # (3, 114, 10)
img = processor.process_csi_sample(csi_data, save_path='output.png')
```

### 3. 在训练中使用

修改配置文件，将数据集类型改为 `mmfi_stft`:

```python
# 在 utils/__init__.py 中添加
def get_dataloader(args):
    if args['dataset'] == 'mmfi_stft':
        from utils.data import iMMFIDataSTFT
        return iMMFIDataSTFT(args)
    elif args['dataset'] == 'mmfi':
        from utils.data import iMMFIData
        return iMMFIData(args)
    # ...
```

## 参数说明

### STFT参数

- `nperseg`: STFT窗口长度
  - 默认值: 8
  - 说明: 由于MMFI数据的时间步只有10，使用较小的窗口长度

- `noverlap`: 窗口重叠长度
  - 默认值: 4
  - 说明: 窗口重叠50%，增加时间分辨率

- `nfft`: FFT点数
  - 默认值: 16
  - 说明: 频率分辨率，可以根据需要调整

- `use_log_scale`: 是否使用对数尺度
  - 默认值: True
  - 说明: 使用log1p变换增强细节

## 数据维度变化

```
输入: (3, 114, 10)
  ├─ 3: 天线/通道数
  ├─ 114: 子载波数
  └─ 10: 时间步数

STFT处理:
  ├─ 对每个通道的每个子载波的时间序列 (10,) 进行STFT
  ├─ 得到频率-时间谱图 (freq_bins, time_bins)
  └─ 组合所有子载波: (3, 114, freq_bins, time_bins)

重塑为2D图像:
  ├─ 展平频率和时间维度: (3, 114, freq_bins×time_bins)
  └─ 转置为RGB格式: (114, freq_bins×time_bins, 3)

输出: (224, 224, 3)
```

## 可视化对比

运行以下脚本查看两种方法的对比：

```bash
python compare_methods.py
```

生成的图像：
- `compare_old_method.png`: 原始粗糙方法
- `compare_stft_method.png`: STFT时频变换方法

## 性能考虑

### 处理时间
- STFT方法比原始方法慢约3-5倍
- 但只在数据预处理阶段执行一次
- 预处理后的图像可以直接用于训练

### 存储空间
- 生成的PNG图像大小相似（约10-50KB/张）
- MMFI数据集（27类×100样本）约需 ~100MB 存储空间

## 其他时频变换方法

除了STFT，还可以考虑：

1. **小波变换（Wavelet Transform）**
   - 更好的时频分辨率
   - 适合非平稳信号

2. **连续小波变换（CWT）**
   - 生成更丰富的时频表示
   - 计算成本较高

3. **幅度-相位分解**
   - 分别处理CSI的幅度和相位
   - 适用于复数CSI数据

这些方法已在 `utils/csi_stft_processor.py` 中预留接口，可以根据需要扩展。

## 测试

```bash
# 测试STFT处理器
python test_stft_processor.py

# 对比两种方法
python compare_methods.py

# 检查MMFI数据
python check_mmfi_data.py
```

## 引用

如果使用STFT方法处理CSI数据，建议引用以下文献：

- STFT理论: Allen, J. (1977). Short term spectral analysis, synthesis, and modification by discrete Fourier transform.
- CSI感知综述: Ma, Y., et al. (2019). WiFi sensing with channel state information: A survey.

## 文件结构

```
LoRA-Sub-DRS-master/
├── utils/
│   ├── data.py                    # 数据加载类（包含iMMFIDataSTFT）
│   └── csi_stft_processor.py      # STFT处理器
├── data/
│   ├── mmfi/                      # 原始NPY文件
│   ├── mmfi_stft/                 # STFT处理后的图像
│   │   ├── train/
│   │   └── test/
│   └── mmfi/                      # 原始方法的图像（如果存在）
│       ├── train/
│       └── test/
├── test_stft_processor.py         # STFT处理器测试
├── compare_methods.py             # 方法对比
├── check_mmfi_data.py             # 数据检查
└── STFT_README.md                 # 本文档
```

## 常见问题

**Q: STFT方法会提升模型性能吗？**
A: 理论上会，因为STFT提取了更有意义的时频特征。但实际效果取决于具体任务和模型架构。

**Q: 可以调整STFT参数吗？**
A: 可以。在创建 `iMMFIDataSTFT` 实例时，可以传入自定义的STFT参数。

**Q: 处理所有MMFI数据需要多长时间？**
A: 约10-30分钟（取决于硬件）。处理完成后会缓存图像，后续直接加载。

**Q: 可以用于其他CSI数据集吗？**
A: 可以。只需修改 `csi_stft_processor.py` 中的参数以适应不同的数据维度。
