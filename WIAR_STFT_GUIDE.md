# Wiar数据集STFT处理指南

## 概述

已为Wiar数据集添加了STFT时频变换处理，使用方式与MMFI-STFT完全相同。

## 快速使用

### 方法1：命令行参数
```bash
cd /data1/rsl/LoRA-Sub-DRS-master
/data1/rsl/anaconda3/envs/consense/bin/python main.py \
  --dataset wiar_stft \
  --data_path data/wiar/
```

### 方法2：配置文件
创建 `configs/wiar_stft.json`:
```json
{
    "dataset": "wiar_stft",
    "data_path": "data/",
    ...
}
```

然后运行:
```bash
python main.py --config configs/wiar_stft.json
```

## 数据集对比

| 项目 | 原始方法 (wiar) | STFT方法 (wiar_stft) |
|------|----------------|---------------------|
| 数据集名称 | wiar | wiar_stft |
| 数据目录 | data/wiar/train/, data/wiar/test/ | data/wiar_stft/train/, data/wiar_stft/test/ |
| 处理方式 | 直接转置 | STFT时频变换 |
| STFT参数 | - | nperseg=20, nfft=32 |
| 类别数 | 16 | 16 |

## STFT参数说明

Wiar数据集使用的STFT参数（在utils/data.py第373-379行）:

```python
stft_processor = CSISTFTProcessor(
    nperseg=20,         # 窗口长度（Wiar时间步较多）
    noverlap=10,        # 窗口重叠50%
    nfft=32,            # FFT点数
    output_size=(224, 224),
    use_log_scale=True  # 对数尺度增强
)
```

**与MMFI的区别**:
- Wiar数据时间维度更长，使用更大的窗口 (20 vs 8)
- 更高的FFT分辨率 (32 vs 16)

## 数据维度

```
假设Wiar原始数据: (time_steps, channels, subcarriers)
例如: (270, 3, 30)

STFT处理后:
1. 对每个子载波的时间序列进行STFT
2. 生成时频谱图
3. 组合为RGB图像 (224, 224, 3)
```

## 首次运行

首次使用wiar_stft会自动：
1. 加载原始.npy文件
2. 对每个样本进行STFT处理
3. 保存为PNG图像到 data/wiar_stft/
4. 处理时间: 约10-30分钟

后续运行会直接加载缓存的图像。

## 文件结构

```
LoRA-Sub-DRS-master/
├── data/
│   ├── wiar/              # 原始NPY文件
│   │   ├── wiar_01.npy
│   │   ├── wiar_02.npy
│   │   └── ... (wiar_16.npy)
│   ├── wiar/train/        # 原始方法处理的图像
│   ├── wiar/test/
│   ├── wiar_stft/train/   # STFT处理的训练集
│   └── wiar_stft/test/    # STFT处理的测试集
```

## 与MMFI对比

| 特性 | MMFI | Wiar |
|------|------|------|
| 类别数 | 27 | 16 |
| 数据形状示例 | (3, 114, 10) | (270, 3, 30) |
| STFT窗口 | 8 | 20 |
| FFT点数 | 16 | 32 |

## 测试脚本

创建测试脚本 `test_wiar_stft.py`:
```python
import numpy as np
from utils.csi_stft_processor import CSISTFTProcessor

# 加载Wiar数据
data = np.load('data/wiar/wiar_01.npy')
print(f"数据形状: {data.shape}")

# 创建处理器
processor = CSISTFTProcessor(
    nperseg=20,
    noverlap=10,
    nfft=32,
    output_size=(224, 224),
    use_log_scale=True
)

# 处理样本
sample = data[0]
img = processor.process_csi_sample(sample, 'test_wiar_stft.png')
print(f"输出图像: {img.shape}")
print("保存到: test_wiar_stft.png")
```

## 常见问题

**Q: Wiar数据格式是什么？**
A: 需要检查实际数据。通常是 (time_steps, channels, subcarriers) 或类似格式。

**Q: 如何调整STFT参数？**
A: 编辑 `utils/data.py` 第373-379行，然后删除 `data/wiar_stft/` 重新生成。

**Q: 可以同时使用wiar和wiar_stft吗？**
A: 可以，它们是独立的数据集。

## 支持的数据集

现在支持以下数据集的STFT版本：
- ✅ mmfi / mmfi_stft (27类)
- ✅ wiar / wiar_stft (16类)
- ✅ xrf (48类) - 可添加STFT版本

## 下一步

1. 测试Wiar数据加载
2. 验证STFT处理效果
3. 对比原始方法和STFT方法的性能

参考 MMFI 的测试过程和结果！
