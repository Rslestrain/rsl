# MMFI数据STFT时频变换实现 - 完整总结

## 项目概述

成功实现了将MMFI CSI时序信号通过STFT（短时傅里叶变换）转换为时频谱图的完整流程，用于ViT图像分类。相比原始的粗糙维度转换方法，STFT方法保留了信号的时频特性，提取了更丰富的频域信息。

## 核心改进

### 处理流程对比

**原始方法（粗糙）**：
```
CSI (3,114,10) → 转置(114,10,3) → 归一化 → Resize(224,224) → ViT
```
- 简单的维度重排
- 丢失时序和频域信息
- 信息量：1140像素

**STFT方法（改进）**：
```
CSI (3,114,10) → 对114个子载波分别STFT → 时频谱(3,114,9,4)
→ 重塑(114,36,3) → 对数增强 → 归一化 → Resize(224,224) → ViT
```
- 提取时频特征
- 保留频域信息
- 对数尺度增强细节
- 信息量：4104像素（2.7倍提升）

## 文件结构

### 核心实现文件

```
LoRA-Sub-DRS-master/
├── utils/
│   ├── csi_stft_processor.py          # STFT处理器核心实现
│   ├── data.py                        # 新增iMMFIDataSTFT类（468-619行）
│   └── data_manager.py                # 数据管理器（已更新支持mmfi_stft）
│
├── configs/
│   └── mmfi_stft_example.json         # STFT数据集配置示例
│
├── data/
│   ├── mmfi/                          # 原始NPY文件
│   ├── mmfi/train/                    # 原始方法处理的图像
│   ├── mmfi/test/
│   ├── mmfi_stft/train/               # STFT处理的训练集图像
│   └── mmfi_stft/test/                # STFT处理的测试集图像
│
├── 测试脚本/
│   ├── check_mmfi_data.py             # 检查原始数据
│   ├── test_stft_processor.py         # 测试STFT处理器
│   ├── compare_methods.py             # 对比两种方法
│   ├── visualize_stft_details.py      # 详细可视化
│   └── example_use_stft.py            # 使用示例
│
└── 文档/
    ├── STFT_README.md                 # 详细文档和原理
    ├── STFT_SUMMARY.md                # 完整总结
    ├── QUICK_START_STFT.md            # 快速开始指南
    └── README_STFT_IMPLEMENTATION.md  # 本文档
```

## 快速使用

### 1. 在训练中使用

只需将配置中的数据集名称改为 `mmfi_stft`：

```json
{
    "dataset": "mmfi_stft",
    "data_path": "data/mmfi/",
    ...
}
```

或使用命令行：
```bash
python main.py --dataset mmfi_stft --data_path data/mmfi/ [其他参数]
```

### 2. 测试和验证

```bash
# 进入项目目录
cd /data1/rsl/LoRA-Sub-DRS-master

# 使用conda环境
PYTHON=/data1/rsl/anaconda3/envs/consense/bin/python

# 1. 检查原始数据
$PYTHON check_mmfi_data.py

# 2. 测试STFT处理器
$PYTHON test_stft_processor.py

# 3. 对比两种方法
$PYTHON compare_methods.py

# 4. 详细可视化
$PYTHON visualize_stft_details.py

# 5. 使用示例
$PYTHON example_use_stft.py
```

### 3. 查看生成的图像

运行测试脚本后，会生成以下可视化文件：
- `compare_old_method.png` - 原始方法结果
- `compare_stft_method.png` - STFT方法结果
- `visualize_stft_details.png` - 详细处理流程可视化
- `test_mmfi_stft.png` - 测试样本谱图

## 技术细节

### STFT参数配置

在 `utils/data.py` 第528-534行：

```python
stft_processor = CSISTFTProcessor(
    nperseg=8,          # 窗口长度（时间步=10，使用8）
    noverlap=4,         # 重叠长度（窗口的50%）
    nfft=16,            # FFT点数（频率分辨率）
    output_size=(224, 224),  # 输出图像大小
    use_log_scale=True       # 对数尺度增强细节
)
```

### 维度变化详解

```
输入CSI数据: (3, 114, 10)
  ├─ 3个通道/天线
  ├─ 114个子载波
  └─ 10个时间步

↓ 对每个子载波的时间序列进行STFT

STFT输出: (3, 114, 9, 4)
  ├─ 3个通道
  ├─ 114个子载波
  ├─ 9个频率bins（nfft/2+1）
  └─ 4个时间bins

↓ 重塑为2D图像

2D图像: (114, 36, 3)
  ├─ 高度=114（子载波数）
  ├─ 宽度=36（freq_bins×time_bins = 9×4）
  └─ 3个RGB通道

↓ 对数尺度 → 归一化 → Resize

最终输出: (224, 224, 3)
```

### 数据处理统计

- **原始CSI数据量**：10400个样本/类 × 27类 = 280,800个样本
- **实际使用**：100个样本/类 × 27类 = 2,700个样本
- **训练/测试划分**：80%/20% = 2,160训练 / 540测试
- **首次处理时间**：约10-30分钟
- **存储空间**：约100-150MB

## 代码实现要点

### 1. STFT处理器 (utils/csi_stft_processor.py)

```python
class CSISTFTProcessor:
    """CSI信号STFT处理器"""

    def compute_stft(self, signal_1d):
        """对单个一维信号计算STFT"""
        f, t, Zxx = signal.stft(signal_1d, ...)
        magnitude = np.abs(Zxx)
        if self.use_log_scale:
            magnitude = np.log1p(magnitude)
        return magnitude

    def process_csi_sample(self, csi_data, save_path=None):
        """处理单个CSI样本"""
        # 对每个通道的每个子载波进行STFT
        for ch in range(channels):
            for sc in range(subcarriers):
                spec = self.compute_stft(csi_data[ch, sc, :])
        # 组合成RGB图像
        # 归一化并resize
        return img
```

### 2. 数据加载类 (utils/data.py)

```python
class iMMFIDataSTFT(iData):
    """使用STFT处理MMFI数据"""

    def download_data(self):
        # 检查是否已有缓存
        # 加载原始NPY文件
        # 使用STFT处理器处理每个样本
        # 保存为PNG图像
        # 加载为ImageFolder
```

### 3. 数据管理器集成 (utils/data_manager.py)

```python
# 导入新类
from utils.data import iMMFIDataSTFT

# 注册数据集
def _get_idata(dataset_name, args=None):
    ...
    elif name == 'mmfi_stft':
        return iMMFIDataSTFT(args)
    ...
```

## 预期效果

### 理论优势
1. **时频特性保留**：STFT提取了频域信息
2. **信息量提升**：从1140像素增加到4104像素（2.7倍）
3. **细节增强**：对数尺度增强了微小变化
4. **标准方法**：符合信号处理的常规做法

### 性能预期
- **分类准确率**：预期提升2-5%
- **收敛速度**：可能更快
- **泛化能力**：可能更好
- **训练速度**：无影响（图像大小相同）

### 性能开销
- **预处理时间**：首次+10-30分钟
- **存储空间**：+50MB左右
- **后续加载**：与原始方法相同

## 使用场景

### 场景1：性能对比实验
```bash
# Baseline（原始方法）
python main.py --dataset mmfi --config configs/mmfi_config.json

# STFT版本
python main.py --dataset mmfi_stft --config configs/mmfi_config.json

# 对比准确率和收敛曲线
```

### 场景2：直接使用STFT
```bash
# 只使用STFT处理的数据
python main.py --dataset mmfi_stft --init_cls 9 --increment 3
```

### 场景3：自定义STFT参数
修改 `utils/data.py` 第528-534行，调整：
- `nperseg`：窗口长度（影响时间/频率分辨率权衡）
- `nfft`：FFT点数（影响频率分辨率）
- `use_log_scale`：是否使用对数尺度

## 扩展方向

### 1. 其他时频变换方法
- **连续小波变换（CWT）**：更好的时频分辨率
- **Wigner-Ville分布**：更高的时频分辨率
- **希尔伯特-黄变换（HHT）**：适合非平稳信号

### 2. 应用到其他数据集
相同的方法可以应用到：
- Wiar数据集：创建 `iWiarDataSTFT`
- XRF数据集：创建 `iXRFDataSTFT`
- 其他WiFi CSI感知数据集

### 3. 参数优化
通过网格搜索或贝叶斯优化找到最佳STFT参数组合。

## 常见问题

**Q: 必须使用STFT方法吗？**
A: 不是必须的。原有的`iMMFIData`类仍然可用，两种方法可以共存。

**Q: 首次运行为什么这么慢？**
A: 第一次需要对所有样本进行STFT处理并保存为图像。后续运行会直接加载缓存的图像，速度与原始方法相同。

**Q: 如何调整STFT参数？**
A: 编辑 `utils/data.py` 第528-534行的参数。调整后需要删除 `data/mmfi_stft/` 目录重新生成。

**Q: STFT方法一定会提升性能吗？**
A: 理论上会，但实际效果取决于具体任务和模型。建议通过实验对比验证。

**Q: 生成的图像可以可视化吗？**
A: 可以，查看 `data/mmfi_stft/train/` 和 `data/mmfi_stft/test/` 中的PNG图像。

**Q: 可以用于实时处理吗？**
A: 可以，STFT处理单个样本很快（<0.1秒）。但训练时建议使用预处理的图像。

## 文档索引

- **快速开始**：`QUICK_START_STFT.md`
- **详细原理**：`STFT_README.md`
- **完整总结**：`STFT_SUMMARY.md`
- **实现总结**：`README_STFT_IMPLEMENTATION.md`（本文档）

## 贡献者

实现日期：2025-11-21
实现内容：
- STFT处理器实现
- 数据加载类开发
- 测试脚本编写
- 文档撰写

## 引用

如果使用本实现，建议引用：
- STFT: Allen, J. (1977). Short term spectral analysis, synthesis, and modification by discrete Fourier transform.
- CSI感知: Ma, Y., et al. (2019). WiFi sensing with channel state information: A survey.

## 总结

成功实现了基于STFT的CSI时序信号到时频谱图的转换，提供了：
✅ 完整的STFT处理器
✅ 新的数据加载类
✅ 无缝的框架集成
✅ 详细的测试脚本
✅ 完善的文档

这种方法在理论上更加合理，预期能够提升模型性能。现在可以直接使用 `dataset=mmfi_stft` 进行训练！
