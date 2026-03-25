# STFT处理MMFI数据快速开始指南

## 一分钟快速开始

### 1. 查看原始vs STFT方法的对比
```bash
cd /data1/rsl/LoRA-Sub-DRS-master
/data1/rsl/anaconda3/envs/consense/bin/python compare_methods.py
```

生成的图像：
- `compare_old_method.png` - 原始粗糙方法（直接转置+resize）
- `compare_stft_method.png` - STFT时频谱图方法

### 2. 在训练中使用STFT数据

**只需要修改一个配置**：将数据集名称从 `mmfi` 改为 `mmfi_stft`

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

### 3. 首次运行会发生什么

第一次使用 `mmfi_stft` 时：
1. 自动加载原始NPY文件
2. 对每个样本进行STFT时频变换
3. 保存为PNG图像到 `data/mmfi_stft/train/` 和 `data/mmfi_stft/test/`
4. 需要约10-30分钟（取决于硬件）

后续运行：
- 直接加载缓存的PNG图像
- 速度与原始方法相同

## 核心区别

### 原始方法（粗糙）
```
CSI数据 (3, 114, 10)
  ↓ 直接转置
(114, 10, 3)
  ↓ 归一化
  ↓ Resize到224×224
输出图像
```
**问题**：丢失时频信息，只是简单的维度重排

### STFT方法（推荐）
```
CSI数据 (3, 114, 10)
  ↓ 对114个子载波的时间序列(10)分别做STFT
时频谱图 (3, 114, freq_bins, time_bins)
  ↓ 重塑为2D图像
  ↓ 对数尺度增强
  ↓ 归一化
  ↓ Resize到224×224
输出图像
```
**优势**：保留时频特性，提取频域信息

## 配置参数（可选）

如果想调整STFT参数，编辑 `utils/data.py` 第528-534行：

```python
stft_processor = CSISTFTProcessor(
    nperseg=8,          # 窗口长度（6-10适合MMFI）
    noverlap=4,         # 重叠长度（窗口的50%）
    nfft=16,            # FFT点数（16-32）
    output_size=(224, 224),
    use_log_scale=True  # 对数尺度增强细节
)
```

## 文件位置

### 核心代码
- `utils/csi_stft_processor.py` - STFT处理器
- `utils/data.py` 第468-619行 - iMMFIDataSTFT类
- `utils/data_manager.py` 第211-212行 - 数据集注册

### 数据目录
- `data/mmfi/` - 原始NPY文件
- `data/mmfi_stft/train/` - STFT处理后的训练集图像
- `data/mmfi_stft/test/` - STFT处理后的测试集图像

### 文档
- `STFT_README.md` - 详细文档
- `STFT_SUMMARY.md` - 完整总结
- `QUICK_START_STFT.md` - 本快速指南

## 测试脚本

```bash
# 1. 测试STFT处理器
python test_stft_processor.py

# 2. 对比两种方法
python compare_methods.py

# 3. 查看数据信息
python check_mmfi_data.py

# 4. 使用示例
python example_use_stft.py
```

## 常见场景

### 场景1：我想对比性能
```bash
# 训练baseline（原始方法）
python main.py --dataset mmfi --config configs/your_config.json

# 训练STFT版本
python main.py --dataset mmfi_stft --config configs/your_config.json

# 对比准确率
```

### 场景2：我只想用STFT
```bash
# 直接使用STFT数据集
python main.py --dataset mmfi_stft --data_path data/mmfi/ [其他参数]
```

### 场景3：我想自定义处理
```python
from utils.csi_stft_processor import CSISTFTProcessor
import numpy as np

# 加载数据
data = np.load('data/mmfi/mmfi_processed_a01.npy')
sample = data[0]  # (3, 114, 10)

# 创建处理器（自定义参数）
processor = CSISTFTProcessor(
    nperseg=10,
    noverlap=5,
    nfft=32,
    use_log_scale=True
)

# 处理
img = processor.process_csi_sample(sample, 'my_output.png')
```

## 预期效果

✅ **理论优势**
- 保留时序信号的频域特性
- 提取更丰富的时频特征
- 更符合信号处理标准做法

📊 **可能的性能提升**
- 分类准确率：+2-5%（预期）
- 收敛速度：可能更快
- 泛化能力：可能更好

⏱️ **性能开销**
- 预处理：首次+10-30分钟
- 训练速度：无影响
- 存储：+50MB左右

## 下一步

1. ✅ 运行 `compare_methods.py` 查看效果
2. ✅ 使用 `mmfi_stft` 数据集训练模型
3. ✅ 对比原始方法和STFT方法的准确率
4. ✅ 根据需要调整STFT参数

## 获取帮助

- 详细原理：查看 `STFT_README.md`
- 完整总结：查看 `STFT_SUMMARY.md`
- 代码细节：查看 `utils/csi_stft_processor.py`

---

**核心要点**：
只需将配置中的 `dataset` 从 `"mmfi"` 改为 `"mmfi_stft"` 即可使用STFT处理的数据！
