# MMFI STFT处理命令速查表

## 环境设置
```bash
cd /data1/rsl/LoRA-Sub-DRS-master
PYTHON=/data1/rsl/anaconda3/envs/consense/bin/python
```

## 快速测试（推荐先运行）

```bash
# 1. 检查原始数据
$PYTHON check_mmfi_data.py

# 2. 对比原始方法 vs STFT方法
$PYTHON compare_methods.py
# 输出：compare_old_method.png 和 compare_stft_method.png

# 3. 测试STFT处理器
$PYTHON test_stft_processor.py
# 输出：test_mmfi_stft.png 等

# 4. 详细可视化处理流程
$PYTHON visualize_stft_details.py
# 输出：visualize_stft_details.png（12个子图详细展示）

# 5. 完整使用示例
$PYTHON example_use_stft.py
```

## 训练命令

### 方法1：使用配置文件
```bash
# 使用STFT数据集训练
python main.py --config configs/mmfi_stft_example.json
```

### 方法2：命令行参数
```bash
# STFT版本
python main.py \
  --dataset mmfi_stft \
  --data_path data/mmfi/ \
  --init_cls 9 \
  --increment 3 \
  --model_name vit_base_patch16_224 \
  --lora_rank 4 \
  --init_epochs 100 \
  --epochs 100 \
  --batch_size 128

# 原始版本（对比）
python main.py \
  --dataset mmfi \
  --data_path data/mmfi/ \
  # ... 其他参数相同
```

## 数据预处理（首次使用）

```bash
# 方式1：通过训练触发（推荐）
python main.py --dataset mmfi_stft --data_path data/mmfi/ --help
# 第一次运行会自动处理数据

# 方式2：手动预处理
$PYTHON -c "
from utils.data import iMMFIDataSTFT
dataset = iMMFIDataSTFT({'data_path': 'data/mmfi/'})
dataset.download_data()
"
```

## 查看生成的数据

```bash
# 查看STFT处理后的图像
ls -lh data/mmfi_stft/train/0/ | head -5
ls -lh data/mmfi_stft/test/0/ | head -5

# 统计样本数量
echo "训练集样本数："
find data/mmfi_stft/train -name "*.png" | wc -l

echo "测试集样本数："
find data/mmfi_stft/test -name "*.png" | wc -l

# 使用图片查看器
eog data/mmfi_stft/train/0/train_0.png  # Linux
open data/mmfi_stft/train/0/train_0.png  # Mac
```

## 清理和重新生成

```bash
# 删除STFT处理的数据（重新生成）
rm -rf data/mmfi_stft/

# 重新运行会自动重新处理
python main.py --dataset mmfi_stft --data_path data/mmfi/
```

## 调整STFT参数

```bash
# 编辑参数文件
vim utils/data.py +528

# 修改这些参数：
# - nperseg: 窗口长度（6-10）
# - noverlap: 重叠长度（窗口的50-75%）
# - nfft: FFT点数（16-32）
# - use_log_scale: True/False

# 保存后删除旧数据重新生成
rm -rf data/mmfi_stft/
python main.py --dataset mmfi_stft --data_path data/mmfi/
```

## 性能对比实验

```bash
# 1. 训练baseline（原始方法）
python main.py \
  --dataset mmfi \
  --config configs/your_config.json \
  --prefix baseline_

# 2. 训练STFT版本
python main.py \
  --dataset mmfi_stft \
  --config configs/your_config.json \
  --prefix stft_

# 3. 对比日志
tensorboard --logdir logs/

# 或查看日志文件
cat logs/*/baseline_*.log
cat logs/*/stft_*.log
```

## 常用检查命令

```bash
# 检查数据是否已生成
ls -ld data/mmfi_stft/train data/mmfi_stft/test

# 检查每个类别的样本数
for i in {0..26}; do
  echo "类别 $i: $(ls data/mmfi_stft/train/$i/*.png 2>/dev/null | wc -l) 训练样本"
done

# 检查图像大小
file data/mmfi_stft/train/0/train_0.png

# 检查磁盘占用
du -sh data/mmfi_stft/
```

## 调试命令

```bash
# Python交互式测试
$PYTHON -i << EOF
from utils.csi_stft_processor import CSISTFTProcessor
import numpy as np

# 加载数据
data = np.load('data/mmfi/mmfi_processed_a01.npy')
sample = data[0]
print(f"样本形状: {sample.shape}")

# 创建处理器
processor = CSISTFTProcessor()

# 处理样本
img = processor.process_csi_sample(sample, 'debug_output.png')
print(f"输出图像形状: {img.shape}")
EOF
```

## 一键运行所有测试

```bash
# 创建测试脚本
cat > run_all_tests.sh << 'EOF'
#!/bin/bash
PYTHON=/data1/rsl/anaconda3/envs/consense/bin/python

echo "=== 1. 检查原始数据 ==="
$PYTHON check_mmfi_data.py

echo -e "\n=== 2. 对比两种方法 ==="
$PYTHON compare_methods.py

echo -e "\n=== 3. 测试STFT处理器 ==="
$PYTHON test_stft_processor.py

echo -e "\n=== 4. 详细可视化 ==="
$PYTHON visualize_stft_details.py

echo -e "\n=== 5. 使用示例 ==="
$PYTHON example_use_stft.py

echo -e "\n=== 所有测试完成！ ==="
ls -lh *.png
EOF

# 运行
chmod +x run_all_tests.sh
./run_all_tests.sh
```

## 文档阅读顺序

```bash
# 1. 快速开始
cat QUICK_START_STFT.md

# 2. 详细原理
cat STFT_README.md

# 3. 完整总结
cat STFT_SUMMARY.md

# 4. 实现细节
cat README_STFT_IMPLEMENTATION.md

# 5. 命令速查（本文档）
cat STFT_COMMANDS_CHEATSHEET.md
```

## 快速参考

| 任务 | 命令 |
|------|------|
| 查看对比 | `python compare_methods.py` |
| 测试处理器 | `python test_stft_processor.py` |
| 训练STFT | `python main.py --dataset mmfi_stft` |
| 训练原始 | `python main.py --dataset mmfi` |
| 清理数据 | `rm -rf data/mmfi_stft/` |
| 查看样本 | `ls data/mmfi_stft/train/0/` |

## 核心概念速记

```
原始方法: (3,114,10) → 转置 → (114,10,3) → resize → (224,224,3)
STFT方法: (3,114,10) → STFT → (3,114,9,4) → 重塑 → (114,36,3) → resize → (224,224,3)

信息量提升: 1140 → 4104 像素（2.7倍）
处理时间: 首次10-30分钟，后续即时加载
存储占用: +50MB左右

关键参数:
- nperseg=8    (窗口长度)
- noverlap=4   (重叠长度)
- nfft=16      (FFT点数)
- use_log_scale=True (对数增强)
```

## 故障排查

```bash
# 问题：找不到scipy模块
pip install scipy

# 问题：matplotlib中文显示乱码
# 忽略警告，不影响功能

# 问题：内存不足
# 减少每类样本数（修改utils/data.py第549-551行）

# 问题：处理太慢
# 使用更少的样本进行测试

# 问题：图像质量不好
# 调整STFT参数（nperseg, nfft）
```
