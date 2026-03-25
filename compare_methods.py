"""
对比原始方法和STFT方法生成的图像
"""
import numpy as np
from PIL import Image
import sys
sys.path.append('.')

from utils.csi_stft_processor import CSISTFTProcessor

# 加载MMFI数据
print("加载MMFI数据...")
data_path = "data/mmfi/mmfi_processed_a01.npy"
data = np.load(data_path)
sample = data[0]
print(f"样本形状: {sample.shape}")

# 方法1: 原始粗糙方法（直接转置+resize）
print("\n方法1: 原始粗糙方法")
channels, height, width = sample.shape  # (3, 114, 10)

# 归一化
min_val = sample.min()
max_val = sample.max()
if max_val > min_val:
    normalized_data = (sample - min_val) / (max_val - min_val)
else:
    normalized_data = sample

# 转置为 (height, width, channels)
img_data_old = np.transpose(normalized_data, (1, 2, 0))  # (114, 10, 3)
img_data_old = (img_data_old * 255).astype(np.uint8)

# 保存并resize
img_old = Image.fromarray(img_data_old)
print(f"原始方法 - resize前图像大小: {img_old.size}")
img_old_resized = img_old.resize((224, 224), Image.LANCZOS)
img_old_resized.save('compare_old_method.png')
print(f"原始方法 - resize后图像大小: {img_old_resized.size}")
print("保存到: compare_old_method.png")

# 方法2: STFT时频变换方法
print("\n方法2: STFT时频变换方法")
processor = CSISTFTProcessor(
    nperseg=8,
    noverlap=4,
    nfft=16,
    output_size=(224, 224),
    use_log_scale=True
)
img_stft = processor.process_csi_sample(sample, save_path='compare_stft_method.png')
print(f"STFT方法 - 输出图像形状: {img_stft.shape}")
print("保存到: compare_stft_method.png")

print("\n对比总结:")
print(f"原始方法: 直接将(3,114,10)转置为(114,10,3)，丢失了时频特性")
print(f"STFT方法: 对每个子载波的时间序列进行STFT，生成(114, freq_bins*time_bins, 3)的谱图")
print("\nSTFT方法的优势:")
print("1. 保留了信号的时频特性")
print("2. 通过对数尺度增强了细节")
print("3. 更符合将时序信号转换为图像的常规做法")
print("4. 生成的谱图更适合ViT进行特征提取")
