"""
测试STFT处理器
"""
import numpy as np
import sys
sys.path.append('.')

from utils.csi_stft_processor import CSISTFTProcessor

# 加载真实的MMFI数据
print("加载MMFI数据...")
data_path = "data/mmfi/mmfi_processed_a01.npy"
data = np.load(data_path)
print(f"数据形状: {data.shape}")

# 取第一个样本测试
sample = data[0]
print(f"样本形状: {sample.shape}")
print(f"样本范围: [{sample.min()}, {sample.max()}]")

# 创建STFT处理器
print("\n创建STFT处理器...")
processor = CSISTFTProcessor(
    nperseg=8,
    noverlap=4,
    nfft=16,
    output_size=(224, 224),
    use_log_scale=True
)

# 处理样本
print("处理样本...")
img = processor.process_csi_sample(sample, save_path='test_mmfi_stft.png')
print(f"输出图像形状: {img.shape}")
print(f"输出图像范围: [{img.min()}, {img.max()}]")
print("谱图已保存到 test_mmfi_stft.png")

# 测试多个样本
print("\n测试处理多个样本...")
for i in range(3):
    sample = data[i]
    img = processor.process_csi_sample(sample, save_path=f'test_mmfi_stft_{i}.png')
    print(f"样本 {i}: 输出图像形状 {img.shape}")

print("\n测试完成！")
