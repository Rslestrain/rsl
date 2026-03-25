"""
示例：如何使用STFT处理的MMFI数据集
"""
import sys
sys.path.append('.')

# 方法1: 使用新的iMMFIDataSTFT类（推荐）
print("="*60)
print("方法1: 使用iMMFIDataSTFT类加载STFT处理的数据")
print("="*60)

from utils.data import iMMFIDataSTFT

# 创建数据集
args = {'data_path': 'data/mmfi/'}
dataset = iMMFIDataSTFT(args)

print("\n正在加载/生成MMFI-STFT数据集...")
print("注意：首次运行会进行STFT处理，需要10-30分钟")
print("处理完成后会缓存图像，下次直接加载\n")

dataset.download_data()

print(f"\n数据集信息:")
print(f"  训练集大小: {len(dataset.train_data)}")
print(f"  测试集大小: {len(dataset.test_data)}")
print(f"  类别数量: {len(dataset.class_order)}")
print(f"  使用路径加载: {dataset.use_path}")

# 查看一些样本路径
print(f"\n训练集样本示例:")
for i in range(min(3, len(dataset.train_data))):
    print(f"  [{i}] {dataset.train_data[i]} -> 类别 {dataset.train_targets[i]}")

print("\n" + "="*60)
print("方法2: 直接使用CSISTFTProcessor处理单个样本")
print("="*60)

from utils.csi_stft_processor import CSISTFTProcessor
import numpy as np

# 加载原始数据
data_path = "data/mmfi/mmfi_processed_a01.npy"
print(f"\n加载原始数据: {data_path}")
data = np.load(data_path)
sample = data[0]
print(f"样本形状: {sample.shape}")

# 创建处理器
processor = CSISTFTProcessor(
    nperseg=8,
    noverlap=4,
    nfft=16,
    output_size=(224, 224),
    use_log_scale=True
)

# 处理样本
print("\n处理样本...")
img = processor.process_csi_sample(sample, save_path='example_output.png')
print(f"输出图像形状: {img.shape}")
print(f"输出图像保存到: example_output.png")

print("\n" + "="*60)
print("完成！")
print("="*60)
print("\n使用建议:")
print("1. 如果要训练模型，使用iMMFIDataSTFT类")
print("2. 如果要自定义处理，使用CSISTFTProcessor")
print("3. 参考 STFT_README.md 了解更多详情")
