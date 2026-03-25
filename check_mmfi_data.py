import numpy as np

# 加载一个样本文件查看数据形状
file_path = "data/mmfi/mmfi_processed_a01.npy"
data = np.load(file_path)

print(f"文件: {file_path}")
print(f"数据形状: {data.shape}")
print(f"数据类型: {data.dtype}")
print(f"数据范围: [{data.min()}, {data.max()}]")
print(f"数据均值: {data.mean()}")
print(f"数据标准差: {data.std()}")

if len(data.shape) >= 2:
    print(f"\n单个样本形状: {data[0].shape}")
    print(f"单个样本范围: [{data[0].min()}, {data[0].max()}]")
