import numpy as np
import matplotlib.pyplot as plt

# 加载一个MMFI文件
file_path = "data/mmfi/mmfi_processed_a01.npy"
data = np.load(file_path)

print(f"文件: {file_path}")
print(f"总样本数: {data.shape[0]}")
print(f"单个样本形状: {data.shape[1:]}")
print()

# 检查连续样本之间的相似度/差异
# 计算相邻样本之间的欧氏距离
num_samples_to_check = min(100, len(data))
consecutive_diffs = []

for i in range(num_samples_to_check - 1):
    sample1 = data[i]
    sample2 = data[i + 1]

    # 计算欧氏距离
    diff = np.linalg.norm(sample1 - sample2)
    consecutive_diffs.append(diff)

consecutive_diffs = np.array(consecutive_diffs)

print("=" * 60)
print("连续样本差异分析（前100个样本）:")
print("=" * 60)
print(f"连续样本间平均差异: {consecutive_diffs.mean():.4f}")
print(f"连续样本间差异标准差: {consecutive_diffs.std():.4f}")
print(f"连续样本间最小差异: {consecutive_diffs.min():.4f}")
print(f"连续样本间最大差异: {consecutive_diffs.max():.4f}")
print()

# 随机抽样比较（作为对照）
np.random.seed(42)
random_indices = np.random.choice(num_samples_to_check, size=50, replace=False)
random_diffs = []

for i in range(len(random_indices) - 1):
    idx1 = random_indices[i]
    idx2 = random_indices[i + 1]

    sample1 = data[idx1]
    sample2 = data[idx2]

    diff = np.linalg.norm(sample1 - sample2)
    random_diffs.append(diff)

random_diffs = np.array(random_diffs)

print("随机样本差异分析（对照组）:")
print("=" * 60)
print(f"随机样本间平均差异: {random_diffs.mean():.4f}")
print(f"随机样本间差异标准差: {random_diffs.std():.4f}")
print()

# 关键判断
print("=" * 60)
print("连续性判断:")
print("=" * 60)

ratio = consecutive_diffs.mean() / random_diffs.mean()
print(f"连续样本差异 / 随机样本差异 = {ratio:.4f}")
print()

if ratio < 0.5:
    print("✓ 结论: 样本很可能是连续的时间切片")
    print("  连续样本之间的差异显著小于随机样本，说明它们在时间上相邻")
    print("  -> 适合进行时域拼接")
elif ratio < 0.8:
    print("? 结论: 样本可能具有一定连续性，但不完全确定")
    print("  建议进一步检查数据集文档或元数据")
else:
    print("✗ 结论: 样本很可能是独立的、被shuffle过的")
    print("  连续样本之间的差异接近随机样本，说明它们不是时间上连续的")
    print("  -> 不适合进行时域拼接！")
print()

# 查看前10个样本的部分数据值
print("=" * 60)
print("前5个样本的数据片段查看 (channel 0, subcarrier 0, 所有时间步):")
print("=" * 60)
for i in range(5):
    print(f"样本 {i}: {data[i, 0, 0, :]}")
print()

# 检查样本是否完全相同（排除重复数据的可能）
unique_samples = 0
seen_hashes = set()
for i in range(min(1000, len(data))):
    sample_hash = hash(data[i].tobytes())
    if sample_hash not in seen_hashes:
        unique_samples += 1
        seen_hashes.add(sample_hash)

print(f"前1000个样本中的唯一样本数: {unique_samples}")
if unique_samples == min(1000, len(data)):
    print("-> 所有样本都是唯一的，没有重复")
else:
    print(f"-> 有 {min(1000, len(data)) - unique_samples} 个重复样本")
