import numpy as np

# 加载MMFI数据
file_path = "data/mmfi/mmfi_processed_a01.npy"
data = np.load(file_path)

print("=" * 80)
print("MMFI数据连续性深度分析")
print("=" * 80)
print(f"文件: {file_path}")
print(f"数据形状: {data.shape}")
print()

# 方法1: 检查相邻样本差异 vs 固定间隔差异
print("方法1: 比较不同间隔的样本差异")
print("-" * 80)

intervals = [1, 2, 5, 10, 20, 50, 100]
for interval in intervals:
    diffs = []
    for i in range(min(500, len(data) - interval)):
        diff = np.linalg.norm(data[i] - data[i + interval])
        diffs.append(diff)

    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)
    print(f"间隔={interval:3d}: 平均差异={mean_diff:.4f}, 标准差={std_diff:.4f}")

print()
print("分析: 如果样本是连续的时间帧:")
print("  - 间隔越大，差异应该越大")
print("  - 应该看到明显的递增趋势")
print("如果样本是独立的:")
print("  - 不同间隔的差异应该接近")
print()

# 方法2: 自相关分析
print("方法2: 自相关分析")
print("-" * 80)

# 取一个样本的时间序列进行分析
sample_series = data[:200, 0, 57, :].flatten()  # 取200个样本的某个subcarrier数据，展平

from scipy.stats import pearsonr
lags = [1, 2, 5, 10, 20, 50]
for lag in lags:
    if lag < len(sample_series):
        correlation, _ = pearsonr(sample_series[:-lag], sample_series[lag:])
        print(f"滞后={lag:2d}: 相关系数={correlation:.4f}")

print()
print("分析: 如果样本是连续的:")
print("  - 相关系数应该随滞后增加而递减")
print("  - lag=1的相关系数应该很高(>0.7)")
print("如果样本是独立的:")
print("  - 相关系数应该接近0")
print()

# 方法3: 检查样本内vs样本间的方差
print("方法3: 样本内方差 vs 样本间方差")
print("-" * 80)

num_samples = 100
within_sample_vars = []
between_sample_vars = []

for i in range(num_samples):
    # 样本内方差：同一个样本内不同时间步的方差
    within_var = np.var(data[i])
    within_sample_vars.append(within_var)

    # 样本间方差：当前样本与下一个样本的差异
    if i < num_samples - 1:
        between_var = np.var(data[i] - data[i+1])
        between_sample_vars.append(between_var)

mean_within = np.mean(within_sample_vars)
mean_between = np.mean(between_sample_vars)

print(f"样本内平均方差: {mean_within:.6f}")
print(f"样本间平均方差: {mean_between:.6f}")
print(f"比值 (样本间/样本内): {mean_between/mean_within:.4f}")
print()
print("分析: 如果样本是连续的:")
print("  - 样本间方差应该较小 (平滑过渡)")
print("  - 比值应该 < 2")
print("如果样本是独立的:")
print("  - 样本间方差应该很大")
print("  - 比值应该 >> 10")
print()

# 方法4: 检查前10个和后10个样本的统计特性
print("方法4: 检查数据集不同部分的统计特性")
print("-" * 80)

segments = [
    ("前100个样本", data[:100]),
    ("中间100个样本", data[5000:5100]),
    ("后100个样本", data[-100:])
]

for name, segment in segments:
    mean_val = segment.mean()
    std_val = segment.std()
    min_val = segment.min()
    max_val = segment.max()

    print(f"{name}:")
    print(f"  均值={mean_val:.4f}, 标准差={std_val:.4f}, 范围=[{min_val:.4f}, {max_val:.4f}]")

print()
print("分析: 如果统计特性差异很大，可能意味着:")
print("  - 数据来自不同的采集批次")
print("  - 或者数据已经被某种方式分组")
print()

# 最终判断
print("=" * 80)
print("综合判断")
print("=" * 80)

# 读取实际的拼接代码来看发生了什么
print("\n检查当前拼接代码的行为:")
print("-" * 80)

# 模拟拼接过程
concat_num = 5
test_samples = data[:10]  # 取前10个样本

print(f"原始: 10个样本，每个形状 {test_samples[0].shape}")
print()

# 拼接前5个
samples_to_concat = test_samples[:concat_num]
concatenated = np.concatenate(samples_to_concat, axis=2)
print(f"拼接: 前{concat_num}个样本拼接后形状 {concatenated.shape}")
print()

# 检查拼接后的连续性
print("拼接后的时间维度数据 (取第一个channel, 第一个subcarrier):")
print(concatenated[0, 0, :])  # 应该是50个时间步
print()

# 对比原始的5个样本
print("原始5个样本在同一位置的数据:")
for i in range(5):
    print(f"样本{i}: {test_samples[i, 0, 0, :]}")
print()

print("如果拼接是正确的，拼接后的50个值应该等于原始5个样本各自的10个值按顺序连接")
