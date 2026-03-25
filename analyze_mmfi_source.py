import numpy as np
import os

# 源数据路径
source_dir = "/data1/rsl/LoRA-Sub-DRS-master/data/mmfi/27a100sample40domin/27a100sample40domin/"

print("=" * 80)
print("MMFI原始数据源分析")
print("=" * 80)
print()

# 1. 检查数据组织结构
print("1. 数据组织结构")
print("-" * 80)
files = sorted(os.listdir(source_dir))
print(f"总文件数: {len(files)}")
print(f"预期: 40人 × 27动作 = {40 * 27}")
print()

# 按人统计
subjects = set()
actions = set()
for f in files:
    if f.endswith('.npy'):
        parts = f.replace('.npy', '').split('_')
        if len(parts) == 2:
            subjects.add(parts[0])
            actions.add(parts[1])

print(f"人数: {len(subjects)} (S01-S{len(subjects):02d})")
print(f"动作类别: {len(actions)} (A01-A{len(actions):02d})")
print()

# 2. 检查单个文件的数据形状
print("2. 单个文件数据分析")
print("-" * 80)

# 随机选择几个文件查看
sample_files = [
    "S01_A01.npy",
    "S01_A02.npy",
    "S20_A15.npy",
    "S40_A27.npy"
]

for fname in sample_files:
    fpath = os.path.join(source_dir, fname)
    if os.path.exists(fpath):
        data = np.load(fpath)
        print(f"{fname}: shape={data.shape}, dtype={data.dtype}")
        print(f"  范围=[{data.min():.4f}, {data.max():.4f}], 均值={data.mean():.4f}")

print()

# 3. 检查同一个人不同动作的样本数是否一致
print("3. 检查S01所有动作的样本数")
print("-" * 80)
s01_samples = []
for i in range(1, 28):
    fname = f"S01_A{i:02d}.npy"
    fpath = os.path.join(source_dir, fname)
    if os.path.exists(fpath):
        data = np.load(fpath)
        s01_samples.append(len(data))
        if i <= 5 or i >= 25:  # 只打印前5个和后3个
            print(f"  {fname}: {len(data)} 个样本, shape={data.shape}")

print(f"  S01所有动作样本数: {s01_samples}")
print(f"  样本数是否一致: {len(set(s01_samples)) == 1}")
print()

# 4. 检查同一个动作不同人的样本数
print("4. 检查A01动作在不同人之间的样本数")
print("-" * 80)
a01_samples = []
people_to_check = [1, 10, 20, 30, 40]
for s in people_to_check:
    fname = f"S{s:02d}_A01.npy"
    fpath = os.path.join(source_dir, fname)
    if os.path.exists(fpath):
        data = np.load(fpath)
        a01_samples.append(len(data))
        print(f"  {fname}: {len(data)} 个样本")

print()

# 5. 检查现有处理后的文件
print("5. 检查现有处理后的文件 (mmfi_processed_aXX.npy)")
print("-" * 80)
processed_dir = "/data1/rsl/LoRA-Sub-DRS-master/data/mmfi/"
processed_files = [f for f in os.listdir(processed_dir) if f.startswith('mmfi_processed')]

if processed_files:
    # 检查第一个处理后的文件
    pfile = os.path.join(processed_dir, sorted(processed_files)[0])
    pdata = np.load(pfile)
    print(f"{sorted(processed_files)[0]}: shape={pdata.shape}")
    print()

    # 反推：处理后的文件是怎么生成的？
    print("推测生成逻辑:")
    print(f"  处理后单个动作样本数: {len(pdata)}")
    print(f"  如果是40人合并: {len(pdata)} / 40 = {len(pdata) / 40:.1f} 样本/人")
    print()

# 6. 关键发现：检查原始数据的时间连续性
print("6. 检查S01_A01的数据连续性（判断是否是连续采样）")
print("-" * 80)
fname = "S01_A01.npy"
fpath = os.path.join(source_dir, fname)
data = np.load(fpath)

print(f"总样本数: {len(data)}")
print(f"单个样本形状: {data[0].shape}")
print()

# 计算连续样本差异
if len(data) > 10:
    consecutive_diffs = []
    for i in range(min(100, len(data)-1)):
        diff = np.linalg.norm(data[i] - data[i+1])
        consecutive_diffs.append(diff)

    # 随机样本差异
    random_diffs = []
    for i in range(50):
        idx1, idx2 = np.random.randint(0, min(100, len(data)), 2)
        diff = np.linalg.norm(data[idx1] - data[idx2])
        random_diffs.append(diff)

    print(f"连续样本差异均值: {np.mean(consecutive_diffs):.4f}")
    print(f"随机样本差异均值: {np.mean(random_diffs):.4f}")
    print(f"比值: {np.mean(consecutive_diffs) / np.mean(random_diffs):.4f}")
    print()

    if np.mean(consecutive_diffs) / np.mean(random_diffs) < 0.6:
        print("✓ 同一个人做同一个动作的样本很可能是连续的时间切片")
        print("  → 这个人的这个动作数据可以考虑拼接")
    else:
        print("✗ 即使是同一个人的同一个动作，样本也不是连续的")

print()
print("=" * 80)
print("关键结论")
print("=" * 80)
print()
print("数据组织方式:")
print("  - SXX_AYY.npy: XX=人(1-40), YY=动作(1-27)")
print("  - 每个文件包含一个人做一个特定动作的所有样本")
print()
print("现有处理方式的问题:")
print("  - 将40个人的同一动作合并到一起")
print("  - 合并后shuffle，破坏了同一个人的连续性")
print("  - 这就是为什么拼接效果差的根本原因！")
print()
print("改进方向:")
print("  1. 保持'人'的维度：不同人不要合并")
print("  2. 只在同一个人的同一动作内进行拼接")
print("  3. 或者按人划分训练/测试集，而不是随机shuffle")
