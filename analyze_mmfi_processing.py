import numpy as np
import os

print("=" * 80)
print("MMFI数据处理流程分析")
print("=" * 80)
print()

# 检查处理后的文件是怎么来的
source_dir = "/data1/rsl/LoRA-Sub-DRS-master/data/mmfi/27a100sample40domin/27a100sample40domin/"
processed_dir = "/data1/rsl/LoRA-Sub-DRS-master/data/mmfi/"

print("1. 原始数据 vs 处理后数据")
print("-" * 80)

# 原始：S01_A01.npy
sample_source = np.load(os.path.join(source_dir, "S01_A01.npy"))
print(f"原始 S01_A01.npy: shape={sample_source.shape}")
print(f"  → 1个人, 1个动作, 100个样本")
print()

# 处理后：mmfi_processed_a01.npy
sample_processed = np.load(os.path.join(processed_dir, "mmfi_processed_a01.npy"))
print(f"处理后 mmfi_processed_a01.npy: shape={sample_processed.shape}")
print(f"  → 所有人, 1个动作, {len(sample_processed)}个样本")
print(f"  → {len(sample_processed)} / 40人 = {len(sample_processed)/40:.1f} 样本/人")
print()

print("2. 推测数据处理逻辑")
print("-" * 80)
print("mmfi_processed_a01.npy 应该是这样生成的:")
print("  合并: S01_A01.npy + S02_A01.npy + ... + S40_A01.npy")
print(f"  预期: 40人 × 100样本/人 = 4000样本")
print(f"  实际: {len(sample_processed)}样本")
print()

if len(sample_processed) != 4000:
    print(f"⚠️  样本数不匹配！多了 {len(sample_processed) - 4000} 个样本")
    print("   可能原因:")
    print("   - 有数据增强")
    print("   - 有重复采样")
    print("   - 或者源文件不止100个样本")
    print()

# 验证：检查所有S*_A01.npy的总样本数
print("3. 验证：统计所有人A01动作的总样本数")
print("-" * 80)
total_samples = 0
all_samples = []
for i in range(1, 41):
    fname = f"S{i:02d}_A01.npy"
    fpath = os.path.join(source_dir, fname)
    if os.path.exists(fpath):
        data = np.load(fpath)
        total_samples += len(data)
        all_samples.append(len(data))

print(f"40个人的A01总样本数: {total_samples}")
print(f"样本数分布: min={min(all_samples)}, max={max(all_samples)}, mean={np.mean(all_samples):.1f}")
print()

if total_samples != len(sample_processed):
    print(f"⚠️  总样本数 {total_samples} ≠ 处理后样本数 {len(sample_processed)}")
    print(f"   差异: {len(sample_processed) - total_samples}")
else:
    print("✓ 总样本数匹配！")

print()
print("=" * 80)
print("关键发现：现有数据处理的问题")
print("=" * 80)
print()

# 现在理解问题了
print("问题1: 数据合并破坏了'人'的结构")
print("-" * 80)
print("原始数据:")
print("  - S01的A01: 100个样本（可能有一定连续性）")
print("  - S02的A01: 100个样本（可能有一定连续性）")
print("  - ...")
print()
print("处理后:")
print("  - mmfi_processed_a01: 所有人混在一起")
print("  - shuffle后: 彻底打乱")
print("  - 结果: S01的样本0和样本1可能被分散到不同位置")
print()
print("时域拼接时:")
print("  - 拼接相邻5个样本")
print("  - 但这5个样本可能来自5个不同的人！")
print("  - 例如: [S01样本3, S17样本28, S05样本91, S33样本12, S22样本67]")
print("  - 这样的拼接完全没有意义！")
print()

print("问题2: 数据量减少")
print("-" * 80)
print(f"原始: {total_samples}个样本")
print(f"拼接后: {total_samples // 5}个样本（减少80%）")
print()

print("=" * 80)
print("解决方案")
print("=" * 80)
print()

print("方案1: 按人拼接（保持同一个人的连续性）✓✓✓✓✓")
print("-" * 80)
print("1. 每个人的每个动作单独处理")
print("   S01_A01: 100样本 → 拼接5个 → 20样本")
print("   S02_A01: 100样本 → 拼接5个 → 20样本")
print("   ...")
print("   S40_A01: 100样本 → 拼接5个 → 20样本")
print()
print("2. 然后合并所有人")
print("   总样本: 40人 × 20样本/人 = 800样本（而不是当前的108样本）")
print()
print("3. 训练/测试集按人划分")
print("   训练集: S01-S32 (80%)")
print("   测试集: S33-S40 (20%)")
print("   → 确保测试集是真正的'未见过的人'")
print()

print("方案2: 不拼接，保留原始粒度 ✓✓✓✓✓")
print("-" * 80)
print(f"总样本: {total_samples}个")
print("训练/测试按人划分")
print("针对T=10优化STFT参数")
print()

print("方案3: 智能拼接（滑动窗口）✓✓✓")
print("-" * 80)
print("在同一个人的数据内用滑动窗口:")
print("  S01_A01样本0-4 → 新样本1")
print("  S01_A01样本1-5 → 新样本2")
print("  ...")
print("  S01_A01样本95-99 → 新样本96")
print("  → 100样本变成96样本（而不是20）")
print()

print("推荐方案: 方案2（不拼接）或方案1（按人拼接）")
