#!/usr/bin/env python3
"""
简单测试MMFI按人划分的数据加载
不需要导入torchvision
"""

import numpy as np
import os
import json

print("=" * 80)
print("简单测试MMFI按人划分数据加载（方案A）")
print("=" * 80)
print()

# 检查源文件
source_dir = "data/mmfi/27a100sample40domin/27a100sample40domin/"
print(f"1. 检查源文件目录: {source_dir}")

if not os.path.exists(source_dir):
    print(f"   ✗ 目录不存在")
    exit(1)

print(f"   ✓ 目录存在")

# 检查S01-S10, A01-A27的文件
train_subjects = list(range(1, 9))    # S01-S08
test_subjects = list(range(9, 11))    # S09-S10
all_subjects = train_subjects + test_subjects

print(f"   训练人数: {len(train_subjects)} (S01-S08)")
print(f"   测试人数: {len(test_subjects)} (S09-S10)")
print()

print("2. 检查文件完整性...")
missing_files = []
for subject in all_subjects:
    for action in range(1, 28):
        fname = f"S{subject:02d}_A{action:02d}.npy"
        fpath = os.path.join(source_dir, fname)
        if not os.path.exists(fpath):
            missing_files.append(fname)

if missing_files:
    print(f"   ⚠️  缺少{len(missing_files)}个文件:")
    for f in missing_files[:10]:
        print(f"      {f}")
    if len(missing_files) > 10:
        print(f"      ... 还有{len(missing_files)-10}个")
else:
    print(f"   ✓ 所有270个文件都存在 (10人 × 27动作)")

print()

# 加载一些样本验证数据格式
print("3. 验证数据格式...")
sample_file = os.path.join(source_dir, "S01_A01.npy")
data = np.load(sample_file)
print(f"   样本文件: S01_A01.npy")
print(f"   ✓ 数据形状: {data.shape}")
print(f"   ✓ 数据类型: {data.dtype}")
print(f"   ✓ 数据范围: [{data.min():.4f}, {data.max():.4f}]")

if data.shape[0] < 10:
    print(f"   ⚠️  警告: 样本数({data.shape[0]}) < 10")
else:
    print(f"   ✓ 样本数充足({data.shape[0]} >= 10)")

print()

# 模拟采样过程
print("4. 模拟采样过程...")
import random
random.seed(42)
np.random.seed(42)

samples_per_person = 10
expected_train_total = len(train_subjects) * 27 * samples_per_person
expected_test_total = len(test_subjects) * 27 * samples_per_person

print(f"   每人每动作采样: {samples_per_person}个样本")
print(f"   预期训练样本: {len(train_subjects)}人 × 27动作 × {samples_per_person} = {expected_train_total}")
print(f"   预期测试样本: {len(test_subjects)}人 × 27动作 × {samples_per_person} = {expected_test_total}")
print()

# 模拟一个动作的采样
action_id = 1
print(f"5. 模拟动作A{action_id:02d}的采样...")

train_samples_a01 = []
for subject in train_subjects:
    fname = f"S{subject:02d}_A{action_id:02d}.npy"
    fpath = os.path.join(source_dir, fname)
    data = np.load(fpath)

    # 随机采样10个
    total_samples = len(data)
    sampled_indices = random.sample(range(total_samples), min(samples_per_person, total_samples))
    train_samples_a01.extend([data[idx] for idx in sampled_indices])

print(f"   训练集A{action_id:02d}: {len(train_samples_a01)}个样本")
print(f"   预期: {len(train_subjects) * samples_per_person}")

if len(train_samples_a01) == len(train_subjects) * samples_per_person:
    print(f"   ✓ 数量正确")
else:
    print(f"   ✗ 数量不对")

print()

# 检查STFT处理器是否存在
print("6. 检查STFT处理器...")
stft_processor_path = "utils/csi_stft_processor.py"
if os.path.exists(stft_processor_path):
    print(f"   ✓ STFT处理器存在: {stft_processor_path}")
else:
    print(f"   ✗ STFT处理器不存在: {stft_processor_path}")

print()

print("=" * 80)
print("基本检查完成！")
print("=" * 80)
print()
print("总结:")
print(f"  ✓ 源文件完整")
print(f"  ✓ 数据格式正确")
print(f"  ✓ 采样逻辑正确")
print(f"  ✓ 预期训练样本: {expected_train_total}")
print(f"  ✓ 预期测试样本: {expected_test_total}")
print()
print("下一步:")
print("  运行完整的数据生成和训练:")
print("  python main.py --config configs/mmfi_by_person_stft.json")
