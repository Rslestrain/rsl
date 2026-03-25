#!/usr/bin/env python3
"""
检查MMFI源数据中每个人每个动作的样本数
"""

import numpy as np
import os

source_dir = "/data1/rsl/LoRA-Sub-DRS-master/data/mmfi/27a100sample40domin/27a100sample40domin/"

print("=" * 80)
print("MMFI源数据样本数统计")
print("=" * 80)
print()

# 检查所有人的所有动作
subjects_to_check = list(range(1, 41))  # S01-S40
actions = list(range(1, 28))  # A01-A27

# 存储结果
sample_counts = {}

print("1. 检查每个人每个动作的样本数...")
print()

for subject in subjects_to_check:
    subject_counts = []
    for action in actions:
        fname = f"S{subject:02d}_A{action:02d}.npy"
        fpath = os.path.join(source_dir, fname)

        if os.path.exists(fpath):
            data = np.load(fpath)
            count = len(data)
            subject_counts.append(count)
        else:
            subject_counts.append(0)

    sample_counts[subject] = subject_counts

    if subject <= 5 or subject >= 38:  # 显示前5个和后3个
        total = sum(subject_counts)
        avg = total / len(subject_counts)
        print(f"S{subject:02d}: 总样本={total:5d}, 平均={avg:6.1f}/动作, 详细={subject_counts[:3]}...{subject_counts[-3:]}")

print(f"...")
print()

# 统计所有人的样本数
print("2. 统计摘要:")
print("-" * 80)

all_samples = []
for subject in subjects_to_check:
    total = sum(sample_counts[subject])
    all_samples.append(total)

print(f"总人数: {len(subjects_to_check)}")
print(f"每个人的总样本数:")
print(f"  最小值: {min(all_samples)}")
print(f"  最大值: {max(all_samples)}")
print(f"  平均值: {np.mean(all_samples):.1f}")
print(f"  中位数: {np.median(all_samples):.1f}")
print()

# 检查是否所有人的样本数一致
if len(set(all_samples)) == 1:
    print(f"✓ 所有人的样本数一致: {all_samples[0]} 个")
else:
    print(f"⚠️  不同人的样本数不一致")
    unique_counts = set(all_samples)
    for count in sorted(unique_counts):
        num_people = all_samples.count(count)
        print(f"  {count}样本: {num_people}人")

print()

# 检查每个动作的样本数是否一致
print("3. 检查每个动作的样本数:")
print("-" * 80)

action_samples = {}
for action_idx, action in enumerate(actions):
    samples_for_this_action = []
    for subject in subjects_to_check:
        samples_for_this_action.append(sample_counts[subject][action_idx])

    action_samples[action] = samples_for_this_action

    if action <= 3 or action >= 26:  # 显示前3个和后2个
        total = sum(samples_for_this_action)
        avg = np.mean(samples_for_this_action)
        print(f"A{action:02d}: 总样本={total:5d} (40人), 平均={avg:6.1f}/人")

print("...")
print()

# 检查是否每个人每个动作都是100个样本
print("4. 验证样本数一致性:")
print("-" * 80)

all_counts = []
for subject in subjects_to_check:
    all_counts.extend(sample_counts[subject])

unique_counts = set(all_counts)
print(f"所有样本数的唯一值: {sorted(unique_counts)}")

if len(unique_counts) == 1:
    count = list(unique_counts)[0]
    print(f"✓ 所有人的所有动作样本数都是: {count}")
else:
    print(f"⚠️  样本数不一致")

print()

# 计算总数据量
print("5. 总数据量:")
print("-" * 80)

total_samples = sum(all_counts)
print(f"总样本数: {total_samples:,}")
print(f"  = {len(subjects_to_check)}人 × {len(actions)}动作 × {all_counts[0]}样本/动作")
print()

# 推荐的数据划分方案
print("=" * 80)
print("推荐的数据划分方案")
print("=" * 80)
print()

if len(unique_counts) == 1:
    samples_per_person_per_action = list(unique_counts)[0]

    print("方案对比:")
    print()

    # 当前方案
    print("当前方案（8+2人，每人10样本）:")
    print(f"  训练: 8人 × 27动作 × 10样本 = 2,160")
    print(f"  测试: 2人 × 27动作 × 10样本 = 540")
    print(f"  利用率: {(2160+540)/(40*27*100)*100:.1f}%")
    print()

    # 方案1: 使用更多人，更少样本
    print("方案1（32+8人，每人10样本）- 推荐！")
    print(f"  训练: 32人 × 27动作 × 10样本 = 8,640")
    print(f"  测试: 8人 × 27动作 × 10样本 = 2,160")
    print(f"  利用率: {(8640+2160)/(40*27*100)*100:.1f}%")
    print(f"  优势: 训练数据增加4倍，测试集更充分")
    print()

    # 方案2: 所有人，部分样本
    print("方案2（32+8人，每人20样本）:")
    print(f"  训练: 32人 × 27动作 × 20样本 = 17,280")
    print(f"  测试: 8人 × 27动作 × 20样本 = 4,320")
    print(f"  利用率: {(17280+4320)/(40*27*100)*100:.1f}%")
    print(f"  优势: 数据量更大，可能效果更好")
    print()

    # 方案3: 所有人，所有样本
    print("方案3（32+8人，所有样本）:")
    print(f"  训练: 32人 × 27动作 × 100样本 = 86,400")
    print(f"  测试: 8人 × 27动作 × 100样本 = 21,600")
    print(f"  利用率: 100%")
    print(f"  优势: 最大化数据利用")
    print()

print("建议:")
print("  推荐方案1或方案2，在数据量和训练时间之间取得平衡")
print("  如果GPU内存充足，可以尝试方案3")
