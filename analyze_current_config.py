#!/usr/bin/env python3
"""
分析当前数据配置和Wiar的对比
找出准确率低的真正原因
"""

import json

print("=" * 80)
print("MMFI vs Wiar 数据配置对比")
print("=" * 80)
print()

# MMFI当前配置
print("MMFI当前配置（按人划分方案A）:")
print("-" * 80)
print("训练集: 8人 × 27动作 × 10样本 = 2,160")
print("测试集: 2人 × 27动作 × 10样本 = 540")
print("每类训练样本: 80")
print("每类测试样本: 20")
print()

# Wiar配置（需要检查）
print("Wiar配置:")
print("-" * 80)
print("类别数: 16")
print("任务划分: 8类初始 + 4个增量任务(每个2类)")

# 检查Wiar的实际数据量
wiar_config_path = "configs/wiar_stft.json"
with open(wiar_config_path, 'r') as f:
    wiar_config = json.load(f)

print(f"Batch size: {wiar_config['batch_size']}")
print()

# 需要查看Wiar的实际训练数据量
print("需要检查: Wiar实际使用了多少训练/测试样本")
print("  → 检查 data/wiar_stft/ 目录")
print()

print("=" * 80)
print("可能的问题分析")
print("=" * 80)
print()

print("问题1: STFT参数")
print("-" * 80)
print("MMFI: nperseg=8, noverlap=4, nfft=16 (T=10)")
print("Wiar: nperseg=20, noverlap=10, nfft=32 (T更大)")
print()
print("影响:")
print("  - T=10太短，STFT频谱分辨率低")
print("  - 可能信息不足以区分27个动作")
print()

print("问题2: 每类样本数 vs 类别数")
print("-" * 80)
print("MMFI: 80样本/类，27类")
print("Wiar: ？样本/类，16类")
print()
print("影响:")
print("  - 如果Wiar每类样本更多，模型学习效果更好")
print("  - 27类比16类更难，需要更多样本")
print()

print("问题3: 测试集设置")
print("-" * 80)
print("当前: 2个完全未见过的人")
print("可能问题:")
print("  - 如果这2个人的动作模式和训练集8个人差异很大")
print("  - 导致严重的泛化问题")
print()
print("建议:")
print("  - 可以尝试使用更多人但较少样本")
print("  - 例如: 16人×5样本 vs 4人×5样本")
print("  - 增加人的多样性，降低个人差异的影响")
print()

print("=" * 80)
print("保持2160/540的替代方案")
print("=" * 80)
print()

schemes = [
    ("当前", 8, 10, 2, 10, "少人多样本，可能人的差异大"),
    ("方案1", 16, 5, 4, 5, "更多人，降低个人差异影响"),
    ("方案2", 4, 20, 1, 20, "少人多样本，但测试只有1人"),
    ("方案3", 2, 40, 1, 20, "极少人，但每人样本多")
]

for name, train_people, train_samples, test_people, test_samples, note in schemes:
    train_total = train_people * 27 * train_samples
    test_total = test_people * 27 * test_samples
    print(f"{name}:")
    print(f"  训练: {train_people}人 × 27动作 × {train_samples}样本 = {train_total}")
    print(f"  测试: {test_people}人 × 27动作 × {test_samples}样本 = {test_total}")
    print(f"  特点: {note}")
    print()

print("=" * 80)
print("建议的改进措施")
print("=" * 80)
print()

print("优先级1: 检查STFT图像质量")
print("  → 查看生成的时频图是否有足够信息")
print("  → 可能需要调整STFT参数")
print()

print("优先级2: 尝试方案1（16+4人，各5样本）")
print("  → 增加人的多样性")
print("  → 测试集4个人更可靠")
print("  → 保持总数据量不变")
print()

print("优先级3: 检查数据加载是否正确")
print("  → 确认train/test划分没问题")
print("  → 确认标签正确")
print()
