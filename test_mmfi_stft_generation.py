#!/usr/bin/env python3
"""
测试MMFI STFT数据生成（无时域拼接版本）
"""

import sys
sys.path.append('/data1/rsl/LoRA-Sub-DRS-master')

from utils.data import iMMFIDataSTFT
import numpy as np

print("=" * 80)
print("测试MMFI STFT数据生成（无时域拼接）")
print("=" * 80)
print()

# 创建数据集对象
args = {'data_path': 'data/'}
data = iMMFIDataSTFT(args)

print("开始生成数据...")
print()

# 触发数据生成
data.download_data()

print()
print("=" * 80)
print("数据生成完成，统计信息：")
print("=" * 80)
print()

print(f"训练集样本数: {len(data.train_data)}")
print(f"测试集样本数: {len(data.test_data)}")
print(f"训练集标签分布: {np.bincount(data.train_targets, minlength=27)}")
print(f"测试集标签分布: {np.bincount(data.test_targets, minlength=27)}")
print()

# 验证是否达到目标
expected_train = 2160
expected_test = 540

if len(data.train_data) == expected_train:
    print(f"✓ 训练集样本数正确: {len(data.train_data)} == {expected_train}")
else:
    print(f"✗ 训练集样本数不符: {len(data.train_data)} != {expected_train}")

if len(data.test_data) == expected_test:
    print(f"✓ 测试集样本数正确: {len(data.test_data)} == {expected_test}")
else:
    print(f"✗ 测试集样本数不符: {len(data.test_data)} != {expected_test}")

print()
print("=" * 80)
