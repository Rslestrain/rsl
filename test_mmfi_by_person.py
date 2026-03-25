#!/usr/bin/env python3
"""
测试MMFI按人划分的数据加载流程（方案A）
"""

import sys
import json
import os

print("=" * 80)
print("测试MMFI按人划分数据加载流程（方案A）")
print("=" * 80)
print()

# 加载配置
config_path = "configs/mmfi_by_person_stft.json"
print(f"1. 加载配置文件: {config_path}")
with open(config_path, 'r') as f:
    config = json.load(f)
print(f"   数据集: {config['dataset']}")
print(f"   初始类别: {config['init_cls']}")
print(f"   增量: {config['increment']}")
print(f"   总会话数: {config['total_sessions']}")
print()

# 导入必要的模块
print("2. 导入模块...")
from utils.data_manager import DataManager
print("   ✓ DataManager导入成功")
print()

# 检查源文件是否存在
print("3. 检查源文件...")
source_dir = "data/mmfi/27a100sample40domin/27a100sample40domin/"
if os.path.exists(source_dir):
    files = os.listdir(source_dir)
    print(f"   ✓ 源目录存在: {source_dir}")
    print(f"   ✓ 文件数: {len(files)}")

    # 检查需要的文件
    required_files = [f"S{i:02d}_A{j:02d}.npy" for i in range(1, 11) for j in range(1, 28)]
    missing = [f for f in required_files if f not in files]
    if missing:
        print(f"   ⚠️  缺少{len(missing)}个文件")
        if len(missing) <= 5:
            print(f"   缺少: {missing}")
    else:
        print(f"   ✓ 所有需要的文件都存在 (S01-S10, A01-A27)")
else:
    print(f"   ✗ 源目录不存在: {source_dir}")
    print("   请确保数据文件在正确的位置")
    sys.exit(1)

print()

# 创建DataManager
print("4. 创建DataManager并加载数据...")
print("   这将需要一些时间来处理STFT...")
print()

try:
    data_manager = DataManager(
        dataset_name=config['dataset'],
        shuffle=config['shuffle'],
        seed=config['seed'][0],
        init_cls=config['init_cls'],
        increment=config['increment'],
        args=config
    )
    print("   ✓ DataManager创建成功")
    print()
except Exception as e:
    print(f"   ✗ 创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 验证数据
print("5. 验证数据加载结果...")
print(f"   训练集大小: {len(data_manager._train_data)}")
print(f"   测试集大小: {len(data_manager._test_data)}")
print(f"   训练标签唯一值: {len(set(data_manager._train_targets))}")
print(f"   测试标签唯一值: {len(set(data_manager._test_targets))}")
print()

# 检查是否符合预期
expected_train = 2160  # 8人 × 27动作 × 10样本
expected_test = 540    # 2人 × 27动作 × 10样本

print("6. 检查数据量是否符合预期...")
if len(data_manager._train_data) == expected_train:
    print(f"   ✓ 训练集: {len(data_manager._train_data)} == {expected_train}")
else:
    print(f"   ⚠️  训练集: {len(data_manager._train_data)} != {expected_train}")

if len(data_manager._test_data) == expected_test:
    print(f"   ✓ 测试集: {len(data_manager._test_data)} == {expected_test}")
else:
    print(f"   ⚠️  测试集: {len(data_manager._test_data)} != {expected_test}")

print()

# 检查标签分布
import numpy as np
train_label_counts = np.bincount(data_manager._train_targets, minlength=27)
test_label_counts = np.bincount(data_manager._test_targets, minlength=27)

print("7. 标签分布检查...")
print(f"   训练集每类样本数:")
for i in range(0, 27, 9):
    counts = train_label_counts[i:i+9]
    print(f"     类{i:02d}-{min(i+8, 26):02d}: {counts.tolist()}")

print(f"   测试集每类样本数:")
for i in range(0, 27, 9):
    counts = test_label_counts[i:i+9]
    print(f"     类{i:02d}-{min(i+8, 26):02d}: {counts.tolist()}")

expected_per_class_train = 80  # 8人 × 10样本
expected_per_class_test = 20   # 2人 × 10样本

if all(count == expected_per_class_train for count in train_label_counts):
    print(f"   ✓ 训练集每类{expected_per_class_train}个样本，分布均匀")
else:
    print(f"   ⚠️  训练集分布不均匀")

if all(count == expected_per_class_test for count in test_label_counts):
    print(f"   ✓ 测试集每类{expected_per_class_test}个样本，分布均匀")
else:
    print(f"   ⚠️  测试集分布不均匀")

print()

# 检查任务划分
print("8. 检查任务划分...")
print(f"   总任务数: {data_manager.nb_tasks}")
for task_id in range(data_manager.nb_tasks):
    task_size = data_manager.get_task_size(task_id)
    print(f"   任务{task_id}: {task_size}个类别")

print()

print("=" * 80)
print("测试完成！")
print("=" * 80)
print()
print("下一步: 运行训练")
print(f"  python main.py --config {config_path}")
