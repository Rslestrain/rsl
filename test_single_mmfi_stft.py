#!/usr/bin/env python3
"""
测试单个MMFI STFT训练，验证数据生成
"""

import os
import json
import subprocess

dataset = "mmfi_stft"
config_path = "configs/mmfi_short_stft.json"
test_seed = 9999  # 使用一个测试种子

# 读取配置
with open(config_path, 'r') as f:
    config = json.load(f)

config['seed'] = [test_seed]

# 保存临时配置
temp_config_path = f"temp_config/temp_{dataset}_{test_seed}.json"
os.makedirs("temp_config", exist_ok=True)

with open(temp_config_path, 'w') as f:
    json.dump(config, f, indent=2)

print("=" * 80)
print("测试MMFI STFT数据生成和训练")
print("=" * 80)
print(f"配置文件: {temp_config_path}")
print(f"测试种子: {test_seed}")
print()

# 运行训练
cmd = f"/data1/rsl/anaconda3/envs/consense/bin/python main.py --config {temp_config_path}"

print("开始训练...")
print()

result = subprocess.run(cmd, shell=True)

# 清理
if os.path.exists(temp_config_path):
    os.remove(temp_config_path)

print()
print("=" * 80)
print("测试完成")
print("=" * 80)

exit(result.returncode)
