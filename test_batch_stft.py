#!/usr/bin/env python3
"""
STFT批量训练测试脚本
只运行2个种子，用于快速验证功能
"""

import os
import json
import subprocess
import random
import numpy as np
import re
import time
import gc
from datetime import datetime


def cleanup_gpu_memory():
    """清理GPU内存"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            return allocated, cached
    except Exception as e:
        print(f"GPU内存清理失败: {e}")
        return None, None


def generate_test_seeds(n=2):
    """生成n个测试种子"""
    random.seed(42)
    return random.sample(range(0, 100), n)


def run_training(dataset, seed, config_path):
    """运行单个训练任务"""
    with open(config_path, 'r') as f:
        config = json.load(f)

    config['seed'] = [seed]

    temp_config_path = f"temp_config/temp_test_{dataset}_{seed}.json"
    os.makedirs("temp_config", exist_ok=True)

    with open(temp_config_path, 'w') as f:
        json.dump(config, f, indent=2)

    cmd = f"/data1/rsl/anaconda3/envs/consense/bin/python main.py --config {temp_config_path}"

    try:
        cleanup_gpu_memory()
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)  # 10分钟超时

        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

        cleanup_gpu_memory()
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
        cleanup_gpu_memory()
        return False, "", str(e)


def parse_accuracy_from_log(dataset, seed):
    """从日志文件中解析准确率"""
    if dataset == "mmfi_stft":
        log_dir = f"logs/mmfi_stft/12_3_sip/lorasub_drs/Adam/{seed}.log"
    elif dataset == "wiar_stft":
        log_dir = f"logs/wiar_stft/8_2_sip/lorasub_drs/Adam/{seed}.log"
    else:
        return None

    if not os.path.exists(log_dir):
        print(f"日志文件不存在: {log_dir}")
        return None

    try:
        with open(log_dir, 'r') as f:
            content = f.read()

        pattern = r'Average Accuracy:\s*([\d.]+)'
        matches = re.findall(pattern, content)

        if matches:
            accuracy = float(matches[-1])
            return accuracy
        else:
            return None
    except Exception as e:
        print(f"解析日志失败: {e}")
        return None


def main():
    """主函数 - 测试版本，只运行2个种子"""
    print("="*60)
    print("STFT批量训练 - 测试模式")
    print("只运行2个种子验证功能")
    print("="*60)
    print()

    # 选择一个数据集进行测试
    dataset = "wiar_stft"  # 使用Wiar因为训练更快
    config_path = "configs/wiar_stft.json"

    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        return

    # 生成2个测试种子
    seeds = generate_test_seeds(2)
    print(f"测试种子: {seeds}")
    print()

    results = []

    for i, seed in enumerate(seeds, 1):
        print(f"[{i}/2] 运行种子 {seed}...")
        import sys
        sys.stdout.flush()

        success, stdout, stderr = run_training(dataset, seed, config_path)

        if success:
            accuracy = parse_accuracy_from_log(dataset, seed)
            if accuracy is not None:
                results.append({'seed': seed, 'accuracy': accuracy, 'status': 'success'})
                print(f"  ✓ 成功! 准确率: {accuracy:.4f}")
            else:
                results.append({'seed': seed, 'accuracy': None, 'status': 'parse_failed'})
                print(f"  ✗ 日志解析失败")
        else:
            results.append({'seed': seed, 'accuracy': None, 'status': 'training_failed'})
            print(f"  ✗ 训练失败")

        time.sleep(1)
        print()

    # 输出测试结果
    print("="*60)
    print("测试完成!")
    print("="*60)

    valid_accuracies = [r['accuracy'] for r in results if r['accuracy'] is not None]

    if len(valid_accuracies) > 0:
        print(f"成功运行: {len(valid_accuracies)}/{len(seeds)}")
        print(f"准确率: {valid_accuracies}")
        print(f"平均值: {np.mean(valid_accuracies):.4f}")
        print()
        print("✓ 脚本功能验证通过!")
        print()
        print("下一步:")
        print("1. 运行完整的100次训练: ./run_batch_stft.sh background")
        print("2. 或直接运行: /data1/rsl/anaconda3/envs/consense/bin/python batch_train_stft.py")
    else:
        print("✗ 测试失败，请检查配置和环境")
        print()
        print("查看最后一次运行的日志:")
        print(f"  tail -100 logs/wiar_stft/8_2_sip/lorasub_drs/Adam/{seeds[-1]}.log")


if __name__ == "__main__":
    main()
