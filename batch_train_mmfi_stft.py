#!/usr/bin/env python3
"""
MMFI STFT数据集批量训练脚本
100个随机种子，统计准确率结果并计算Top5平均
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


def generate_random_seeds(n=100, seed_range=(0, 10000)):
    """生成n个不重复的随机种子"""
    random.seed(42)
    return random.sample(range(seed_range[0], seed_range[1]), n)


def run_training_with_memory_management(dataset, seed, config_path):
    """运行单个训练任务，包含内存管理"""
    with open(config_path, 'r') as f:
        config = json.load(f)

    config['seed'] = [seed]

    temp_config_path = f"temp_config/temp_{dataset}_{seed}.json"
    os.makedirs("temp_config", exist_ok=True)

    with open(temp_config_path, 'w') as f:
        json.dump(config, f, indent=2)

    cmd = f"/data1/rsl/anaconda3/envs/consense/bin/python main.py --config {temp_config_path}"

    try:
        cleanup_gpu_memory()
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3600)

        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

        cleanup_gpu_memory()
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print(f"训练超时: 种子 {seed}")
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
        cleanup_gpu_memory()
        return False, "", "Training timeout"
    except Exception as e:
        print(f"训练异常: {e}")
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
        cleanup_gpu_memory()
        return False, "", str(e)


def parse_accuracy_from_log(dataset, seed):
    """从日志文件中解析准确率"""
    log_dir = f"logs/mmfi_stft/12_3_sip/lorasub_drs/Adam/{seed}.log"

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
            print(f"未找到准确率: {log_dir}")
            return None
    except Exception as e:
        print(f"解析日志文件失败 {log_dir}: {e}")
        return None


def calculate_statistics(accuracies):
    """计算统计信息，包括Top5平均"""
    if not accuracies:
        return None

    acc_array = np.array(accuracies)

    stats = {
        'mean': np.mean(acc_array),
        'std': np.std(acc_array),
        'count': len(acc_array),
        'min': np.min(acc_array),
        'max': np.max(acc_array)
    }

    if len(acc_array) >= 5:
        top5 = np.sort(acc_array)[-5:]
        stats['top5_mean'] = np.mean(top5)
        stats['top5_std'] = np.std(top5)
        stats['top5_values'] = top5.tolist()

    return stats


def check_gpu_memory():
    """检查GPU内存状态"""
    try:
        import torch
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            allocated = torch.cuda.memory_allocated() / 1024**3
            free_memory = total_memory - allocated
            return free_memory
        else:
            print("GPU不可用，使用CPU模式")
            return 0
    except Exception as e:
        print(f"检查GPU内存失败: {e}")
        return 0


def main():
    """主函数"""
    dataset = "mmfi_stft"
    config_path = "configs/mmfi_short_stft.json"

    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return

    free_memory = check_gpu_memory()
    if free_memory < 2.0:
        print("警告: GPU可用内存较少")

    seeds = generate_random_seeds(100)
    print(f"Generated {len(seeds)} random seeds: {seeds[:10]}...")

    all_results = {}
    dataset_results = []

    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset}")
    print(f"{'='*60}")

    for i, seed in enumerate(seeds, 1):
        print(f"\n[{dataset}] Seed {i}/100: {seed}")
        import sys
        sys.stdout.flush()

        current_free_memory = check_gpu_memory()
        if current_free_memory < 1.0:
            print("GPU内存不足，强制清理内存...")
            sys.stdout.flush()
            cleanup_gpu_memory()
            time.sleep(3)

        success, stdout, stderr = run_training_with_memory_management(dataset, seed, config_path)

        if "CUDA out of memory" in stderr or "CUDA out of memory" in stdout:
            print("检测到CUDA内存不足错误，跳过此种子并继续下一个")
            dataset_results.append({
                'seed': seed,
                'accuracy': None,
                'status': 'cuda_oom'
            })
            cleanup_gpu_memory()
            time.sleep(5)
            continue

        if success:
            accuracy = parse_accuracy_from_log(dataset, seed)

            if accuracy is not None:
                dataset_results.append({
                    'seed': seed,
                    'accuracy': accuracy,
                    'status': 'success'
                })
                print(f"Accuracy: {accuracy:.4f}")
                sys.stdout.flush()
            else:
                dataset_results.append({
                    'seed': seed,
                    'accuracy': None,
                    'status': 'log_parse_failed'
                })
                print("Failed to parse accuracy from log")
                sys.stdout.flush()
        else:
            dataset_results.append({
                'seed': seed,
                'accuracy': None,
                'status': 'training_failed'
            })
            print(f"Training failed: {stderr[:200]}")

        time.sleep(2)

    valid_accuracies = [r['accuracy'] for r in dataset_results if r['accuracy'] is not None]
    stats = calculate_statistics(valid_accuracies)

    all_results[dataset] = {
        'results': dataset_results,
        'statistics': stats
    }

    if stats:
        print(f"\n{dataset} Statistics:")
        print(f"  Total Runs: {stats['count']}")
        print(f"  Mean Accuracy: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  Min Accuracy: {stats['min']:.4f}")
        print(f"  Max Accuracy: {stats['max']:.4f}")
        if 'top5_mean' in stats:
            print(f"  Top5 Mean: {stats['top5_mean']:.4f} ± {stats['top5_std']:.4f}")
            print(f"  Top5 Values: {[f'{v:.4f}' for v in stats['top5_values']]}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"batch_results_mmfi_stft_{timestamp}.json"

    with open(result_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("Batch training completed!")
    print(f"Results saved to: {result_file}")

    print(f"\n{dataset.upper()}:")
    if stats:
        print(f"  Runs: {stats['count']}")
        print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
        if 'top5_mean' in stats:
            print(f"  Top5 Mean: {stats['top5_mean']:.4f} ± {stats['top5_std']:.4f}")
            top5_seeds = [r['seed'] for r in dataset_results if r['accuracy'] is not None and r['accuracy'] >= min(stats['top5_values'])][:5]
            print(f"  Top5 Seeds: {top5_seeds}")


if __name__ == "__main__":
    main()
