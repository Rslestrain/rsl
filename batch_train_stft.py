#!/usr/bin/env python3
"""
STFT数据集批量训练脚本
支持 mmfi_stft 和 wiar_stft 数据集
每个数据集使用100个随机种子，统计准确率结果并计算Top5平均
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
            # 清理GPU缓存
            torch.cuda.empty_cache()
            # 强制垃圾回收
            gc.collect()

            # 获取当前GPU内存使用情况
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            cached = torch.cuda.memory_reserved() / 1024**3  # GB

            return allocated, cached
    except Exception as e:
        print(f"GPU内存清理失败: {e}")
        return None, None


def generate_random_seeds(n=100, seed_range=(0, 10000)):
    """生成n个不重复的随机种子"""
    random.seed(42)  # 固定随机种子确保可重复性
    return random.sample(range(seed_range[0], seed_range[1]), n)


def run_training_with_memory_management(dataset, seed, config_path):
    """运行单个训练任务，包含内存管理"""
    # 修改配置文件中的种子
    with open(config_path, 'r') as f:
        config = json.load(f)

    config['seed'] = [seed]

    # 保存临时配置文件
    temp_config_path = f"temp_config/temp_{dataset}_{seed}.json"
    os.makedirs("temp_config", exist_ok=True)

    with open(temp_config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # 运行训练
    cmd = f"/data1/rsl/anaconda3/envs/consense/bin/python main.py --config {temp_config_path}"

    try:
        # 训练前清理内存
        cleanup_gpu_memory()

        # 运行训练
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3600)  # 1小时超时

        # 清理临时配置文件
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

        # 训练完成后清理GPU内存
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
    # 根据数据集和配置确定日志路径
    if dataset == "mmfi_stft":
        log_dir = f"logs/mmfi_stft/12_3_sip/lorasub_drs/Adam/{seed}.log"
    elif dataset == "wiar_stft":
        log_dir = f"logs/wiar_stft/8_2_sip/lorasub_drs/Adam/{seed}.log"
    else:
        print(f"未知数据集: {dataset}")
        return None

    if not os.path.exists(log_dir):
        print(f"日志文件不存在: {log_dir}")
        return None

    try:
        with open(log_dir, 'r') as f:
            content = f.read()

        # 查找最后一个 "Average Accuracy" 值
        pattern = r'Average Accuracy:\s*([\d.]+)'
        matches = re.findall(pattern, content)

        if matches:
            # 返回最后一个匹配的值（最终的平均准确率）
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

    # Top5统计
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
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
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
    # 可以选择运行的数据集
    datasets = ["mmfi_stft", "wiar_stft"]  # 同时运行MMFI和Wiar
    # datasets = ["mmfi_stft"]  # 只运行MMFI STFT
    # datasets = ["wiar_stft"]  # 只运行Wiar STFT

    config_files = {
        "mmfi_stft": "configs/mmfi_short_stft.json",
        "wiar_stft": "configs/wiar_stft.json"
    }

    # 检查GPU内存状态
    free_memory = check_gpu_memory()

    if free_memory < 2.0:  # 小于2GB可用内存
        print("警告: GPU可用内存较少，建议检查训练配置或使用CPU模式")

    # 生成100个随机种子
    seeds = generate_random_seeds(100)
    print(f"Generated {len(seeds)} random seeds: {seeds[:10]}...")

    # 结果存储
    all_results = {}

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset}")
        print(f"{'='*60}")

        config_path = config_files[dataset]
        if not os.path.exists(config_path):
            print(f"Config file not found: {config_path}")
            continue

        dataset_results = []

        for i, seed in enumerate(seeds, 1):
            print(f"\n[{dataset}] Seed {i}/100: {seed}")
            import sys
            sys.stdout.flush()

            # 检查GPU内存状态
            current_free_memory = check_gpu_memory()
            if current_free_memory < 1.0:  # 小于1GB可用内存
                print("GPU内存不足，强制清理内存...")
                sys.stdout.flush()
                cleanup_gpu_memory()
                time.sleep(3)

            # 运行训练
            success, stdout, stderr = run_training_with_memory_management(dataset, seed, config_path)

            # 检查是否有CUDA内存不足错误
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
                # 解析准确率
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

            # 每个种子训练完成后等待
            time.sleep(2)

        # 计算统计信息
        valid_accuracies = [r['accuracy'] for r in dataset_results if r['accuracy'] is not None]
        stats = calculate_statistics(valid_accuracies)

        all_results[dataset] = {
            'results': dataset_results,
            'statistics': stats
        }

        # 打印当前数据集统计结果
        if stats:
            print(f"\n{dataset} Statistics:")
            print(f"  Total Runs: {stats['count']}")
            print(f"  Mean Accuracy: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"  Min Accuracy: {stats['min']:.4f}")
            print(f"  Max Accuracy: {stats['max']:.4f}")
            if 'top5_mean' in stats:
                print(f"  Top5 Mean: {stats['top5_mean']:.4f} ± {stats['top5_std']:.4f}")
                print(f"  Top5 Values: {[f'{v:.4f}' for v in stats['top5_values']]}")

    # 保存完整结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"batch_results_stft_{timestamp}.json"

    with open(result_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("Batch training completed!")
    print(f"Results saved to: {result_file}")

    # 打印最终汇总
    print("\n" + "="*60)
    print("Final Summary")
    print("="*60)
    for dataset in datasets:
        if dataset in all_results and all_results[dataset]['statistics']:
            stats = all_results[dataset]['statistics']
            print(f"\n{dataset.upper()}:")
            print(f"  Runs: {stats['count']}")
            print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
            if 'top5_mean' in stats:
                print(f"  Top5 Mean: {stats['top5_mean']:.4f} ± {stats['top5_std']:.4f}")
                print(f"  Top5 Seeds: {[r['seed'] for r in all_results[dataset]['results'] if r['accuracy'] is not None and r['accuracy'] >= min(stats['top5_values'])][:5]}")


if __name__ == "__main__":
    main()
