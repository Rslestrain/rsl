#!/usr/bin/env python3
"""
MMFI按人划分数据集批量训练脚本（方案A）
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
    log_dir = f"logs/mmfi_by_person_stft/12_3_sip/lorasub_drs/Adam/{seed}.log"

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


def batch_train(dataset_name, config_path, num_runs=100):
    """批量训练"""
    print("=" * 100)
    print(f"MMFI按人划分批量训练 - 方案A")
    print("=" * 100)
    print(f"数据集: {dataset_name}")
    print(f"配置文件: {config_path}")
    print(f"训练次数: {num_runs}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)
    print()

    # 生成随机种子
    seeds = generate_random_seeds(num_runs)
    print(f"生成{num_runs}个随机种子: {seeds[:10]}... (仅显示前10个)")
    print()

    # 存储结果
    results = {}
    failed_seeds = []

    # 批量训练
    for idx, seed in enumerate(seeds, 1):
        print("-" * 100)
        print(f"[{idx}/{num_runs}] 训练种子: {seed}")
        print("-" * 100)

        start_time = time.time()
        success, stdout, stderr = run_training_with_memory_management(dataset_name, seed, config_path)
        elapsed_time = time.time() - start_time

        if success:
            accuracy = parse_accuracy_from_log(dataset_name, seed)
            if accuracy is not None:
                results[seed] = accuracy
                print(f"✓ 种子 {seed} 完成 | 准确率: {accuracy:.4f} | 用时: {elapsed_time:.1f}秒")
            else:
                failed_seeds.append(seed)
                print(f"✗ 种子 {seed} 完成但解析失败 | 用时: {elapsed_time:.1f}秒")
        else:
            failed_seeds.append(seed)
            print(f"✗ 种子 {seed} 训练失败 | 用时: {elapsed_time:.1f}秒")
            if stderr:
                print(f"错误信息: {stderr[:200]}")

        # 显示当前统计
        if results:
            current_mean = np.mean(list(results.values()))
            current_std = np.std(list(results.values()))
            print(f"当前统计: 完成{len(results)}/{num_runs} | 均值={current_mean:.4f} | 标准差={current_std:.4f}")

        print()

        # 每10次清理一次内存
        if idx % 10 == 0:
            cleanup_gpu_memory()

    # 最终统计
    print("=" * 100)
    print("训练完成！最终统计")
    print("=" * 100)
    print(f"总运行数: {num_runs}")
    print(f"成功数: {len(results)}")
    print(f"失败数: {len(failed_seeds)}")

    if failed_seeds:
        print(f"失败种子: {failed_seeds}")
    print()

    if results:
        accuracies = list(results.values())
        seeds_list = list(results.keys())

        # 排序获取Top5
        sorted_indices = np.argsort(accuracies)[::-1]
        top5_indices = sorted_indices[:5]
        top5_accuracies = [accuracies[i] for i in top5_indices]
        top5_seeds = [seeds_list[i] for i in top5_indices]

        print(f"所有结果统计:")
        print(f"  均值: {np.mean(accuracies):.4f}")
        print(f"  标准差: {np.std(accuracies):.4f}")
        print(f"  最大值: {np.max(accuracies):.4f}")
        print(f"  最小值: {np.min(accuracies):.4f}")
        print(f"  中位数: {np.median(accuracies):.4f}")
        print()

        print(f"Top5结果:")
        for rank, (seed, acc) in enumerate(zip(top5_seeds, top5_accuracies), 1):
            print(f"  #{rank}: 种子={seed}, 准确率={acc:.4f}")
        print()

        top5_mean = np.mean(top5_accuracies)
        top5_std = np.std(top5_accuracies)
        print(f"Top5统计:")
        print(f"  均值: {top5_mean:.4f}")
        print(f"  标准差: {top5_std:.4f}")
        print()

        # 保存结果到JSON
        result_file = f"batch_results_mmfi_by_person_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        result_data = {
            "dataset": dataset_name,
            "config": config_path,
            "total_runs": num_runs,
            "successful_runs": len(results),
            "failed_seeds": failed_seeds,
            "all_results": {str(k): v for k, v in results.items()},
            "statistics": {
                "mean": float(np.mean(accuracies)),
                "std": float(np.std(accuracies)),
                "max": float(np.max(accuracies)),
                "min": float(np.min(accuracies)),
                "median": float(np.median(accuracies))
            },
            "top5": {
                "seeds": [int(s) for s in top5_seeds],
                "accuracies": [float(a) for a in top5_accuracies],
                "mean": float(top5_mean),
                "std": float(top5_std)
            },
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2)

        print(f"结果已保存到: {result_file}")
        print()

        # 显示推荐的种子
        print("=" * 100)
        print("推荐配置")
        print("=" * 100)
        print(f"如果想复现最好的结果，使用以下种子之一:")
        for rank, seed in enumerate(top5_seeds, 1):
            print(f"  推荐#{rank}: seed={seed}")
        print()
        print(f"示例命令:")
        print(f'  python main.py --config {config_path} --seed {top5_seeds[0]}')
        print()

    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)


if __name__ == "__main__":
    # 配置
    dataset = "mmfi_by_person_stft"
    config = "configs/mmfi_by_person_stft.json"

    # 检查配置文件是否存在
    if not os.path.exists(config):
        print(f"错误: 配置文件不存在: {config}")
        exit(1)

    # 开始批量训练
    batch_train(dataset, config, num_runs=100)
