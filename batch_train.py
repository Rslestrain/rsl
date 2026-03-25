#!/usr/bin/env python3
"""
改进版批量训练脚本 - 顺序执行wiar、mmfi、xrf三个数据集
每个数据集使用50个随机种子，统计准确率结果
添加完整的内存管理和错误处理机制
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


def generate_random_seeds(n=50, seed_range=(0, 10000)):
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
    cmd = f"python main.py --config {temp_config_path}"
    # print(f"Running: {cmd}")
    
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
        # 确保清理临时文件
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
        cleanup_gpu_memory()
        return False, "", str(e)


def parse_accuracy_from_log(dataset, seed):
    """从日志文件中解析准确率"""
    # 根据数据集和配置确定日志路径
    if dataset == "wiar_short":
        log_dir = f"logs/wiar/8_2_sip/lorasub_drs/Adam/{seed}.log"
    elif dataset == "wiar_long":
        log_dir = f"logs/wiar/2_2_sip/lorasub_drs/Adam/{seed}.log"
    elif dataset == "mmfi_short":
        log_dir = f"logs/mmfi/12_3_sip/lorasub_drs/Adam/{seed}.log"
    elif dataset == "mmfi_long":
        log_dir = f"logs/mmfi/3_3_sip/lorasub_drs/Adam/{seed}.log"
    elif dataset == "xrf_short":
        log_dir = f"logs/xrf/24_6_sip/lorasub_drs/Adam/{seed}.log"
    elif dataset == "xrf_long":
        log_dir = f"logs/xrf/6_6_sip/lorasub_drs/Adam/{seed}.log"
    else:
        return None
    
    if not os.path.exists(log_dir):
        return None
    
    # 读取日志文件
    with open(log_dir, 'r') as f:
        log_content = f.read()
    
    # 查找最后一个Average Accuracy
    pattern = r'Average Accuracy: ([\d.]+)'
    matches = re.findall(pattern, log_content)
    
    if matches:
        # 返回最后一个Average Accuracy
        return float(matches[-1])
    else:
        return None


def calculate_statistics(accuracies):
    """计算统计信息"""
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
    # datasets = ["wiar_short", "wiar_long", "mmfi_short", "mmfi_long", "xrf_short", "xrf_long"]
    datasets = ["xrf_long"]
    config_files = {
        "wiar_short": "configs/wiar_short.json",
        "wiar_long": "configs/wiar_long.json",
        "mmfi_short": "configs/mmfi_short.json",
        "mmfi_long": "configs/mmfi_long.json",
        "xrf_short": "configs/xrf_short.json",
        "xrf_long": "configs/xrf_long.json"
    }
    
    # 检查GPU内存状态
    free_memory = check_gpu_memory()
    
    # 如果GPU内存不足，建议减少batch_size或使用CPU
    if free_memory < 2.0:  # 小于2GB可用内存
        print("警告: GPU可用内存较少，建议检查训练配置或使用CPU模式")
    
    # 生成随机种子
    seeds = generate_random_seeds(50)
    print(f"Generated {len(seeds)} random seeds: {seeds[:5]}...")
    
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
            print(f"\n[{dataset}] Seed {i}/50: {seed}")
            import sys
            sys.stdout.flush()
            
            # 检查GPU内存状态
            current_free_memory = check_gpu_memory()
            if current_free_memory < 1.0:  # 小于1GB可用内存
                print("GPU内存不足，强制清理内存...")
                import sys
                sys.stdout.flush()
                cleanup_gpu_memory()
                time.sleep(3)  # 等待3秒
            
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
                # 强制清理内存并等待一段时间
                cleanup_gpu_memory()
                time.sleep(5)  # 等待5秒让GPU完全释放内存
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
                    # 强制刷新输出缓冲区
                    import sys
                    sys.stdout.flush()
                else:
                    dataset_results.append({
                        'seed': seed,
                        'accuracy': None,
                        'status': 'log_parse_failed'
                    })
                    print("Failed to parse accuracy from log")
                    # 强制刷新输出缓冲区
                    import sys
                    sys.stdout.flush()
            else:
                dataset_results.append({
                    'seed': seed,
                    'accuracy': None,
                    'status': 'training_failed'
                })
                print(f"Training failed: {stderr}")
            
            # 每个种子训练完成后等待一段时间，确保GPU内存完全释放
            time.sleep(2)  # 等待2秒
        
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
            print(f"  Mean Accuracy: {stats['mean']:.4f}")
            print(f"  Std Deviation: {stats['std']:.4f}")
            print(f"  Min Accuracy: {stats['min']:.4f}")
            print(f"  Max Accuracy: {stats['max']:.4f}")
            if 'top5_mean' in stats:
                print(f"  Top5 Mean: {stats['top5_mean']:.4f}")
                print(f"  Top5 Std: {stats['top5_std']:.4f}")
                print(f"  Top5 Values: {stats['top5_values']}")
    
    # 保存完整结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"batch_results_{timestamp}.json"
    
    with open(result_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Batch training completed!")
    print(f"Results saved to: {result_file}")
    
    # 打印最终汇总
    print("\nFinal Summary:")
    for dataset in datasets:
        if dataset in all_results and all_results[dataset]['statistics']:
            stats = all_results[dataset]['statistics']
            print(f"\n{dataset.upper()}:")
            print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
            if 'top5_mean' in stats:
                print(f"  Top5 Mean: {stats['top5_mean']:.4f} ± {stats['top5_std']:.4f}")


if __name__ == "__main__":
    main()