#!/usr/bin/env python3
"""
对比分析：MMFI原方案 vs 按人划分方案A
比较两种方法的准确率提升
"""

import json
import glob
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime


def load_latest_result(pattern):
    """加载最新的结果文件"""
    files = glob.glob(pattern)
    if not files:
        return None

    # 按修改时间排序，获取最新的
    latest = max(files, key=os.path.getmtime)
    print(f"加载: {latest}")

    with open(latest, 'r') as f:
        return json.load(f)


def compare_results():
    """对比两种方法的结果"""
    print("=" * 80)
    print("MMFI方法对比分析")
    print("=" * 80)
    print()

    # 加载原方案结果（拼接方案）
    print("1. 加载原方案结果（拼接）...")
    old_result = load_latest_result("batch_results_mmfi_stft_*.json")

    # 加载新方案结果（按人划分）
    print("2. 加载新方案结果（按人划分）...")
    new_result = load_latest_result("batch_results_mmfi_by_person_*.json")

    print()

    if old_result is None and new_result is None:
        print("错误: 未找到任何结果文件")
        return
    elif old_result is None:
        print("警告: 未找到原方案结果，仅显示新方案")
        display_single_result("新方案（按人划分）", new_result)
        return
    elif new_result is None:
        print("警告: 未找到新方案结果，仅显示原方案")
        display_single_result("原方案（拼接）", old_result)
        return

    # 提取统计数据
    old_stats = old_result.get('statistics', {})
    new_stats = new_result.get('statistics', {})

    old_top5 = old_result.get('top5', {})
    new_top5 = new_result.get('top5', {})

    # 显示对比
    print("=" * 80)
    print("整体统计对比")
    print("=" * 80)
    print()

    print(f"{'指标':<20} {'原方案（拼接）':<20} {'新方案（按人划分）':<20} {'提升':<15}")
    print("-" * 80)

    metrics = ['mean', 'std', 'max', 'min', 'median']
    metric_names = ['平均准确率', '标准差', '最大准确率', '最小准确率', '中位数']

    for metric, name in zip(metrics, metric_names):
        old_val = old_stats.get(metric, 0)
        new_val = new_stats.get(metric, 0)
        improvement = new_val - old_val
        improvement_pct = (improvement / old_val * 100) if old_val > 0 else 0

        print(f"{name:<20} {old_val:<20.4f} {new_val:<20.4f} {improvement:>+.4f} ({improvement_pct:>+.1f}%)")

    print()
    print("=" * 80)
    print("Top5对比")
    print("=" * 80)
    print()

    old_top5_mean = old_top5.get('mean', 0)
    new_top5_mean = new_top5.get('mean', 0)
    top5_improvement = new_top5_mean - old_top5_mean
    top5_improvement_pct = (top5_improvement / old_top5_mean * 100) if old_top5_mean > 0 else 0

    print(f"原方案Top5均值: {old_top5_mean:.4f}")
    print(f"新方案Top5均值: {new_top5_mean:.4f}")
    print(f"提升: {top5_improvement:+.4f} ({top5_improvement_pct:+.1f}%)")
    print()

    print("原方案Top5种子:", old_top5.get('seeds', [])[:5])
    print("新方案Top5种子:", new_top5.get('seeds', [])[:5])
    print()

    # 可视化对比
    print("=" * 80)
    print("生成可视化对比图...")
    print("=" * 80)

    # 获取所有结果
    old_all = old_result.get('all_results', {})
    new_all = new_result.get('all_results', {})

    old_accuracies = list(old_all.values())
    new_accuracies = list(new_all.values())

    # 创建对比图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('MMFI方法对比：原方案（拼接） vs 新方案（按人划分）', fontsize=16, fontweight='bold')

    # 1. 分布直方图
    ax1 = axes[0, 0]
    ax1.hist(old_accuracies, bins=30, alpha=0.6, label=f'原方案 (均值={old_stats.get("mean", 0):.4f})',
             color='red', edgecolor='black')
    ax1.hist(new_accuracies, bins=30, alpha=0.6, label=f'新方案 (均值={new_stats.get("mean", 0):.4f})',
             color='green', edgecolor='black')
    ax1.axvline(old_stats.get('mean', 0), color='red', linestyle='--', linewidth=2, label='原方案均值')
    ax1.axvline(new_stats.get('mean', 0), color='green', linestyle='--', linewidth=2, label='新方案均值')
    ax1.set_xlabel('准确率', fontsize=12)
    ax1.set_ylabel('频数', fontsize=12)
    ax1.set_title('准确率分布对比', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. 箱线图
    ax2 = axes[0, 1]
    ax2.boxplot([old_accuracies, new_accuracies],
                labels=['原方案（拼接）', '新方案（按人划分）'],
                patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    ax2.set_ylabel('准确率', fontsize=12)
    ax2.set_title('准确率箱线图对比', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. 累积分布函数
    ax3 = axes[1, 0]
    old_sorted = np.sort(old_accuracies)
    new_sorted = np.sort(new_accuracies)
    old_cdf = np.arange(1, len(old_sorted) + 1) / len(old_sorted)
    new_cdf = np.arange(1, len(new_sorted) + 1) / len(new_sorted)
    ax3.plot(old_sorted, old_cdf, label='原方案（拼接）', color='red', linewidth=2)
    ax3.plot(new_sorted, new_cdf, label='新方案（按人划分）', color='green', linewidth=2)
    ax3.set_xlabel('准确率', fontsize=12)
    ax3.set_ylabel('累积概率', fontsize=12)
    ax3.set_title('累积分布函数对比', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 4. 统计对比柱状图
    ax4 = axes[1, 1]
    x = np.arange(len(metrics))
    width = 0.35
    old_values = [old_stats.get(m, 0) for m in metrics]
    new_values = [new_stats.get(m, 0) for m in metrics]

    bars1 = ax4.bar(x - width/2, old_values, width, label='原方案（拼接）', color='red', alpha=0.7)
    bars2 = ax4.bar(x + width/2, new_values, width, label='新方案（按人划分）', color='green', alpha=0.7)

    ax4.set_xlabel('统计指标', fontsize=12)
    ax4.set_ylabel('准确率', fontsize=12)
    ax4.set_title('统计指标对比', fontsize=14)
    ax4.set_xticks(x)
    ax4.set_xticklabels(metric_names, rotation=15, ha='right')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_file = f"mmfi_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"对比图已保存: {output_file}")
    print()

    # 保存对比报告
    report_file = f"mmfi_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("MMFI方法对比报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("原方案（拼接）:\n")
        f.write(f"  - 数据集: {old_result.get('dataset', 'N/A')}\n")
        f.write(f"  - 运行次数: {old_result.get('total_runs', 0)}\n")
        f.write(f"  - 成功次数: {old_result.get('successful_runs', 0)}\n\n")

        f.write("新方案（按人划分）:\n")
        f.write(f"  - 数据集: {new_result.get('dataset', 'N/A')}\n")
        f.write(f"  - 运行次数: {new_result.get('total_runs', 0)}\n")
        f.write(f"  - 成功次数: {new_result.get('successful_runs', 0)}\n\n")

        f.write("整体统计对比:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'指标':<20} {'原方案':<15} {'新方案':<15} {'提升':<20}\n")
        f.write("-" * 80 + "\n")

        for metric, name in zip(metrics, metric_names):
            old_val = old_stats.get(metric, 0)
            new_val = new_stats.get(metric, 0)
            improvement = new_val - old_val
            improvement_pct = (improvement / old_val * 100) if old_val > 0 else 0
            f.write(f"{name:<20} {old_val:<15.4f} {new_val:<15.4f} {improvement:>+.4f} ({improvement_pct:>+.1f}%)\n")

        f.write("\nTop5对比:\n")
        f.write(f"  原方案Top5均值: {old_top5_mean:.4f}\n")
        f.write(f"  新方案Top5均值: {new_top5_mean:.4f}\n")
        f.write(f"  提升: {top5_improvement:+.4f} ({top5_improvement_pct:+.1f}%)\n\n")

        f.write("结论:\n")
        if top5_improvement > 0:
            f.write(f"  ✓ 新方案（按人划分）相比原方案提升了 {top5_improvement_pct:.1f}%\n")
            f.write(f"  ✓ 证明了按人划分和避免错误拼接的重要性\n")
        else:
            f.write(f"  需要进一步分析\n")

    print(f"对比报告已保存: {report_file}")
    print()

    # 总结
    print("=" * 80)
    print("总结")
    print("=" * 80)
    if top5_improvement > 0:
        print(f"✓ 新方案Top5平均准确率提升: {top5_improvement_pct:.1f}%")
        print(f"✓ 从 {old_top5_mean:.4f} 提升到 {new_top5_mean:.4f}")
        print(f"✓ 证明了按人划分方法的有效性")
    else:
        print("需要进一步分析结果")
    print()


def display_single_result(name, result):
    """显示单个结果"""
    print(f"=" * 80)
    print(f"{name} 结果")
    print(f"=" * 80)
    print()

    stats = result.get('statistics', {})
    top5 = result.get('top5', {})

    print(f"运行次数: {result.get('total_runs', 0)}")
    print(f"成功次数: {result.get('successful_runs', 0)}")
    print()

    print("统计:")
    print(f"  平均准确率: {stats.get('mean', 0):.4f}")
    print(f"  标准差: {stats.get('std', 0):.4f}")
    print(f"  最大准确率: {stats.get('max', 0):.4f}")
    print(f"  最小准确率: {stats.get('min', 0):.4f}")
    print()

    print("Top5:")
    print(f"  均值: {top5.get('mean', 0):.4f}")
    print(f"  种子: {top5.get('seeds', [])[:5]}")
    print()


if __name__ == "__main__":
    compare_results()
