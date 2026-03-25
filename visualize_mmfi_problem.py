import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
import matplotlib.pyplot as plt

# 加载MMFI数据
file_path = "data/mmfi/mmfi_processed_a01.npy"
data = np.load(file_path)

print(f"加载文件: {file_path}")
print(f"数据形状: {data.shape}")
print()

# 取前20个样本进行可视化
num_samples = 20
samples = data[:num_samples]

# 创建图表
fig, axes = plt.subplots(4, 5, figsize=(20, 16))
fig.suptitle('MMFI数据：前20个样本的可视化\n如果是连续帧，应该看到平滑的变化；如果是独立样本，会看到跳跃变化',
             fontsize=16, fontweight='bold')

for idx in range(num_samples):
    row = idx // 5
    col = idx % 5
    ax = axes[row, col]

    # 可视化: 取第一个channel的数据 (114 x 10)
    sample_data = samples[idx, 0, :, :]  # shape: (114, 10)

    im = ax.imshow(sample_data, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_title(f'样本 {idx}', fontsize=10)
    ax.set_xlabel('Time')
    ax.set_ylabel('Subcarrier')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('mmfi_samples_visualization.png', dpi=150, bbox_inches='tight')
print("图表1保存至: mmfi_samples_visualization.png")
print()

# 创建时间序列对比图
fig2, axes2 = plt.subplots(3, 1, figsize=(16, 12))
fig2.suptitle('MMFI数据连续性分析：如果是连续帧，曲线应该平滑过渡', fontsize=14, fontweight='bold')

# 选择一个subcarrier来跟踪
subcarrier_idx = 57  # 中间的一个subcarrier
time_steps = 10

# 绘制连续20个样本在特定subcarrier上的时序变化
for sample_idx in range(20):
    sample = samples[sample_idx, 0, subcarrier_idx, :]  # 10个时间步

    # 为了在一张图上显示，给每个样本分配一个时间段
    x_offset = sample_idx * time_steps
    x_positions = np.arange(x_offset, x_offset + time_steps)

    axes2[0].plot(x_positions, sample, marker='o', markersize=3, linewidth=1, alpha=0.7)

    # 在样本边界画竖线
    if sample_idx > 0:
        axes2[0].axvline(x=x_offset, color='red', linestyle='--', alpha=0.3, linewidth=1)

axes2[0].set_xlabel('时间索引 (每10个点=1个样本)', fontsize=11)
axes2[0].set_ylabel('CSI幅值', fontsize=11)
axes2[0].set_title(f'连续20个样本的时序变化 (Subcarrier {subcarrier_idx}, Channel 0)\n红色虚线=样本边界。如果是连续帧，边界处应该平滑过渡',
                  fontsize=12)
axes2[0].grid(True, alpha=0.3)

# 计算样本间差异
consecutive_diffs = []
random_diffs = []

for i in range(100):
    # 连续样本差异
    diff_consecutive = np.linalg.norm(data[i] - data[i+1])
    consecutive_diffs.append(diff_consecutive)

    # 随机样本差异
    random_idx = np.random.randint(0, 1000)
    diff_random = np.linalg.norm(data[i] - data[random_idx])
    random_diffs.append(diff_random)

axes2[1].plot(consecutive_diffs, label='连续样本间差异', color='blue', linewidth=2)
axes2[1].plot(random_diffs, label='随机样本间差异', color='red', linewidth=2, alpha=0.7)
axes2[1].axhline(y=np.mean(consecutive_diffs), color='blue', linestyle='--',
                label=f'连续样本平均: {np.mean(consecutive_diffs):.2f}')
axes2[1].axhline(y=np.mean(random_diffs), color='red', linestyle='--',
                label=f'随机样本平均: {np.mean(random_diffs):.2f}')
axes2[1].set_xlabel('样本索引', fontsize=11)
axes2[1].set_ylabel('欧氏距离', fontsize=11)
axes2[1].set_title('样本间差异对比：如果是连续帧，蓝线应该远低于红线', fontsize=12)
axes2[1].legend(fontsize=10)
axes2[1].grid(True, alpha=0.3)

# 分布直方图
axes2[2].hist(consecutive_diffs, bins=30, alpha=0.6, label='连续样本差异', color='blue', edgecolor='black')
axes2[2].hist(random_diffs, bins=30, alpha=0.6, label='随机样本差异', color='red', edgecolor='black')
axes2[2].axvline(x=np.mean(consecutive_diffs), color='blue', linestyle='--', linewidth=2,
                label=f'连续样本均值: {np.mean(consecutive_diffs):.2f}')
axes2[2].axvline(x=np.mean(random_diffs), color='red', linestyle='--', linewidth=2,
                label=f'随机样本均值: {np.mean(random_diffs):.2f}')
axes2[2].set_xlabel('欧氏距离', fontsize=11)
axes2[2].set_ylabel('频数', fontsize=11)
axes2[2].set_title('差异分布：两个分布高度重叠说明样本是独立的、非连续的', fontsize=12)
axes2[2].legend(fontsize=10)
axes2[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('mmfi_continuity_analysis.png', dpi=150, bbox_inches='tight')
print("图表2保存至: mmfi_continuity_analysis.png")
print()

# 打印结论
print("=" * 80)
print("分析结论:")
print("=" * 80)
ratio = np.mean(consecutive_diffs) / np.mean(random_diffs)
print(f"连续样本差异均值: {np.mean(consecutive_diffs):.4f}")
print(f"随机样本差异均值: {np.mean(random_diffs):.4f}")
print(f"比值: {ratio:.4f}")
print()

if ratio > 0.8:
    print("✗ 结论: MMFI样本是独立的、被shuffle过的，不是连续时间帧")
    print("  → 时域拼接是错误的操作！")
    print()
    print("原因:")
    print("  1. 连续样本间差异 ≈ 随机样本间差异")
    print("  2. 样本很可能来自不同的人/不同的执行")
    print("  3. 拼接会创造出不存在的'虚假'长时序数据")
    print()
    print("建议:")
    print("  1. 不要使用时域拼接")
    print("  2. 直接用单个样本 (T=10) 进行STFT")
    print("  3. 或者调整STFT参数以适应短时序")
else:
    print("✓ 样本可能是连续的")
    print("  → 可以考虑时域拼接")
