"""
详细可视化STFT处理的各个步骤
展示从CSI原始数据到最终谱图的转换过程
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
import sys
sys.path.append('.')

# 加载MMFI数据
print("加载MMFI数据...")
data_path = "data/mmfi/mmfi_processed_a01.npy"
data = np.load(data_path)
sample = data[0]  # (3, 114, 10)
print(f"样本形状: {sample.shape}")

# 创建大图展示整个流程
fig = plt.figure(figsize=(20, 12))

# 1. 原始CSI数据（选择第一个通道和几个子载波）
ax1 = plt.subplot(3, 4, 1)
selected_subcarriers = [0, 30, 60, 90]
for sc in selected_subcarriers:
    ax1.plot(sample[0, sc, :], label=f'Subcarrier {sc}', marker='o')
ax1.set_title('原始CSI时间序列\n(通道0, 4个子载波)', fontsize=10)
ax1.set_xlabel('时间步')
ax1.set_ylabel('幅度')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# 2. 所有子载波的热图
ax2 = plt.subplot(3, 4, 2)
im = ax2.imshow(sample[0, :, :], aspect='auto', cmap='viridis')
ax2.set_title('所有子载波的时序数据\n(通道0, 114个子载波)', fontsize=10)
ax2.set_xlabel('时间步')
ax2.set_ylabel('子载波')
plt.colorbar(im, ax=ax2)

# 3. 单个子载波的STFT谱图
ax3 = plt.subplot(3, 4, 3)
time_series = sample[0, 30, :]  # 选择第30个子载波
f, t, Zxx = signal.stft(time_series, nperseg=8, noverlap=4, nfft=16)
magnitude = np.abs(Zxx)
magnitude_log = np.log1p(magnitude)
im = ax3.pcolormesh(t, f, magnitude_log, shading='gouraud', cmap='viridis')
ax3.set_title('单个子载波的STFT谱图\n(子载波30, 对数尺度)', fontsize=10)
ax3.set_ylabel('频率')
ax3.set_xlabel('时间')
plt.colorbar(im, ax=ax3)

# 4. 频率分布直方图
ax4 = plt.subplot(3, 4, 4)
ax4.hist(magnitude_log.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
ax4.set_title('STFT谱图值分布\n(对数尺度)', fontsize=10)
ax4.set_xlabel('对数幅度值')
ax4.set_ylabel('频数')
ax4.grid(True, alpha=0.3)

# 5-7. 三个通道的STFT处理结果
for ch in range(3):
    ax = plt.subplot(3, 4, 5 + ch)

    # 对所有子载波进行STFT
    all_spectrograms = []
    for sc in range(sample.shape[1]):
        f, t, Zxx = signal.stft(sample[ch, sc, :], nperseg=8, noverlap=4, nfft=16)
        magnitude = np.abs(Zxx)
        magnitude_log = np.log1p(magnitude)
        all_spectrograms.append(magnitude_log)

    # 堆叠成2D图像
    all_spectrograms = np.array(all_spectrograms)  # (114, freq_bins, time_bins)
    freq_bins, time_bins = all_spectrograms.shape[1], all_spectrograms.shape[2]
    img_2d = all_spectrograms.reshape(sample.shape[1], freq_bins * time_bins)  # (114, freq*time)

    im = ax.imshow(img_2d, aspect='auto', cmap='viridis')
    ax.set_title(f'通道{ch}的完整STFT谱图\n(114×{freq_bins*time_bins})', fontsize=10)
    ax.set_xlabel('频率×时间')
    ax.set_ylabel('子载波')
    plt.colorbar(im, ax=ax)

# 8. 原始方法的结果
ax8 = plt.subplot(3, 4, 8)
# 原始方法：直接转置
img_old = np.transpose(sample, (1, 2, 0))  # (114, 10, 3)
# 只显示第一个通道
im = ax8.imshow(img_old[:, :, 0], aspect='auto', cmap='viridis')
ax8.set_title('原始粗糙方法\n(直接转置, 通道0)', fontsize=10)
ax8.set_xlabel('时间步 (10)')
ax8.set_ylabel('子载波 (114)')
plt.colorbar(im, ax=ax8)

# 9-11. 三个通道组合成RGB图像
ax9 = plt.subplot(3, 4, 9)
all_channels = []
for ch in range(3):
    all_spectrograms = []
    for sc in range(sample.shape[1]):
        f, t, Zxx = signal.stft(sample[ch, sc, :], nperseg=8, noverlap=4, nfft=16)
        magnitude = np.abs(Zxx)
        magnitude_log = np.log1p(magnitude)
        all_spectrograms.append(magnitude_log)
    all_spectrograms = np.array(all_spectrograms)
    freq_bins, time_bins = all_spectrograms.shape[1], all_spectrograms.shape[2]
    img_2d = all_spectrograms.reshape(sample.shape[1], freq_bins * time_bins)
    all_channels.append(img_2d)

# 堆叠为RGB
rgb_image = np.stack(all_channels, axis=-1)  # (114, freq*time, 3)
# 归一化到0-1
rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
ax9.imshow(rgb_image)
ax9.set_title(f'STFT方法RGB组合\n({rgb_image.shape[0]}×{rgb_image.shape[1]}×3)', fontsize=10)
ax9.set_xlabel('频率×时间')
ax9.set_ylabel('子载波')
ax9.axis('off')

# 10. 原始方法RGB
ax10 = plt.subplot(3, 4, 10)
img_old_norm = (img_old - img_old.min()) / (img_old.max() - img_old.min())
ax10.imshow(img_old_norm)
ax10.set_title(f'原始方法RGB\n({img_old.shape[0]}×{img_old.shape[1]}×3)', fontsize=10)
ax10.set_xlabel('时间步')
ax10.set_ylabel('子载波')
ax10.axis('off')

# 11. 统计对比
ax11 = plt.subplot(3, 4, 11)
stats_data = [
    ['方法', '高度', '宽度', '信息量'],
    ['STFT', rgb_image.shape[0], rgb_image.shape[1], f'{rgb_image.shape[0]*rgb_image.shape[1]}'],
    ['原始', img_old.shape[0], img_old.shape[1], f'{img_old.shape[0]*img_old.shape[1]}']
]
table = ax11.table(cellText=stats_data, cellLoc='center', loc='center',
                   colWidths=[0.25, 0.25, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)
# 设置表头样式
for i in range(4):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')
ax11.axis('off')
ax11.set_title('尺寸对比', fontsize=10, pad=20)

# 12. 处理流程图（文本说明）
ax12 = plt.subplot(3, 4, 12)
ax12.axis('off')
flow_text = """
STFT处理流程：

1. 输入: (3, 114, 10)
   ├─ 3个通道
   ├─ 114个子载波
   └─ 10个时间步

2. 对每个子载波做STFT:
   ├─ 窗口长度: 8
   ├─ 重叠: 4
   └─ FFT点数: 16

3. 生成谱图:
   ├─ 频率bins: 9
   ├─ 时间bins: 3
   └─ 总尺寸: 27

4. 重塑为2D:
   (114, 27, 3) RGB图像

5. Resize: (224, 224, 3)
"""
ax12.text(0.1, 0.9, flow_text, transform=ax12.transAxes,
         fontsize=9, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle('MMFI CSI数据的STFT时频变换详细过程', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('visualize_stft_details.png', dpi=150, bbox_inches='tight')
print("\n可视化完成！")
print("详细图表已保存到: visualize_stft_details.png")

# 打印详细信息
print("\n" + "="*60)
print("处理细节:")
print("="*60)
print(f"原始CSI数据形状: {sample.shape}")
print(f"  - 通道数: {sample.shape[0]}")
print(f"  - 子载波数: {sample.shape[1]}")
print(f"  - 时间步数: {sample.shape[2]}")
print(f"\nSTFT参数:")
print(f"  - 窗口长度(nperseg): 8")
print(f"  - 重叠长度(noverlap): 4")
print(f"  - FFT点数(nfft): 16")
print(f"\nSTFT输出:")
print(f"  - 频率bins: {freq_bins}")
print(f"  - 时间bins: {time_bins}")
print(f"  - 单个谱图尺寸: ({freq_bins}, {time_bins})")
print(f"\n2D图像尺寸:")
print(f"  - STFT方法: ({rgb_image.shape[0]}, {rgb_image.shape[1]}, 3)")
print(f"  - 原始方法: ({img_old.shape[0]}, {img_old.shape[1]}, 3)")
print(f"\n信息量对比:")
print(f"  - STFT方法: {rgb_image.shape[0] * rgb_image.shape[1]} = {114 * 27} 像素")
print(f"  - 原始方法: {img_old.shape[0] * img_old.shape[1]} = {114 * 10} 像素")
print(f"  - STFT提供了 {(114*27)/(114*10):.1f}x 更多的空间信息")
print("="*60)
