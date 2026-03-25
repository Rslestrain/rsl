#!/usr/bin/env python3
"""
重新生成MMFI STFT数据集（使用时域拼接优化）
"""
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def regenerate_mmfi_stft():
    """重新生成MMFI STFT数据集"""
    print("="*70)
    print("重新生成MMFI STFT数据集（时域拼接优化）")
    print("="*70)
    print()

    # 创建配置参数
    class Args:
        pass

    args = Args()

    # 导入数据类
    from utils.data import iMMFIDataSTFT

    # 创建数据实例
    print("初始化MMFI STFT数据加载器...")
    data_manager = iMMFIDataSTFT(args)

    # 触发数据生成
    print("\n开始生成数据...\n")
    data_manager.download_data()

    print("\n" + "="*70)
    print("✓ MMFI STFT数据生成完成！")
    print("="*70)
    print(f"训练集: {len(data_manager.train_data)} 个样本")
    print(f"测试集: {len(data_manager.test_data)} 个样本")
    print()
    print("优化效果:")
    print("  - 时间步: T=10 → T=50 (5x拼接)")
    print("  - STFT参数: nperseg=16, noverlap=8, nfft=32")
    print("  - 时间分辨率: 4 → 8 bins (2.0x)")
    print("  - 频率分辨率: 9 → 17 bins (1.9x)")
    print("  - 生成图像: 114×36 → 114×136 (3.8x像素)")
    print()
    print("下一步: 运行训练测试新数据集效果")
    print("  python main.py --config configs/mmfi_short_stft.json")

if __name__ == "__main__":
    try:
        regenerate_mmfi_stft()
    except KeyboardInterrupt:
        print("\n\n用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
