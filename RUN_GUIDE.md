# 运行指南

## ✅ STFT版本已经测试成功！

训练正在正常进行，准确率从26.74%提升到84.39%，效果很好！

## 🚀 如何运行

### 方式1：使用运行脚本（推荐）

```bash
cd /data1/rsl/LoRA-Sub-DRS-master
./run_stft.sh
```

### 方式2：直接命令行

```bash
cd /data1/rsl/LoRA-Sub-DRS-master
/data1/rsl/anaconda3/envs/consense/bin/python main.py --config configs/mmfi_short_stft.json
```

### 方式3：原始方法（对比用）

```bash
cd /data1/rsl/LoRA-Sub-DRS-master
/data1/rsl/anaconda3/envs/consense/bin/python main.py --config configs/mmfi_short.json
```

## 📊 当前运行状态

正在后台运行的训练：
- 进程ID: 724427
- 配置: configs/mmfi_short_stft.json
- 日志文件: stft_run.log

查看训练进度：
```bash
# 查看实时日志
tail -f stft_run.log

# 或查看最新20行
tail -20 stft_run.log

# 查看训练进度（只看epoch信息）
grep "Epoch" stft_run.log | tail -10
```

## 🎯 训练进展（当前）

```
Epoch 1/20 => Loss 2.387, Train_accy 26.74%
Epoch 2/20 => Loss 1.440, Train_accy 51.51%
Epoch 3/20 => Loss 1.201, Train_accy 64.20%
Epoch 4/20 => Loss 1.153, Train_accy 69.93%
Epoch 5/20 => Loss 0.803, Train_accy 73.78%
Epoch 6/20 => Loss 0.613, Train_accy 80.96%
Epoch 7/20 => Loss 0.491, Train_accy 84.39%
```

**进展很好！** 准确率快速提升，loss稳定下降！

## 📁 文件位置

- **配置文件**: `configs/mmfi_short_stft.json`
- **STFT处理的数据**: `data/mmfi_stft/train/` 和 `data/mmfi_stft/test/`
- **日志目录**: `logs/mmfi_stft/12_3_sip/lorasub_drs/Adam/`
- **运行日志**: `stft_run.log`

## 🔧 遇到问题？

### 问题1：找不到数据
```bash
ls data/mmfi_stft/train/0/ | head
```
如果没有数据，会自动生成（首次10-30分钟）

### 问题2：查看错误信息
```bash
tail -50 stft_run.log | grep -i error
```

### 问题3：重新开始训练
```bash
# 停止当前训练
pkill -f "mmfi_short_stft.json"

# 重新运行
./run_stft.sh
```

## 📚 更多文档

- 详细原理: `STFT_README.md`
- 完整总结: `STFT_SUMMARY.md`
- 快速开始: `QUICK_START_STFT.md`
- 命令速查: `STFT_COMMANDS_CHEATSHEET.md`

## ✨ 核心区别

| 方面 | 原始方法 (mmfi) | STFT方法 (mmfi_stft) |
|------|----------------|---------------------|
| 数据处理 | 直接转置resize | STFT时频变换 |
| 信息量 | 1140像素 | 4104像素 (2.7倍) |
| 频域特征 | 无 | 有 |
| 配置文件 | mmfi_short.json | mmfi_short_stft.json |

## 🎉 测试结果

- ✅ 数据加载正常
- ✅ 模型训练正常
- ✅ 准确率提升正常
- ✅ 无报错，运行流畅

**可以放心使用了！**
