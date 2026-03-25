#!/bin/bash
# STFT版本运行脚本
# 使用STFT时频变换处理的MMFI数据进行训练

cd /data1/rsl/LoRA-Sub-DRS-master

# 使用consense环境
PYTHON=/data1/rsl/anaconda3/envs/consense/bin/python

echo "================================"
echo "运行STFT版本的MMFI训练"
echo "================================"
echo "配置文件: configs/mmfi_short_stft.json"
echo "数据集: mmfi_stft (使用STFT时频变换)"
echo "日志目录: logs/mmfi_stft/"
echo "================================"

# 运行训练
$PYTHON main.py --config configs/mmfi_short_stft.json

echo "================================"
echo "训练完成！"
echo "查看日志: tail -f logs/mmfi_stft/12_3_sip/lorasub_drs/Adam/0.log"
echo "================================"
