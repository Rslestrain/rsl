#!/usr/bin/env python3
"""
自动监控Wiar训练完成后启动MMFI训练
"""
import os
import subprocess
import time
from datetime import datetime

WIAR_PID = 2777717
MMFI_SCRIPT = "batch_train_mmfi_stft.py"
LOG_DIR = "batch_results"
WIAR_LOG = f"{LOG_DIR}/wiar_stft_100runs_20251121_162600.log"

print("=" * 60)
print("自动启动脚本 - 等待Wiar完成后启动MMFI")
print("=" * 60)
print(f"Wiar PID: {WIAR_PID}")
print(f"启动时间: {datetime.now()}")
print()

# 检查Wiar进程是否存在
def is_process_running(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

# 获取Wiar进度
def get_wiar_progress():
    try:
        with open(WIAR_LOG, 'r') as f:
            content = f.read()
            return content.count("Accuracy:")
    except:
        return 0

if not is_process_running(WIAR_PID):
    print("⚠ Wiar进程不存在，立即启动MMFI训练")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{LOG_DIR}/mmfi_stft_100runs_{timestamp}.log"

    cmd = f"/data1/rsl/anaconda3/envs/consense/bin/python {MMFI_SCRIPT}"
    with open(log_file, 'w') as f:
        proc = subprocess.Popen(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)

    print(f"✓ MMFI训练已启动 (PID: {proc.pid})")
    print(f"日志文件: {log_file}")
    exit(0)

# 等待Wiar完成
print("等待Wiar训练完成...")
last_progress = 0

while is_process_running(WIAR_PID):
    progress = get_wiar_progress()
    if progress != last_progress:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Wiar进度: {progress}/100")
        last_progress = progress

    time.sleep(60)  # 每分钟检查一次

print()
print("✓ Wiar训练已完成！")
print()

# 等待GPU内存释放
print("等待GPU内存释放（30秒）...")
time.sleep(30)

# 启动MMFI训练
print("启动MMFI训练...")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"{LOG_DIR}/mmfi_stft_100runs_{timestamp}.log"

cmd = f"/data1/rsl/anaconda3/envs/consense/bin/python {MMFI_SCRIPT}"
with open(log_file, 'w') as f:
    proc = subprocess.Popen(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)

print()
print("=" * 60)
print("✓ MMFI训练已启动")
print("=" * 60)
print(f"PID: {proc.pid}")
print(f"启动时间: {datetime.now()}")
print(f"日志文件: {log_file}")
print()
print("监控命令:")
print(f"  tail -f {log_file}")
print(f"  ps aux | grep {proc.pid}")
print("=" * 60)
