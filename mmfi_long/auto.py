import os
import subprocess

# 定义参数
abc = 5.0
fsda = 65

# 循环执行 seed 从 0 到 100
for seed in range(1001):
    # 构造命令
    command = f"/data1/smy1/.conda/envs/py38/bin/python /data1/smy1/lr/work2_compress/consense_mmfi_onehead_fixclass/main.py --n_tasks=9 --batch_size=16 --n_warmup_epochs=0 --n_epochs=20 --half_iid=1 --seed={seed}"
    
    # 打印当前正在执行的命令（可以根据需要删除）
    print(f"Executing command with seed={seed}")
    
    # 执行命令
    subprocess.run(command, shell=True)
