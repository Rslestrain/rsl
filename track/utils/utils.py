import os
import copy
import random
import numpy as np
from collections import OrderedDict as OD
from collections import defaultdict as DD
from collections.abc import Iterable
import logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

def set_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

def get_logger(path):
    now = datetime.now()
    year = now.year
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute
    second = now.second
    formatted_date_time = f"{year:04d}-{month:02d}-{day:02d}-{hour:02d}-{minute:02d}-{second:02d}.log"
    log_path_name = path+formatted_date_time
    return set_logger(log_path_name)

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

# >>> Optimizer Stuff <<<
def set_optimizer(args, named_parameters):
    """
    设置优化器，支持为可学习提示词设置不同的学习率。
    
    Args:
        args: 配置参数
        named_parameters: 模型.named_parameters() 的输出
    """
    
    # 分离主网络参数和提示词参数
    main_params = []
    prompt_params = []
    
    # 正确地遍历 (name, param) 元组
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
            
        # 检查参数名是否包含 'prompt' 关键字
        if 'prompt_distiller' in name or 'prompt_embeddings' in name:
            prompt_params.append(param)
        else:
            main_params.append(param)

    # 检查是否有 prompt_lr 配置
    prompt_lr = getattr(args, 'prompt_lr', args.lr)
    
    print(f"Optimizer: Found {len(prompt_params)} learnable prompt parameter groups.")
    print(f"Optimizer: Found {len(main_params)} main network parameter groups.")

    # 使用不同的学习率
    optimizer = torch.optim.Adam([
        {'params': main_params, 'lr': args.lr, 'weight_decay': 0.0001},
        {'params': prompt_params, 'lr': prompt_lr, 'weight_decay': 0.0} # 提示词通常不需要权重衰减
    ])
    
    return optimizer

# Save hyperparameters to log files
def log_hyperparameters(args, logger):  
    logger.info("Hyperparameters:")
    for key, value in vars(args).items():
        logger.info(f"{key}: {value}")

# Print model structure and parameter information
def print_model_parameters(model, mylogger):
    param_info = [(name, param.size(), param.numel())
                  for name, param in model.named_parameters()]
    param_info.sort(key=lambda x: x[2], reverse=True)  # Sort by number of parameters
    mylogger.info("Model's state_dict:")
    for name, size, num_params in param_info:
        mylogger.info(
            f"Layer: {name} | Size: {size} | Total Parameters: {num_params}")
    mylogger.info(
        f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")

def log_gpu_memory_usage(device, mylogger, step=""):
    allocated = torch.cuda.memory_allocated(
        device=device) / (1024 * 1024)  
    reserved = torch.cuda.memory_reserved(
        device=device) / (1024 * 1024)  
    max_allocated = torch.cuda.max_memory_allocated(
        device=device) / (1024 * 1024)  
    max_reserved = torch.cuda.max_memory_reserved(
        device=device) / (1024 * 1024)  
    mylogger.info(f"{step} GPU Memory Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB, Max Allocated: {max_allocated:.2f} MB, Max Reserved: {max_reserved:.2f} MB")