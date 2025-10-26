import argparse
import yaml
from pathlib import Path

def get_parser():
    # >>> 参数解析器 <<<
    parser = argparse.ArgumentParser(description="Continual Learning with Language Guidance")

    # 我们只保留一个参数：配置文件的路径
    parser.add_argument('--config_path', type=str, default='config.yaml', help='Path to the YAML configuration file.')
    
    args = parser.parse_args()

    # --- 从YAML文件加载配置 ---
    # 原理：打开指定的yaml文件，将其内容读入一个字典中。
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    # --- 将字典转换为argparse的Namespace对象 ---
    # 原理：这样做是为了让代码的其他部分可以像以前一样使用 args.batch_size 这样的形式来访问参数，
    # 无需大的改动。我们创建一个空的Namespace，然后把config字典里的键值对逐个添加进去。
    for key, value in config.items():
        setattr(args, key, value)

    return args
