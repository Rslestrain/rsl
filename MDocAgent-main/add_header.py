# add_header_enhanced.py
import os

def add_path_comments_to_files(root_dir='.'):
    """
    遍历指定目录下的所有支持的文件，
    并在文件开头添加其相对路径作为注释。

    Args:
        root_dir (str): 要扫描的根目录路径。默认为当前目录。
    """
    # 定义不同文件类型的注释格式
    # 键是文件扩展名，值是注释的格式化字符串
    comment_formats = {
        '.py': '# {}',
        '.js': '/* {} */',
        '.css': '/* {} */',
        '.html': '<!-- {} -->',
        '.vue': '<!-- {} -->',  # .vue 文件顶层最安全的注释方式是 HTML 注释
        '.yaml': '# {}',
        '.yml': '# {}'
    }
    
    supported_extensions = tuple(comment_formats.keys())

    # 获取脚本自身的绝对路径，以便在遍历时排除它
    try:
        script_path = os.path.abspath(__file__)
    except NameError:
        # 在某些交互式环境（如Jupyter）中 __file__ 可能未定义
        script_path = os.path.abspath('add_header_with_yaml.py') # 确保这里是当前脚本的文件名
        
    start_path = os.path.abspath(root_dir)
    print(f"开始扫描目录: {start_path}")
    print(f"支持的文件类型: {', '.join(supported_extensions)}")

    # 遍历目录树
    for dirpath, _, filenames in os.walk(start_path):
        for filename in filenames:
            # 检查文件扩展名是否在支持的列表中
            file_ext = os.path.splitext(filename)[1]
            if file_ext in supported_extensions:
                file_abs_path = os.path.join(dirpath, filename)

                # 跳过脚本文件本身
                if file_abs_path == script_path:
                    continue

                # 计算相对路径，并统一使用斜杠'/'作为分隔符
                relative_path = os.path.relpath(file_abs_path, start_path).replace('\\', '/')
                
                # 根据文件类型获取正确的注释格式
                comment_template = comment_formats[file_ext]
                comment = comment_template.format(relative_path) + '\n'
                
                try:
                    # 使用 'r+' 模式读写文件，并指定 utf-8 编码
                    with open(file_abs_path, 'r+', encoding='utf-8') as f:
                        # 读取第一行，检查注释是否已存在
                        first_line = f.readline()
                        
                        # 如果注释已存在，则跳过
                        # 使用 strip() 来移除换行符等空白字符，以便精确匹配
                        if first_line.strip() == comment_template.format(relative_path):
                            print(f"已跳过 (注释已存在): {relative_path}")
                            continue

                        # 如果注释不存在，则重置文件指针到开头
                        f.seek(0)
                        # 读取文件所有原始内容
                        original_content = f.read()
                        
                        # 再次重置文件指针，写入新注释和原始内容
                        f.seek(0)
                        f.write(comment + original_content)
                        print(f"已添加注释到: {relative_path}")

                except UnicodeDecodeError:
                    print(f"警告: 文件 '{relative_path}' 不是 UTF-8 编码，已跳过。请手动转换编码后再试。")
                except Exception as e:
                    print(f"处理文件 '{relative_path}' 时发生错误: {e}")

if __name__ == "__main__":
    # 将此脚本放置在您的项目根目录并从那里运行。
    # 它将递归地扫描所有子目录。
    add_path_comments_to_files('.')
    print("\n脚本执行完毕。")
