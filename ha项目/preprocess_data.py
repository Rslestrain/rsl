import json
def load_jsonl(file_path):
    """读取 JSON Lines 文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data
# 1. 加载基础Prompt模板
with open('base_system_prompt.txt', 'r', encoding='utf-8') as f:
    prompt_template = f.read()

# 2. 加载并格式化工具信息 (和之前的方案一样)
with open('prompts/tools_v0.json', 'r', encoding='utf-8') as f:
    tools = json.load(f)
# ... (此处省略将 tools 转换成 tools_info 字符串的代码) ...
tools_info = "" # 你已经知道怎么生成这个了
# --- 替换开始 ---
tool_info_list = []
for tool in tools:
    # 检查是否存在 parameters 键，并且它不是 None
    params = tool.get("parameters")
    if params: # 只有当 params 不为 None 且不为空列表时才执行
        param_str = "\n".join(
            [
                # 假设 param 字典里一定有 'name' 和 'description'
                f"  - {param.get('name', 'N/A')}: {param.get('description', 'N/A')}"
                for param in params
            ]
        )
    else: # 如果没有参数
        param_str = "  - 无"
        
    t_info = (
        f"函数名: {tool['name']}\n"
        f"描述: {tool['description']}\n"
        f"参数: \n"
        f"{param_str}"
    )
    tool_info_list.append(t_info)

tools_info = "\n\n".join(tool_info_list)
# --- 替换结束 ---
print("--- [1/5] Preprocessing script started ---")
# 3. 加载原始数据 (指令-答案对)
# 假设你已经有了一个 all_samples.jsonl, 每一行是 {"query": "...", "answer": "..."}
raw_data = load_jsonl('all_samples.jsonl') 

# 4. 生成最终的训练文件
with open('finetune_data.jsonl', 'w', encoding='utf-8') as f:
    for item in raw_data:
        user_query = item.get('query')
        ground_truth_call = item.get('answer')

        if not user_query or not ground_truth_call:
            print(f"Warning: Skipping invalid item: {item}")
            continue # <-- 如果条件成立，就会跳过写入步骤！
        
        # 填充Prompt模板
        filled_prompt = prompt_template.format(
            tools_info=tools_info,
            user_query=user_query
        )
        
        # 构造最终的训练文本
        # 格式：[完整的输入提示] + [期望的输出]
        # 注意：需要加上模型的特殊token来分隔prompt和completion，这里以<|endoftext|>为例
        training_text = f"{filled_prompt}<tool>{ground_truth_call}</tool>"
        
        f.write(json.dumps({"text": training_text}) + '\n')
