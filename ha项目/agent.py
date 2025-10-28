import re
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

class CustomAgent:
    def __init__(self):
        print("--- Initializing Merged Model Agent ---")

        # **关键修改**: 直接指向合并后的新模型路径
        merged_model_path = "./models/Qwen3-4B-Instruct-2507-merged"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        print(f"Loading merged model and tokenizer from: {merged_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(merged_model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            merged_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device) # 直接加载并移动到GPU
        
        self.model.eval()

        print("Preparing prompt template and tools info...")
        # ... (这部分代码和之前完全一样，保持不变) ...
        try:
            with open('base_system_prompt.txt', 'r', encoding='utf-8') as f:
                self.prompt_template = f.read()
        except FileNotFoundError:
            raise FileNotFoundError("Error: 'base_system_prompt.txt' not found.")
        with open('prompts/tools_v0.json', 'r', encoding='utf-8') as f:
            tools = json.load(f)
        tool_info_list = []
        for tool in tools:
            param_signature = ", ".join([f"{p['name']}" for p in tool.get("parameters", [])])
            func_signature = f"{tool['name']}({param_signature})"
            param_details = [f"  - {p['name']}: {p.get('description', '')}" for p in tool.get("parameters", [])]
            t_info = (f"函数: {func_signature}\n描述: {tool['description']}\n参数详情:\n" + "\n".join(param_details))
            tool_info_list.append(t_info)
        self.tools_info = "\n\n".join(tool_info_list)

        self.generation_config = GenerationConfig(
            max_new_tokens=128,
            eos_token_id=self.tokenizer.convert_tokens_to_ids("</tool>"),
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False,
            use_cache=True,
        )
        print("--- Agent initialization complete ---")

    def run(self, input_messages) -> str:
        # 这个 run 方法和上一个版本完全一样，不需要修改
        user_query = input_messages[-1]['content']
        prompt = self.prompt_template.format(tools_info=self.tools_info, user_query=user_query)
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**model_inputs, generation_config=self.generation_config)
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        tool_code_match = re.search(r"<tool>(.*?)(?:</tool>|$)", response, re.DOTALL)
        if tool_code_match:
            tool_call = tool_code_match.group(1).strip()
        else:
            # 如果标准匹配失败，尝试从 Markdown 代码块中提取
            # 这个正则表达式会查找 ```tool ... ``` 或 ``` ... ``` 里的内容
            md_match = re.search(r"```(?:tool)?\s*(.*?)\s*```", response, re.DOTALL)
            if md_match:
                # 提取出代码块内容后，再尝试查找类似函数调用的部分
                tool_call_text = md_match.group(1).strip()
                # 查找类似 FunctionName(...) 的模式
                func_match = re.search(r"(\w+\s*\(.*\))", tool_call_text)
                if func_match:
                    tool_call = func_match.group(1)
                else:
                    # 如果代码块里没有括号，就用原始文本
                    tool_call = tool_call_text
            else:
                # 如果两种方法都失败，打印警告并使用原始响应
                print(f"Warning: No <tool> tag or valid markdown block found in model response: '{response}'")
                return response.strip()

        # 对提取出的 tool_call 进行统一的后处理
        if "(" not in tool_call and ")" not in tool_call:
            tool_call += "()"
        
        return tool_call.strip()