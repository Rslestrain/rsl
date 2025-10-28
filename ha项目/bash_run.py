import json
from tqdm import tqdm

from agent import CustomAgent
from demo_agent import DirectAgent, HierarchicalAgent
from agent import CustomAgent

def pre_input(json_data):
    """
    提取在第一个助手（assistant）回合之前的所有对话历史。
    这个修改后的版本更加健壮，避免了因为判断条件过于具体而返回空列表的问题。
    """
    input_turns = []
    for turn in json_data["data"]:
        if turn["role"] == "assistant":
            # 只要遇到 assistant 的角色，就停止，意味着之前的所有内容都是输入
            break
        input_turns.append(turn)
    return input_turns


def load_data(data_path):
    if data_path.endswith(".jsonl"):
        data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        return data
    elif data_path.endswith(".json"):
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    else:
        raise ValueError("Only json and jsonl files are supported")


if __name__ == "__main__":
    data = load_data("data/单轮-冒烟测试集.jsonl")
    # data = load_data("data/多轮-冒烟测试集.jsonl")
    #agent = DirectAgent()
    # agent = HierarchicalAgent()
    agent = CustomAgent()

    results = []
    for item in tqdm(data):
        # 准备输入，并增加一个安全检查，防止空输入导致程序崩溃
        processed_input = pre_input(item)
        if not processed_input:
            print(f"警告：跳过一个没有有效用户输入的样本。样本 ID: {item.get('id', 'N/A')}")
            results.append({"input": item, "output": "SKIPPED_EMPTY_INPUT"})
            continue

        response = agent.run(processed_input)
        results.append({"input": item, "output": response})
        # print("Input:", item)
        # print("Output:", response)
        # print("-" * 50)
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)