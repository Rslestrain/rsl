import json

def load_jsonl(file_path):
    """读取 JSON Lines 文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def get_ground_truth(data_item):
    """从原始数据项中提取标准答案"""
    for message in data_item.get('data', []):
        if message.get('role') == 'assistant':
            # 找到第一个 assistant 的回复作为标准答案
            return message.get('content')
    return None

def main():
    # --- 文件路径 ---
    results_file = 'results.json'
    ground_truth_file = 'data/单轮-冒烟测试集.jsonl'

    print(f"Loading predictions from: {results_file}")
    print(f"Loading ground truth from: {ground_truth_file}")
    
    # 1. 加载你的模型预测结果
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            predictions_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Prediction file '{results_file}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{results_file}'. It might be empty or corrupted.")
        return

    # 2. 加载标准答案数据集
    # 注意：在 results.json 中，我们已经保存了原始输入，所以不需要再读 ground_truth_file
    # ground_truth_data = load_jsonl(ground_truth_file) 
    # 这里我们直接从 predictions_data 中提取原始输入，更方便对齐

    correct_count = 0
    total_count = len(predictions_data)
    
    mismatched_predictions = []

    # 3. 逐条对比
    for item in predictions_data:
        prediction = item.get('output', '').strip()
        original_input = item.get('input')
        
        if not original_input:
            print("Warning: Found a result item without original input data. Skipping.")
            total_count -=1
            continue

        ground_truth = get_ground_truth(original_input)
        
        if ground_truth is None:
            print(f"Warning: Could not find ground truth for input ID {original_input.get('id')}. Skipping.")
            total_count -=1
            continue

        ground_truth = ground_truth.strip()

        if prediction == ground_truth:
            correct_count += 1
        else:
            # 记录错误的例子，方便分析
            mismatched_predictions.append({
                "id": original_input.get('id'),
                "user_query": original_input['data'][0]['content'],
                "prediction": prediction,
                "ground_truth": ground_truth
            })

    # 4. 计算并打印结果
    if total_count == 0:
        print("No valid predictions to evaluate.")
        return
        
    accuracy = (correct_count / total_count) * 100

    print("\n--- Evaluation Report ---")
    print(f"Total Samples:    {total_count}")
    print(f"Correct Samples:  {correct_count}")
    print(f"Incorrect Samples: {total_count - correct_count}")
    print(f"Accuracy:         {accuracy:.2f}%")
    print("-------------------------\n")
    
    # 5. (可选但强烈推荐) 打印出错误的例子进行分析
    if mismatched_predictions:
        print("--- Mismatched Predictions ---")
        for i, mismatch in enumerate(mismatched_predictions[:10]): # 只打印前10个
            print(f"Example {i+1}:")
            print(f"  Query:        {mismatch['user_query']}")
            print(f"  Prediction:   {mismatch['prediction']}")
            print(f"  Ground Truth: {mismatch['ground_truth']}")
            print("-" * 20)
        
        # 将所有错误保存到文件
        with open('mismatched_results.json', 'w', encoding='utf-8') as f:
            json.dump(mismatched_predictions, f, ensure_ascii=False, indent=2)
        print("\nAll mismatched predictions have been saved to 'mismatched_results.json' for detailed analysis.")


if __name__ == '__main__':
    main()