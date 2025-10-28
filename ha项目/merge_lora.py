import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

print("--- Starting LoRA merge script ---")

# --- 路径配置 ---
base_model_path = "models/Qwen3-4B-Instruct-2507"
lora_adapter_path = "./lora_final_adapter"
merged_model_path = "./models/Qwen3-4B-Instruct-2507-merged" # 新模型的保存路径

print(f"Loading base model from: {base_model_path}")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

print(f"Loading LoRA adapter from: {lora_adapter_path}")
lora_model = PeftModel.from_pretrained(base_model, lora_adapter_path)

print("Merging LoRA weights into the base model...")
# 使用PEFT库的内置功能进行合并
merged_model = lora_model.merge_and_unload()
print("Merge complete.")

# --- 保存全新的、合并后的模型 ---
print(f"Saving merged model to: {merged_model_path}")
os.makedirs(merged_model_path, exist_ok=True)
merged_model.save_pretrained(merged_model_path)

# --- 保存分词器到新目录 ---
print(f"Saving tokenizer to: {merged_model_path}")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.save_pretrained(merged_model_path)

print("--- Merge script finished successfully! ---")
print(f"Your new, fine-tuned model is ready at: {merged_model_path}")