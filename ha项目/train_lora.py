import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# 1. 模型和分词器路径
model_name = "models/Qwen3-4B-Instruct-2507"

# 2. 加载数据集
dataset = load_dataset("json", data_files="finetune_data.jsonl", split="train")

# 3. LoRA 配置
lora_config = LoraConfig(
    r=16,  # LoRA的秩，可以调整
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # 针对Qwen3的层
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 4. 加载模型和分词器 (可以使用4-bit量化节省显存)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map={'': 0},  # <--- 修改这里！
    trust_remote_code=True,
)

model = get_peft_model(model, lora_config)

# 5. 训练参数
training_args = TrainingArguments(
    output_dir="./lora_checkpoint",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
  
    learning_rate=1e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=50,
    fp16=True, # 如果你的GPU支持bf16，用bf16=True更好
    max_grad_norm=0.3,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
)

# 6. 初始化训练器并开始训练
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    # 根据你的Prompt长度调整
    
    args=training_args,
)

trainer.train()

# 7. 保存LoRA适配器
model.save_pretrained("./lora_final_adapter")