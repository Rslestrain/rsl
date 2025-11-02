# learnable_prompt_distillation.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

class LearnablePromptDistillation(nn.Module):
    """
    使用通用LLM（如Qwen2）进行多模态知识蒸馏的模块
    """
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device
        
        # 1. 加载本地的LLM模型和Tokenizer
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_name, 
            torch_dtype="auto", 
            device_map="auto",
            trust_remote_code=True 
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.llm_model_name, 
            use_fast=False, 
            trust_remote_code=True
        )
        
        # 核心修正 1: 在加载模型后，立刻获取并存储模型的数据类型
        self.model_dtype = self.llm_model.dtype
        print(f"LLM model loaded with dtype: {self.model_dtype}")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 2. 冻结LLM的绝大部分参数
        for param in self.llm_model.parameters():
            param.requires_grad = False
        
        # 获取LLM的嵌入维度
        self.llm_dim = self.llm_model.config.hidden_size
        
        # 3. 初始化可学习的提示词 (现在可以安全地使用 self.model_dtype)
        self._init_learnable_prompts()
        
        # 4. 特征投影层（学生适配器）
        if args.dataset == 'xrf48':
            self.visual_dim = 270
        elif args.dataset == 'wiar16':
            self.visual_dim = 810
        else:  # mmfi27
            self.visual_dim = 342
            
        # 核心修正 2: 在创建模块时就指定正确的 dtype
        self.visual_projector = nn.Sequential(
            nn.Linear(self.visual_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.llm_dim)
        ).to(device=self.device, dtype=self.model_dtype)
        
    def _init_learnable_prompts(self):
        """初始化可学习的提示词"""
        self.n_tasks = self.args.n_tasks
        self.n_domains = self.args.n_domains
        self.prompt_len = 8
        
        # 核心修正 3: 在创建Parameter时，使用正确的dtype
        self.session_prompts = nn.Parameter(
            torch.randn(self.n_tasks, self.prompt_len, self.llm_dim, dtype=self.model_dtype) * 0.02
        )
        
        self.domain_prompts = nn.Parameter(
            torch.randn(self.n_domains, self.prompt_len, self.llm_dim, dtype=self.model_dtype) * 0.02
        )
        
        self.class_templates = {i: f"The person is performing the action: {name}." for i, name in enumerate(self.args.class_names)}

    def get_teacher_target(self, class_ids, task_id, domain_ids, f_embedded):
        """获取教师目标 z_hat_augmented"""
        batch_size = len(class_ids)
        
        # 1. 准备文本部分
        texts = []
        for i in range(batch_size):
            class_id = class_ids[i].item()
            texts.append(self.class_templates[class_id])
        
        tokenized_texts = self.tokenizer(texts, return_tensors="pt", padding='longest').to(self.device)
        # 核心修正 4: 确保从embed_tokens出来的也是正确的dtype
        text_embeds = self.llm_model.model.embed_tokens(tokenized_texts.input_ids).to(self.model_dtype)
        
        # 2. 准备可学习的提示部分 (它们已经是正确的dtype)
        session_prompt_batch = self.session_prompts[task_id].unsqueeze(0).expand(batch_size, -1, -1)
        domain_prompt_batch = self.domain_prompts[domain_ids.clamp(min=0)]
        
        # 3. 准备传感特征部分 (f_embedded 已经是正确的dtype)
        sensor_embeds = f_embedded.unsqueeze(1)

        # 4. 拼接所有部分形成最终输入序列
        input_embeds = torch.cat([
            session_prompt_batch,
            domain_prompt_batch,
            text_embeds,
            sensor_embeds
        ], dim=1)
        
        # 5. 通过LLM模型
        outputs = self.llm_model(inputs_embeds=input_embeds, output_hidden_states=True)
        
        last_hidden_states = outputs.hidden_states[-1]
        teacher_target = last_hidden_states[:, -1, :]
        
        return teacher_target

    def distillation_loss(self, visual_features, class_ids, task_id, domain_ids=None):
        """计算知识蒸馏损失"""
        # 1. 学生分支: 将WiFi特征投影到LLM空间
        # 核心修正 5: 在送入projector之前，将来自骨干网络的float32特征转换为正确的dtype
        f_embedded = self.visual_projector(visual_features.to(self.model_dtype))
        
        # 在计算损失前，将结果转回float32以获得更稳定的梯度
        f_embedded_norm = F.normalize(f_embedded.float(), p=2, dim=-1)
        
        # 2. 教师分支: 获取融合了 f_embedded 的教师目标
        teacher_target = self.get_teacher_target(class_ids, task_id, domain_ids, f_embedded)
        teacher_target_norm = F.normalize(teacher_target.float(), p=2, dim=-1)
        
        # 3. 计算对齐损失 (在float32下计算)
        loss = 1 - (f_embedded_norm * teacher_target_norm).sum(dim=-1).mean()
        
        return loss