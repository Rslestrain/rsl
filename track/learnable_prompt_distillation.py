# 新建文件: learnable_prompt_distillation.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

class LearnablePromptDistillation(nn.Module):
    """可学习提示词的多模态知识蒸馏模块"""
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device
        
        # 加载预训练的CLIP模型
        self.clip_model = CLIPModel.from_pretrained(args.clip_model_name).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(args.clip_model_name)
        
        # 冻结CLIP的参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # 获取文本编码器的嵌入维度
        self.text_dim = self.clip_model.text_projection.in_features  # 512 for CLIP-base
        
        # 初始化可学习的提示词模板
        self._init_learnable_prompts()
        
        # 特征投影层（用于对齐维度）
        if args.dataset == 'xrf48':
            self.visual_dim = 270
        elif args.dataset == 'wiar16':
            self.visual_dim = 810
        else:  # mmfi27
            self.visual_dim = 342
            
        self.visual_projector = nn.Sequential(
            nn.Linear(self.visual_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.text_dim)
        ).to(self.device)
        
    def _init_learnable_prompts(self):
        """初始化可学习的提示词"""
        
        # 为每个任务创建可学习的提示词前缀
        self.n_tasks = self.args.n_tasks
        self.n_tokens_per_prompt = 10  # 每个提示词的可学习token数
        
        # 创建可学习的前缀嵌入
        self.task_prompt_embeddings = nn.ParameterList([
            nn.Parameter(torch.randn(self.n_tokens_per_prompt, self.text_dim) * 0.02)
            for _ in range(self.n_tasks)
        ])
        
        # 创建动作描述的模板
        self.action_templates = self._create_action_templates()
        
        # 域特定的可学习提示词（Person 1-5）
        self.n_domains = self.args.n_domains
        self.domain_prompt_embeddings = nn.ParameterList([
            nn.Parameter(torch.randn(5, self.text_dim) * 0.02)  # 5个token per domain
            for _ in range(self.n_domains)
        ])
        
    def _create_action_templates(self):
        """创建动作类别的文本模板"""
        templates = {}
        
        # 根据数据集选择类别名称
        if self.args.dataset == 'xrf48':
            class_names = [
                'carrying weight', 'mopping the floor', 'using a phone',
                # ... 其他44个类别
            ]
        elif self.args.dataset == 'wiar16':
            class_names = [
                'horizontal arm wave', 'high arm wave', 'two hands wave',
                # ... 其他13个类别
            ]
        else:  # mmfi27
            class_names = self.args.class_names
            
        # 为每个类别创建固定的后缀token
        for i, name in enumerate(class_names):
            text = f"performing {name} action"
            tokens = self.tokenizer(text, return_tensors="pt", padding=False)
            with torch.no_grad():
                # 获取文本的token嵌入
                text_embeds = self.clip_model.text_model.embeddings(
                    input_ids=tokens['input_ids'].to(self.device)
                )
                templates[i] = text_embeds.squeeze(0)  # [seq_len, dim]
                
        return templates
    
    def get_text_features(self, class_ids, task_id, domain_ids=None):
        """获取带有可学习提示词的文本特征"""
        batch_size = len(class_ids)
        text_features = []
        
        for i in range(batch_size):
            class_id = class_ids[i].item()
            
            # 1. 获取任务特定的可学习前缀
            task_prefix = self.task_prompt_embeddings[task_id]  # [n_tokens, dim]
            
            # 2. 获取类别特定的固定后缀
            class_suffix = self.action_templates[class_id]  # [seq_len, dim]
            
            # 3. 如果有域信息，添加域特定的可学习token
            if domain_ids is not None:
                domain_id = domain_ids[i].item()
                if domain_id > 0:  # -1表示未知域
                    domain_tokens = self.domain_prompt_embeddings[domain_id - 1]  # [5, dim]
                    # 组合: [task_prefix, domain_tokens, class_suffix]
                    full_embedding = torch.cat([
                        task_prefix, domain_tokens, class_suffix
                    ], dim=0)
                else:
                    # 未知域：只使用任务和类别信息
                    full_embedding = torch.cat([task_prefix, class_suffix], dim=0)
            else:
                full_embedding = torch.cat([task_prefix, class_suffix], dim=0)
            
            # 4. 通过CLIP的文本编码器
            # 添加batch维度
            full_embedding = full_embedding.unsqueeze(0)  # [1, seq_len, dim]
            
            # 通过transformer层
            text_outputs = self.clip_model.text_model.encoder(
                inputs_embeds=full_embedding
            )
            
            # 获取[EOS] token的表示（最后一个位置）
            text_feature = text_outputs.last_hidden_state[:, -1, :]  # [1, dim]
            
            # 通过最终投影层
            text_feature = self.clip_model.text_projection(text_feature)
            
            text_features.append(text_feature)
            
        return torch.cat(text_features, dim=0)  # [batch_size, text_dim]
    
    def distillation_loss(self, visual_features, class_ids, task_id, domain_ids=None):
        """计算知识蒸馏损失"""
        
        # 1. 投影视觉特征到文本空间
        visual_proj = self.visual_projector(visual_features)  # [batch, text_dim]
        visual_proj = F.normalize(visual_proj, p=2, dim=-1)
        
        # 2. 获取文本特征
        text_features = self.get_text_features(class_ids, task_id, domain_ids)
        text_features = F.normalize(text_features, p=2, dim=-1)
        
        # 3. 计算对比损失
        # 正样本：同一个样本的视觉和文本特征
        pos_similarity = (visual_proj * text_features).sum(dim=-1)  # [batch]
        
        # 负样本：不同样本之间的交叉相似度
        similarity_matrix = torch.matmul(visual_proj, text_features.T)  # [batch, batch]
        
        # 温度参数
        temperature = 0.07
        
        # InfoNCE损失
        labels = torch.arange(visual_proj.size(0)).to(self.device)
        loss_v2t = F.cross_entropy(similarity_matrix / temperature, labels)
        loss_t2v = F.cross_entropy(similarity_matrix.T / temperature, labels)
        
        # 4. 额外的MSE对齐损失
        mse_loss = F.mse_loss(visual_proj, text_features)
        
        return (loss_v2t + loss_t2v) / 2 + 0.1 * mse_loss