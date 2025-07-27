import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

class LGCLWrapper(nn.Module):
    def __init__(self, args, base_model):
        super().__init__()
        self.args = args
        self.consense_model = base_model
        self.device = args.device
        
        # 加载CLIP模型并移动到正确的设备
        self.clip_model = CLIPModel.from_pretrained(args.clip_model_name).to(self.device)
        self.text_model = self.clip_model.text_model
        self.tokenizer = CLIPTokenizer.from_pretrained(args.clip_model_name)
        
        # 冻结文本编码器
        for param in self.text_model.parameters():
            param.requires_grad = False
        
        # 类别名称
        self.class_names = args.class_names
        
        # 域描述（新增）
        self.domain_descriptions = [
            "performed by person one",
            "performed by person two", 
            "performed by person three",
            "performed by person four",
            "performed by person five"
        ]
        
        # 预计算所有可能的文本特征
        self._precompute_text_features()
    
    def _precompute_text_features(self):
        """预计算类别和域组合的文本特征"""
        self.class_text_features = {}
        self.domain_text_features = {}
        self.class_domain_text_features = {}
        
        with torch.no_grad():
            # 1. 类别文本特征
            for i, class_name in enumerate(self.class_names):
                text = f"A person is {class_name}"
                inputs = self.tokenizer(text, return_tensors="pt", 
                                       padding=True, truncation=True)
                # 将所有输入移动到设备
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 使用CLIP的方式获取文本特征
                text_features = self.clip_model.get_text_features(**inputs)
                self.class_text_features[i] = text_features
            
            # 2. 域文本特征
            for i, domain_desc in enumerate(self.domain_descriptions):
                text = f"An action {domain_desc}"
                inputs = self.tokenizer(text, return_tensors="pt",
                                       padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                text_features = self.clip_model.get_text_features(**inputs)
                self.domain_text_features[i+1] = text_features
            
            # 3. 类别+域组合文本特征
            for class_idx, class_name in enumerate(self.class_names):
                for domain_idx, domain_desc in enumerate(self.domain_descriptions):
                    text = f"A person is {class_name}, {domain_desc}"
                    inputs = self.tokenizer(text, return_tensors="pt",
                                           padding=True, truncation=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    text_features = self.clip_model.get_text_features(**inputs)
                    key = (class_idx, domain_idx+1)
                    self.class_domain_text_features[key] = text_features
    
    def forward(self, inc_data, task_info=None):
        # 获取数据和标签
        x = inc_data['x']
        y = inc_data['y']
        task = inc_data['t']
        
        # 获取域信息
        domains = inc_data.get('domains', inc_data.get('person', None))
        
        # 通过base model获取特征，传递域信息
        logits, visual_features, task_prefix = self.consense_model(x, task, domains)
        
        # 计算LGCL损失
        loss_class = torch.tensor(0.0).to(self.device)
        loss_task = torch.tensor(0.0).to(self.device)
        loss_domain = torch.tensor(0.0).to(self.device)
        
        if self.args.lgcl_enabled and task_info is not None:
            # 1. 类别级对齐
            batch_size = x.size(0)
            for i in range(batch_size):
                label = y[i].item()
                domain = domains[i].item() if domains is not None else None
                
                # 如果有域信息，使用类别+域的组合特征
                if domain is not None and (label, domain) in self.class_domain_text_features:
                    text_feat = self.class_domain_text_features[(label, domain)]
                else:
                    text_feat = self.class_text_features[label]
                
                visual_feat = visual_features[i:i+1]
                
                # 余弦相似度损失
                cos_sim = F.cosine_similarity(visual_feat, text_feat, dim=1)
                loss_class += (1 - cos_sim).mean()
            
            loss_class /= batch_size
            
            # 2. 任务级对齐
            task_classes = task_info['classes_in_task']
            task_text_features = []
            for class_idx in task_classes:
                if class_idx in self.class_text_features:
                    task_text_features.append(self.class_text_features[class_idx])
            
            if task_text_features and task_prefix is not None:
                task_text_features = torch.stack(task_text_features).mean(dim=0)
                # task_prefix可能需要处理形状
                if hasattr(task_prefix, 'weight'):
                    # 如果task_prefix是参数，获取其权重
                    prefix_projection = task_prefix.weight.mean(dim=0).unsqueeze(0)
                    # 投影到512维以匹配CLIP的维度
                    if prefix_projection.size(-1) != 512:
                        # 需要一个投影层，但由于task_prefix来自attention层，其维度是256
                        # 我们使用feature_projection来投影
                        prefix_projection = self.consense_model.feature_projection(
                            task_prefix.weight.mean(dim=0).unsqueeze(0).repeat(1, 342//256, 1).reshape(1, -1)[:, :342]
                        )
                else:
                    # 如果task_prefix是张量
                    prefix_projection = task_prefix
                    if prefix_projection.dim() > 2:
                        prefix_projection = prefix_projection.mean(dim=0).unsqueeze(0)
                    elif prefix_projection.dim() == 1:
                        prefix_projection = prefix_projection.unsqueeze(0)
                
                # 确保维度匹配
                if prefix_projection.size(-1) == task_text_features.size(-1):
                    task_sim = F.cosine_similarity(prefix_projection, task_text_features.unsqueeze(0))
                    loss_task = (1 - task_sim).mean()
            
            # 3. 域级对齐（新增）
            if domains is not None and hasattr(self.args, 'use_domain_alignment') and self.args.use_domain_alignment:
                unique_domains = torch.unique(domains)
                for domain in unique_domains:
                    domain_mask = domains == domain
                    if domain_mask.sum() > 0:
                        domain_visual_feats = visual_features[domain_mask].mean(dim=0, keepdim=True)
                        domain_text_feat = self.domain_text_features[domain.item()]
                        domain_sim = F.cosine_similarity(domain_visual_feats, domain_text_feat)
                        loss_domain += (1 - domain_sim).mean()
                
                if len(unique_domains) > 0:
                    loss_domain /= len(unique_domains)
        
        # 根据是否启用域对齐返回不同数量的损失
        if hasattr(self.args, 'use_domain_alignment') and self.args.use_domain_alignment:
            return logits, loss_class, loss_task, loss_domain
        else:
            return logits, loss_class, loss_task