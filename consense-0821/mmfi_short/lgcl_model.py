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
        
        # 域描述
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
        
        self.clip_model = CLIPModel.from_pretrained(args.clip_model_name).to(self.device)
        self.text_model = self.clip_model.text_model
        self.tokenizer = CLIPTokenizer.from_pretrained(args.clip_model_name)
        
        for param in self.text_model.parameters():
            param.requires_grad = False
        
        self.class_names = args.class_names
        
        self.domain_descriptions = [
            "performed by person one", "performed by person two", 
            "performed by person three", "performed by person four", "performed by person five"
        ]
        
        self._precompute_text_features()
    
    def _precompute_text_features(self):
        self.class_text_features = {}
        self.domain_text_features = {}
        self.class_domain_text_features = {}
        
        with torch.no_grad():
            for i, class_name in enumerate(self.class_names):
                inputs = self.tokenizer(f"A person is {class_name}", return_tensors="pt", padding=True, truncation=True).to(self.device)
                self.class_text_features[i] = self.clip_model.get_text_features(**inputs)
            
            for i, domain_desc in enumerate(self.domain_descriptions):
                inputs = self.tokenizer(f"An action {domain_desc}", return_tensors="pt", padding=True, truncation=True).to(self.device)
                self.domain_text_features[i+1] = self.clip_model.get_text_features(**inputs)
            
            for class_idx, class_name in enumerate(self.class_names):
                for domain_idx, domain_desc in enumerate(self.domain_descriptions):
                    text = f"A person is {class_name}, {domain_desc}"
                    inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
                    key = (class_idx, domain_idx+1)
                    self.class_domain_text_features[key] = self.clip_model.get_text_features(**inputs)
    
    def forward(self, inc_data, task_info=None):
        x = inc_data['x']
        y = inc_data['y']
        task = inc_data['t'] # 我们需要这个 task ID
        domains = inc_data.get('domains', None)
        
        logits, visual_features, task_prefix, domain_logits = self.consense_model(x, task, domains)
        
        completed_domains = domains.clone() if domains is not None else None

        if domains is not None and domain_logits is not None:
            unknown_mask = (domains == -1)
            if unknown_mask.any():
                with torch.no_grad():
                    predicted_domains = domain_logits.argmax(dim=1) + 1
                
                completed_domains[unknown_mask] = predicted_domains[unknown_mask] 
                
                if self.consense_model.use_domain_heads:
                    # --- 【核心修改点】 ---
                    # 调用 return_hidden 时，把 task ID 传进去
                    unknown_features = self.consense_model.return_hidden(x[unknown_mask], task=task)
                    
                    new_logits = self.consense_model.forward_domain_heads(
                        unknown_features, 
                        completed_domains[unknown_mask].long()
                    )
                    logits[unknown_mask] = new_logits

        loss_class = torch.tensor(0.0).to(self.device)
        loss_task = torch.tensor(0.0).to(self.device)
        loss_domain_align = torch.tensor(0.0).to(self.device)
        
        if self.args.lgcl_enabled and task_info is not None and completed_domains is not None:
            batch_size = x.size(0)
            for i in range(batch_size):
                label = y[i].item()
                domain_id = int(completed_domains[i].item())
                
                text_key = (label, domain_id)
                text_feat = self.class_domain_text_features.get(text_key, self.class_text_features.get(label))
                
                if text_feat is not None:
                    visual_feat = visual_features[i:i+1]
                    cos_sim = F.cosine_similarity(visual_feat, text_feat, dim=1)
                    loss_class += (1 - cos_sim).mean()
            
            if batch_size > 0:
                loss_class /= batch_size
            
            # 任务级对齐 (省略以保持简洁)
            
            if hasattr(self.args, 'use_domain_alignment') and self.args.use_domain_alignment:
                unique_domains_in_batch = torch.unique(completed_domains)
                count = 0
                for domain_id_tensor in unique_domains_in_batch:
                    domain_id = int(domain_id_tensor.item())
                    if domain_id != -1:
                        domain_mask = (completed_domains == domain_id)
                        if domain_mask.any():
                            domain_visual_feats = visual_features[domain_mask].mean(dim=0, keepdim=True)
                            domain_text_feat = self.domain_text_features.get(domain_id)
                            if domain_text_feat is not None:
                                domain_sim = F.cosine_similarity(domain_visual_feats, domain_text_feat)
                                loss_domain_align += (1 - domain_sim).mean()
                                count += 1
                if count > 0:
                    loss_domain_align /= count

        return logits, loss_class, loss_task, loss_domain_align, domain_logits