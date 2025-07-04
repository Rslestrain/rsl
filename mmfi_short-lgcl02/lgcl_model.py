# mmfi_short/lgcl_model.py

import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPTextModel

# 导入你原来的HARTrans模型
from model import HARTrans 

class LGCLWrapper(nn.Module):
    """
    LGCLWrapper (Language-Guided Continual Learning Wrapper)
    这是一个“包装器”模型。它像一个壳，把你的原始ConSense模型(HARTrans)包在里面。
    它的作用是：
    1. 持有一个冻结的文本编码器（我们的“语言老师”）。
    2. 在训练时，协调HARTrans和语言老师，计算额外的“语义对齐损失”。
    3. 在推理时，它会“隐身”，只让内部的HARTrans工作，不产生任何额外开销。
    """
    def __init__(self, args, har_model: HARTrans):
        """
        初始化函数
        Args:
            args: 包含所有配置参数的对象 (来自config.yaml)
            har_model: 你原始的、实例化的HARTrans模型
        """
        super().__init__()
        
        # 保存传入的模型和配置
        self.args = args
        self.consense_model = har_model
        
        # --- 初始化语言模块 (仅当配置中启用时) ---
        if self.args.lgcl_enabled:
            print("LGCL is enabled. Initializing text encoder...")
            # 从Hugging Face加载预训练的CLIP模型和处理器
            # 处理器(Processor)负责把文本字符串转换成模型能懂的数字ID
            # 模型(TextModel)负责把这些ID转换成高维的语义向量
            self.text_processor = CLIPProcessor.from_pretrained(self.args.clip_model_name)
            self.text_encoder = CLIPTextModel.from_pretrained(self.args.clip_model_name)

            # 冻结文本编码器！这是整个方法成功的关键。
            # 我们不希望训练它，只把它当作一个固定的、知识渊博的“度量衡”。
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            
            # 定义用于计算损失的函数
            self.cosine_similarity = nn.CosineSimilarity(dim=-1)
            print("Text encoder loaded and frozen.")

    def to(self, device):
        """
        一个辅助函数，确保模型的所有部分（包括语言模型）都被移动到正确的设备上（如GPU）。
        """
        # 首先移动包装器自身和内部的ConSense模型
        super().to(device)
        self.consense_model.to(device)
        # 如果启用了LGCL，也要移动语言模型
        if self.args.lgcl_enabled:
            self.text_encoder.to(device)
        return self

    def get_text_features(self, texts):
        """
        一个核心辅助函数，负责将一串文本描述转换成语义特征向量。
        Args:
            texts (list of str): e.g., ["a wifi signal of walk", "a wifi signal of run"]
        Returns:
            torch.Tensor: 形状为 [batch_size, feature_dim] 的语义向量张量。
        """
        # 1. 使用processor将文本转换成模型输入格式
        inputs = self.text_processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.consense_model.device)
        # 2. 将输入喂给文本编码器
        outputs = self.text_encoder(**inputs)
        # 3. 获取语义向量。这里我们取最后一层隐藏层的平均值作为句子的表示，是一种常用且稳健的做法。
        # [CLS] token也可以，但平均池化通常泛化性更好。
        text_features = outputs.last_hidden_state.mean(dim=1)
        return text_features

    def forward(self, inc_data, task_info):
        """
        这是模型的核心逻辑所在。
        它根据当前是训练还是推理，执行不同的操作。
        """
        wifi_data = inc_data['x']
        task_id = inc_data['t']

        # --- 第一步：通过原始ConSense模型获取输出 ---
        # 我们需要修改HARTrans的forward，让它返回三个东西：
        # 1. logits: 用于分类的最终输出
        # 2. class_feature: 用于类别对齐的WiFi信号高级特征
        # 3. task_prefix: 用于任务对齐的、当前任务新学习的前缀
        logits, class_feature, task_prefix = self.consense_model(wifi_data, task=task_id)

        # --- 第二步：如果是在训练阶段，并且LGCL已启用，则计算语义损失 ---
        if self.training and self.args.lgcl_enabled:
            labels = inc_data['y']
            
            # (A) 计算类别级别(Class-Level)对齐损失
            # 1. 获取当前批次所有样本的类别名称
            class_names = [self.args.class_names[label.item()] for label in labels]
            # 2. 生成文本提示 (Prompt Engineering)
            class_prompts = [f"a wifi signal of {name} activity" for name in class_names]
            # 3. 得到这些文本提示的语义向量
            class_text_features = self.get_text_features(class_prompts)
            # 4. 计算WiFi特征和文本特征的余弦相似度。我们希望它们尽可能接近（相似度趋近于1）。
            # 损失定义为 1 - 相似度。相似度越高，损失越小。
            loss_class = 1.0 - self.cosine_similarity(class_feature, class_text_features).mean()

            # (B) 计算任务级别(Task-Level)对齐损失
            loss_task = torch.tensor(0.0).to(self.consense_model.device) # 初始化为0
            if task_prefix is not None: # 只有在学习新任务时，task_prefix才不为None
                # 1. 将任务前缀（可能是一个多层、多维的张量）池化成一个单一的向量
                prefix_feature = torch.mean(task_prefix, dim=[0, 1])
                
                # 2. 获取当前任务的描述
                current_task_classes = task_info['classes_in_task']
                task_class_names = [self.args.class_names[c] for c in current_task_classes]
                task_name_str = " or ".join(task_class_names)
                task_prompt = f"a continual learning task about activities like {task_name_str}"
                
                # 3. 得到任务描述的语义向量
                task_text_features = self.get_text_features([task_prompt])[0] # 批次大小为1，取第一个
                
                # 4. 计算前缀特征和任务文本特征的相似度
                loss_task = 1.0 - self.cosine_similarity(prefix_feature.unsqueeze(0), task_text_features.unsqueeze(0)).mean()
            
            return logits, loss_class, loss_task
        else:
            # 推理阶段：不计算任何额外损失，保持高效
            return logits, None, None