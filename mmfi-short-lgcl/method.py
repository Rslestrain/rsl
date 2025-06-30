import copy
from typing import Any, Callable, Dict
from os import listdir
from os.path import isfile, join
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis as FCA
from utils import set_optimizer
from matplotlib import pyplot as plt
from model import model_init
from model import MaskedLinearDynamic

def reset_frozen_gradients(network, freeze_masks):
    mask_index = 0
    for module in network.modules():
        if isinstance(module,MaskedLinearDynamic):
            module.weight.grad[freeze_masks[mask_index][0]] = 0
            module.bias.grad[freeze_masks[mask_index][1]] = 0
            mask_index += 1
    return network

class Method(nn.Module):
    def __init__(self, model, args,mylogger,device):
        super(Method, self).__init__()

        self.args = args
        self.model = model
        self.mylogger = mylogger
        self.device = device
        self.classification_loss = F.cross_entropy
        
        # 优化器只优化 ConSense 模型部分的参数
        # 文本编码器是冻结的，所以它的参数不会出现在这里
        self.opt = set_optimizer(args, parameters=self.model.consense_model.parameters())
        
        self.premodel = None # 保留用于计算神经元稳定性的逻辑

        model_init(self.model.consense_model)
        self.freeze_masks = None
        self.stable_indices = None


    @property
    def name(self):
        return "myMethod"

    def observe(self, inc_data, task_info, freeze_masks):
        """
        这是单步训练的核心。
        """
        # 1. 模型前向传播，得到所有输出
        # 注意，我们把 task_info 传给了模型
        logits, loss_class, loss_task = self.model(inc_data, task_info)
        
        # 2. 计算分类损失
        # 注意 inc_data['y'] 可能需要根据你的数据增强策略调整
        # 这里假设没有数据增强，或者标签已经对应好
        loss_clf = self.classification_loss(logits, inc_data["y"].repeat(2))
        
        # 3. 计算总损失
        total_loss = loss_clf
        if self.args.lgcl_enabled:
            # 从配置中读取权重
            alpha = self.args.alpha
            beta = self.args.beta
            total_loss = loss_clf + alpha * loss_class + beta * loss_task

        # 4. 更新模型参数
        self.update(total_loss, freeze_masks)
        
        # 返回总损失用于打印
        return total_loss.item()
    
    def compute_prototypes(self, n_tasks, half_iid, dataloader):
        task_cls_num = 27//n_tasks
        if half_iid:
            first_task = ((n_tasks // 2) -1)
        else:
            first_task = 0 
        prototypes = [torch.zeros(task_cls_num, 10, 3, 114)
                      for _ in range(n_tasks)]
        prototypes = [p.to(self.device) for p in prototypes]
        for task_t in range(n_tasks):
            task_prototypes = prototypes[task_t]

            dataloader.sampler.set_task(task_t)

            class_samples = {i: [] for i in range(task_cls_num)}

            task_classes = [task_t * task_cls_num,
                            task_t * task_cls_num + 1, task_t * task_cls_num + 2]

            for data, target in dataloader:
                for i in range(len(target)):
                    label = target[i].item()
                    if label in task_classes:
                        class_samples[label - task_classes[0]].append(data[i])

            for class_id in range(task_cls_num):
                label = task_classes[class_id]
                if len(class_samples[class_id]) > 0:
                    class_samples_tensor = torch.stack(
                        class_samples[class_id], dim=0)
                    class_prototype = class_samples_tensor.mean(dim=0)
                    task_prototypes[class_id] = class_prototype

        return [torch.cat(prototypes[:first_task+1],dim=0)] + prototypes[first_task+1:]

    
    def process_inc(self, inc_data: Dict[str, torch.Tensor],start_task) -> torch.FloatTensor:
        x1, x2 = (inc_data["x"], inc_data["x"])
        aug_data = torch.cat((x1, x2), dim=0)
        pred = self.chosemodel(aug_data, inc_data)
        loss_c = self.loss(pred, inc_data["y"].repeat(2))
        return  loss_c 

    def chosemodel(self, aug_data, inc_data):
        task = inc_data['t']
        features = self.model.return_hidden(aug_data,task)
        pred = self.model.forward_classifier(features)
        return pred
    
    def predict(self, x: torch.FloatTensor, task_id: int ) -> torch.FloatTensor:
        # 使用 LGCLWrapper 内部的 consense 模型进行预测
        logits, features, _ = self.model.consense_model(x, task_id)
        return logits, features
    
    def update(self, loss,freeze_masks):
        self.opt.zero_grad()
        loss.backward()
        
        if freeze_masks is not None:
            weight_masks = [mask[0] for mask in freeze_masks]
            bias_masks = [mask[1] for mask in freeze_masks]   
            mask_index = 0
            for module in self.model.modules():
                if isinstance(module,MaskedLinearDynamic):
                    for row in range(module.weight.grad.shape[0]):
                        for col in range(module.weight.grad.shape[1]):
                            if weight_masks[mask_index][row][col] == 1:
                                module.weight.grad[row][col] = 0
                    module.bias.grad[bias_masks[mask_index] == 1] = 0
                    mask_index += 1

        self.opt.step()
        
    def train(self):
        self.model.train()

    def eval(self, freeze_linear_heads=True):
        self.model.eval()

    @torch.no_grad()
    def evaluate(self, loader, task,start_task):

        self.eval(freeze_linear_heads=True)
        accs = np.zeros(shape=(self.args.n_tasks,))
        for task_t in range(task + 1):
            # for task_t in range(1):

            n_ok, n_total = 0, 0
            loader.sampler.set_task(task_t)

            # iterate over samples from task
            for i, (data, target) in enumerate(loader):
                target = target % 27 
                data, target = data.to(self.device), target.to(self.device)
                logits, features = self.predict(data, task_t)
                n_total += data.size(0)
                if logits is not None:
                    pred = logits.max(1)[1]
                    n_ok += pred.eq(target).sum().item()

            accs[task_t] = (n_ok / n_total) * 100
        avg_acc = np.mean(accs[: task + 1])
        accs_msg = "\t".join([str(int(x)) for x in accs])
        avg_acc_msg = f"\tAvg Acc: {avg_acc:.2f}"
        self.mylogger.info(f"\nAccuracy:{accs_msg}{avg_acc_msg}")

        return accs.tolist(),avg_acc

    def eval_agent(self, loader, task,start_task):
        eval_loader = loader
        accs,acc = self.evaluate(eval_loader, task,start_task)
        return accs,acc

    def on_task_start(self,task,start_task):

        if task <= start_task:
            pass
        if task == start_task + 1:
            pass
            # Freeze the qkv of the attention layer
            for name, param in self.model.consense_model.named_parameters():
                if 'qkv' in name:
                    param.requires_grad = False
        if task >= start_task + 2:
            # Freeze the qkv of the attention layer
            for name, param in self.model.consense_model.named_parameters():
                if 'qkv' in name:
                    param.requires_grad = False
            # Freeze the prev_conPerfix layer
            for name, param in self.model.consense_model.named_parameters():
                if 'prev_conPerfix' in name:
                    param.requires_grad = False
                    
        for i, status in enumerate(self.model.consense_model.feats_status):
            if status == 0:  
                self.model.consense_model.feats_status[i] = 1
            

    def on_task_finish(self,task,start_task):

        if task <= start_task:
            pass
        else:
            # Save the conPerfix layer
            self.model.consense_model.transformer.model.layers.self_attn.prev_conPerfix = copy.deepcopy(self.model.consense_model.transformer.model.layers.self_attn.conPerfix)
        
        self.premodel = copy.deepcopy(self.model.consense_model)
        
        for i, status in enumerate(self.model.feats_status):
            if status == 1:  
                self.model.feats_status[i] = 2
        #self.model.feats.append(self.model.consense_model._create_feat())
        self.model.consense_model.feats_status.append(0)
        
        
        
        
