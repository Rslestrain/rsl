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
        
        # 优化器优化所有参数（包括域分类头）
        self.opt = set_optimizer(args, parameters=self.model.consense_model.parameters())
        
        self.premodel = None
        
        # 根据是否使用域分类头来初始化模型
        if hasattr(self.model.consense_model, 'use_domain_heads') and self.model.consense_model.use_domain_heads:
            # 域分类头不需要特殊的初始化
            pass
        else:
            model_init(self.model.consense_model)
        
        self.freeze_masks = None
        self.stable_indices = None
        
        # 域相关设置
        self.domain_loss_weight = getattr(args, 'domain_loss_weight', 0.1)
        
        # NCD配置
        if hasattr(args, 'use_ncd') and args.use_ncd:
            from utils.ncd_config import get_mmfi_ncd_config
            self.ncd_config = get_mmfi_ncd_config()

    @property
    def name(self):
        return "myMethod"

    def observe(self, inc_data, task_info, freeze_masks):
        """处理包含域信息的数据"""
        # 如果数据中有person字段，将其作为域信息
        if 'person' in inc_data:
            inc_data['domains'] = inc_data['person']
        
        # 模型前向传播，可能返回3个或4个值
        outputs = self.model(inc_data, task_info)
        if len(outputs) == 4:
            logits, loss_class, loss_task, loss_domain = outputs
        else:
            logits, loss_class, loss_task = outputs
            loss_domain = torch.tensor(0.0).to(self.device)
        
        # 计算分类损失
        loss_clf = self.classification_loss(logits, inc_data["y"])
        
        # 计算总损失
        total_loss = loss_clf
        if self.args.lgcl_enabled:
            alpha = self.args.alpha
            beta = self.args.beta
            gamma = self.domain_loss_weight
            total_loss = loss_clf + alpha * loss_class + beta * loss_task
            # 如果有域损失，加入总损失
            if hasattr(self.args, 'use_domain_alignment') and self.args.use_domain_alignment:
                total_loss += gamma * loss_domain
        
        # 更新模型参数
        self.update(total_loss, freeze_masks)
        
        return total_loss.item()
    
    def compute_prototypes(self, n_tasks, half_iid, dataloader):
        """计算原型（可以扩展为域特定原型）"""
        task_cls_num = 27//n_tasks
        if half_iid:
            first_task = ((n_tasks // 2) -1)
        else:
            first_task = 0 
        
        if hasattr(self.args, 'use_domain_specific_prototypes') and self.args.use_domain_specific_prototypes:
            # 为每个域计算单独的原型
            prototypes = {}
            for domain in range(1, 6):  # 5个域
                domain_prototypes = [torch.zeros(task_cls_num, 10, 3, 114).to(self.device)
                                   for _ in range(n_tasks)]
                # ... 计算域特定原型的逻辑 ...
                prototypes[domain] = domain_prototypes
            return prototypes
        else:
            # 使用原有的原型计算方法
            prototypes = [torch.zeros(task_cls_num, 10, 3, 114).to(self.device)
                         for _ in range(n_tasks)]
            
            for task_t in range(n_tasks):
                task_prototypes = prototypes[task_t]
                dataloader.sampler.set_task(task_t)
                
                class_samples = {i: [] for i in range(task_cls_num)}
                task_classes = [task_t * task_cls_num + i for i in range(task_cls_num)]
                
                for batch in dataloader:
                    if len(batch) == 3:
                        data, target, _ = batch
                    else:
                        data, target = batch
                    
                    for i in range(len(target)):
                        label = target[i].item()
                        if label in task_classes:
                            class_samples[label - task_classes[0]].append(data[i])
                
                for class_id in range(task_cls_num):
                    if len(class_samples[class_id]) > 0:
                        class_samples_tensor = torch.stack(class_samples[class_id], dim=0)
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
    
    def predict(self, x: torch.FloatTensor, task_id: int, domains=None) -> torch.FloatTensor:
        """预测时传递域信息"""
        # 直接调用consense_model的forward
        logits, features, _ = self.model.consense_model(x, task_id, domains)
        return logits, features
    
    def update(self, loss, freeze_masks):
        """更新模型参数"""
        self.opt.zero_grad()
        loss.backward()
        
        # 对于域分类头，我们可能不需要应用freeze_masks
        # 因为每个域有独立的分类头
        if freeze_masks is not None and not (hasattr(self.model.consense_model, 'use_domain_heads') 
                                           and self.model.consense_model.use_domain_heads):
            # 只对非域分类头应用冻结
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
    def evaluate(self, loader, task, start_task):
        """评估函数，支持域级别的评估"""
        self.eval(freeze_linear_heads=True)
        
        # 任务级准确率
        accs = np.zeros(shape=(self.args.n_tasks,))
        
        # 域级准确率
        domain_accs = {}
        if hasattr(self.args, 'evaluate_domains') and self.args.evaluate_domains:
            for t in range(task + 1):
                domain_accs[t] = {d: {'correct': 0, 'total': 0} 
                                 for d in range(1, 6)}
        
        for task_t in range(task + 1):
            n_ok, n_total = 0, 0
            loader.sampler.set_task(task_t)
            
            for i, batch in enumerate(loader):
                if len(batch) == 3:
                    data, target, person = batch
                    person = person.to(self.device)
                else:
                    data, target = batch
                    person = None
                
                target = target % 27
                data, target = data.to(self.device), target.to(self.device)
                
                # 预测时传递域信息
                logits, features = self.predict(data, task_t, person)
                n_total += data.size(0)
                
                if logits is not None:
                    pred = logits.max(1)[1]
                    correct = pred.eq(target)
                    n_ok += correct.sum().item()
                    
                    # 记录域级准确率
                    if person is not None and hasattr(self.args, 'evaluate_domains') and self.args.evaluate_domains:
                        for idx in range(len(person)):
                            domain = person[idx].item()
                            domain_accs[task_t][domain]['total'] += 1
                            if correct[idx]:
                                domain_accs[task_t][domain]['correct'] += 1
            
            accs[task_t] = (n_ok / n_total) * 100
        
        # 打印结果
        avg_acc = np.mean(accs[: task + 1])
        accs_msg = "\t".join([str(int(x)) for x in accs])
        avg_acc_msg = f"\tAvg Acc: {avg_acc:.2f}"
        self.mylogger.info(f"\nAccuracy:{accs_msg}{avg_acc_msg}")
        
        # 打印域级结果
        if hasattr(self.args, 'evaluate_domains') and self.args.evaluate_domains and domain_accs:
            self.mylogger.info("\nDomain-wise accuracy:")
            for t in range(task + 1):
                domain_msg = f"Task {t}: "
                for d in range(1, 6):
                    if domain_accs[t][d]['total'] > 0:
                        acc = domain_accs[t][d]['correct'] / domain_accs[t][d]['total'] * 100
                        domain_msg += f"P{d}:{acc:.1f}% "
                self.mylogger.info(domain_msg)
        
        return accs.tolist(), avg_acc

    def eval_agent(self, loader, task, start_task):
        eval_loader = loader
        accs, acc = self.evaluate(eval_loader, task, start_task)
        return accs, acc

    def on_task_start(self, task, start_task):
        """任务开始时的处理"""
        consense = self.model.consense_model
        
        if hasattr(self.args, 'use_ncd') and self.args.use_ncd:
            # NCD模式下的处理
            task_config = self.ncd_config[task]
            
            # 对于域分类头，我们可能需要根据任务调整哪些域的分类头是活跃的
            if hasattr(consense, 'use_domain_heads') and consense.use_domain_heads:
                # 可以在这里设置哪些域的分类头需要训练
                # 例如，根据task_config['data']中的域信息
                active_domains = set()
                for class_id, domains in task_config['data'].items():
                    active_domains.update(domains)
                
                self.mylogger.info(f"Task {task}: Active domains: {active_domains}")
        
        # 原有的处理逻辑
        if task <= start_task:
            pass
        if task == start_task + 1:
            # Freeze the qkv of the attention layer
            for name, param in consense.named_parameters():
                if 'qkv' in name:
                    param.requires_grad = False
        if task >= start_task + 2:
            # Freeze the qkv of the attention layer
            for name, param in consense.named_parameters():
                if 'qkv' in name:
                    param.requires_grad = False
            # Freeze the prev_conPerfix layer
            for name, param in consense.named_parameters():
                if 'prev_conPerfix' in name:
                    param.requires_grad = False
                    
        # 更新feats_status（如果不使用域分类头）
        if hasattr(consense, 'feats_status'):
            for i, status in enumerate(consense.feats_status):
                if status == 0:  
                    consense.feats_status[i] = 1

    def on_task_finish(self, task, start_task):
        """任务结束时的处理"""
        consense = self.model.consense_model
        
        if task <= start_task:
            pass
        else:
            # Save the conPerfix layer
            if hasattr(consense.transformer.model.layers.self_attn, 'conPerfix'):
                consense.transformer.model.layers.self_attn.prev_conPerfix = \
                    copy.deepcopy(consense.transformer.model.layers.self_attn.conPerfix)
        
        self.premodel = copy.deepcopy(self.model.consense_model)
        
        # 更新feats_status（如果不使用域分类头）
        if hasattr(consense, 'feats_status'):
            for i, status in enumerate(consense.feats_status):
                if status == 1:  
                    consense.feats_status[i] = 2
            consense.feats_status.append(0)