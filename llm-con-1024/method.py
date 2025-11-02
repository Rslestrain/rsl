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
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os

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
        self.opt = set_optimizer(args, named_parameters=self.model.named_parameters())
        
        self.premodel = None
        
        # 根据是否使用域分类头来初始化模型
        if hasattr(self.model, 'use_domain_heads') and self.model.use_domain_heads:
    # 域分类头不需要特殊的初始化
            pass
        else:
            model_init(self.model)
        
        self.freeze_masks = None
        self.stable_indices = None
        
        # 域相关设置
        self.domain_loss_weight = getattr(args, 'domain_loss_weight', 0.1)
        
        # NCD配置
        if hasattr(args, 'use_ncd') and args.use_ncd:
            from utils.ncd_config import get_mmfi_ncd_config
            self.ncd_config = get_mmfi_ncd_config()
        
        # 创建保存混淆矩阵的目录
        self.confusion_matrix_dir = os.path.join(args.log_path, 'confusion_matrices')
        os.makedirs(self.confusion_matrix_dir, exist_ok=True)

    @property
    def name(self):
        return "myMethod"

    def observe(self, inc_data, task_info, freeze_masks):
    
    
    
    
    # 1. 模型前向传播
    # self.model 现在就是 HARTrans，它内部调用了新的蒸馏模块
    # 我们需要从 inc_data 中手动传递参数
        outputs = self.model(
        inc_data['x'], 
        task=inc_data['t'], 
        domains=inc_data['domains'], 
        labels=inc_data['y']
        )
    
    # 【核心修改】解包 HARTrans 的输出 (5个值)
    # HARTrans 返回: logits, projected_feature, task_prefix, domain_logits, distillation_loss
    # 我们暂时不需要 projected_feature 和 task_prefix
        logits, _, _, domain_logits, distillation_loss = outputs
    
    # 2. 计算分类损失 (Activity Classification Loss)
        loss_clf = self.classification_loss(logits, inc_data["y"])
    
    # 3. 计算域预测损失 (Domain Prediction Loss)
        loss_domain_pred = torch.tensor(0.0).to(self.device)
        if domain_logits is not None and self.args.use_domain_prediction:
            true_domains = inc_data['true_domains']
        # 只在有标签的样本上计算损失 (域标签不为-1的样本)
            known_mask = (inc_data['domains'] != -1)
            if known_mask.any():
                loss_domain_pred = F.cross_entropy(
                    domain_logits[known_mask],
                    true_domains[known_mask] - 1  # 真实标签 P1-P5 -> 索引 0-4
                )

    # 4. 计算总损失 (Total Loss)
    #    总损失现在由三部分组成: 分类损失, 蒸馏损失, 域预测损失
        total_loss = loss_clf
    
    # 添加新的LLM蒸馏损失
        if distillation_loss is not None and self.args.use_prompt_distillation:
            distillation_weight = getattr(self.args, 'distillation_weight', 0.5) # 注意：对于新LLM，这个权重可能需要调高
            total_loss += distillation_weight * distillation_loss
    
    # 添加域预测损失
        if self.args.use_domain_prediction:
            domain_prediction_weight = getattr(self.args, 'domain_prediction_weight', 0.2)
            total_loss += domain_prediction_weight * loss_domain_pred
    
    # --- 废弃的旧损失项 ---
    # 旧的 alpha, beta, gamma 权重不再需要，因为 loss_class, loss_task, loss_domain_align
    # 已经被统一的 distillation_loss 替代了。
    
    # 5. 更新模型参数
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
        """
        【已修改】预测时正确处理 HARTrans 返回的4个值
        """
        inc_data = {'x': x, 't': task_id, 'domains': domains}
        
        # 调用 self.model() 而不是 self.model.consense_model()
        # self.model 是 LGCLWrapper，它的 forward 方法包含了“猜测-填空-重分类”的智能逻辑
        # 它会返回5个值，我们只需要第一个，即最终的分类 logits
        # 我们只关心第一个返回值 logits，用 *rest 接收所有其他我们不关心的返回值
        outputs = self.model(
        data=x, 
        task=task_id, 
        domains=domains,
        labels=None 
        )
        logits = outputs[0]
        
        # 在评估阶段，我们通常不需要 visual_features，但如果需要可以修改
        # 这里为了接口统一，我们返回一个 None
        return logits, None 
    
    def update(self, loss, freeze_masks):
        """更新模型参数"""
        self.opt.zero_grad()
        loss.backward()
        
        # 对于域分类头，我们可能不需要应用freeze_masks
        # 因为每个域有独立的分类头
        if freeze_masks is not None and not (hasattr(self.model, 'use_domain_heads') 
                                           and self.model.use_domain_heads):
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
    def evaluate_with_confusion_matrix(self, loader, task, start_task):
        """
        【已修改】评估函数，现在可以正确处理 DataLoader 返回的数据字典
        """
        self.eval(freeze_linear_heads=True)
        
        accs = np.zeros(shape=(self.args.n_tasks,))
        
        domain_accs = {}
        if hasattr(self.args, 'evaluate_domains') and self.args.evaluate_domains:
            for t in range(task + 1):
                domain_accs[t] = {d: {'correct': 0, 'total': 0} for d in range(1, 6)}
        
        all_predictions = []
        all_targets = []
        all_domains = []
        
        seen_classes = set()
        if hasattr(self.args, 'use_ncd') and self.args.use_ncd:
            for t in range(task + 1):
                task_config = self.ncd_config[t]
                seen_classes.update(task_config['data'].keys())
            seen_classes = sorted(list(seen_classes))
        else:
            classes_per_task = self.args.n_classes // self.args.n_tasks
            seen_classes = list(range((task + 1) * classes_per_task))
        
        for task_t in range(task + 1):
            n_ok, n_total = 0, 0
            # 注意：set_task 现在在 main.py 的 train 循环中为 test_loader 设置
            # loader.sampler.set_task(task_t) # 这行可以保留，确保在独立调用时也有效

            # --- 【核心修改】 ---
            # 循环现在直接接收数据字典 inc_data
            for i, inc_data in enumerate(loader):
                # 从字典中提取数据
                data = inc_data['x']
                target = inc_data['y']
                # 对于评估，我们总是使用真实的域标签来分析性能
                person = inc_data.get('true_domains') 
                
                target = target % 27
                data = data.to(self.device)
                target = target.to(self.device)
                if person is not None:
                    person = person.to(self.device)
                
                # 预测时传递域信息（注意：这里应该传递测试时可能未知的域）
                test_domains = inc_data.get('domains').to(self.device) # 'domains' 包含-1
                logits, features = self.predict(data, task_t, test_domains)
                
                n_total += data.size(0)
                
                if logits is not None:
                    pred = logits.max(1)[1]
                    correct = pred.eq(target)
                    n_ok += correct.sum().item()
                    
                    all_predictions.extend(pred.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
                    
                    if person is not None:
                        all_domains.extend(person.cpu().numpy())
                    
                    if person is not None and hasattr(self.args, 'evaluate_domains') and self.args.evaluate_domains:
                        for idx in range(len(person)):
                            domain = person[idx].item()
                            if domain in domain_accs[task_t]: # 确保域ID有效
                                domain_accs[task_t][domain]['total'] += 1
                                if correct[idx]:
                                    domain_accs[task_t][domain]['correct'] += 1
            
            accs[task_t] = (n_ok / n_total) * 100 if n_total > 0 else 0
        
        if len(all_predictions) > 0 and len(all_targets) > 0:
            self._plot_confusion_matrix(all_targets, all_predictions, seen_classes, task, all_domains)
        
        avg_acc = np.mean(accs[: task + 1])
        accs_msg = "\t".join([str(int(x)) for x in accs])
        avg_acc_msg = f"\tAvg Acc: {avg_acc:.2f}"
        self.mylogger.info(f"\nAccuracy:{accs_msg}{avg_acc_msg}")
        
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
    
    def _plot_confusion_matrix(self, y_true, y_pred, classes, task, domains=None):
        """
        【已修改】生成并保存混淆矩阵，为每个域（Person）生成独立的子图
        """
        # 1. 计算总的混淆矩阵
        cm_overall = confusion_matrix(y_true, y_pred, labels=classes)
        
        # 2. 准备绘图
        unique_domains = sorted(list(np.unique(domains))) if domains is not None and len(domains) > 0 else []
        num_plots = 1 + len(unique_domains)
        
        # 根据图的数量调整画布大小和布局
        # 每行最多显示3个图
        num_cols = min(num_plots, 3)
        num_rows = (num_plots + num_cols - 1) // num_cols
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10 * num_cols, 8 * num_rows))
        axes = np.array(axes).flatten() # 将axes转为一维数组，方便索引

        # 3. 绘制总的混淆矩阵
        sns.heatmap(cm_overall, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=classes, yticklabels=classes,
                   cbar_kws={'label': 'Count'}, ax=axes[0])
        axes[0].set_title(f'Overall Confusion Matrix - Task {task}')
        axes[0].set_xlabel('Predicted Label')
        axes[0].set_ylabel('True Label')
        
        # 4. 为每个域绘制独立的混淆矩阵
        if unique_domains:
            for i, domain_id in enumerate(unique_domains):
                ax = axes[i + 1]
                domain_mask = (np.array(domains) == domain_id)
                
                # 筛选出当前域的数据
                domain_y_true = np.array(y_true)[domain_mask]
                domain_y_pred = np.array(y_pred)[domain_mask]
                
                if len(domain_y_true) == 0:
                    ax.set_title(f'Person {domain_id} - No Data')
                    ax.axis('off') # 如果没有数据，关闭坐标轴
                    continue

                # 确定当前域的类别范围
                domain_classes = sorted(list(set(domain_y_true) | set(domain_y_pred)))
                
                cm_domain = confusion_matrix(domain_y_true, domain_y_pred, labels=domain_classes)
                
                sns.heatmap(cm_domain, annot=True, fmt='d', cmap='Greens',
                           xticklabels=domain_classes, yticklabels=domain_classes,
                           cbar_kws={'label': 'Count'}, ax=ax)
                ax.set_title(f'Person {domain_id} Confusion Matrix - Task {task}')
                ax.set_xlabel('Predicted Label')
                ax.set_ylabel('True Label')

        # 5. 隐藏多余的子图
        for i in range(num_plots, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        
        # 6. 保存图形
        filename = os.path.join(self.confusion_matrix_dir, f'confusion_matrix_task_{task}.png')
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
        
        self.mylogger.info(f"Confusion matrix saved to {filename}")
        
        # 7. 计算并打印统计信息
        self._print_confusion_matrix_stats(cm_overall, classes)
    
    def _print_confusion_matrix_stats(self, cm, classes):
        """打印混淆矩阵的统计信息"""
        # 计算每个类别的精确率、召回率
        self.mylogger.info("\nPer-class statistics:")
        self.mylogger.info("Class\tPrecision\tRecall\tF1-Score\tSupport")
        
        for i, class_label in enumerate(classes):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            support = tp + fn
            
            self.mylogger.info(f"{class_label}\t{precision:.3f}\t\t{recall:.3f}\t\t{f1:.3f}\t\t{int(support)}")
        
        # 总体准确率
        total_correct = np.trace(cm)
        total_samples = cm.sum()
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
        self.mylogger.info(f"\nOverall Accuracy from CM: {overall_accuracy:.3f}")

    def evaluate(self, loader, task, start_task):
        """原evaluate函数，调用新的带混淆矩阵的版本"""
        return self.evaluate_with_confusion_matrix(loader, task, start_task)

    def eval_agent(self, loader, task, start_task):
        eval_loader = loader
        accs, acc = self.evaluate(eval_loader, task, start_task)
        return accs, acc

    def on_task_start(self, task, start_task):
        """任务开始时的处理"""
        consense = self.model
        
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
        consense = self.model
        
        if task <= start_task:
            pass
        else:
            # Save the conPerfix layer
            if hasattr(consense.transformer.model.layers.self_attn, 'conPerfix'):
                consense.transformer.model.layers.self_attn.prev_conPerfix = \
                    copy.deepcopy(consense.transformer.model.layers.self_attn.conPerfix)
        
        self.premodel = copy.deepcopy(self.model)
    
        # 更新feats_status（如果不使用域分类头）
        if hasattr(consense, 'feats_status'):
            for i, status in enumerate(consense.feats_status):
                if status == 1:  
                    consense.feats_status[i] = 2
            consense.feats_status.append(0)