import time
import numpy as np
from collections import OrderedDict as OD
from data.base import * # 这里会导入 get_data 和 ContinualSampler
from model import HARTrans
from method import Method
from utils import get_parser, set_seed,get_logger, log_hyperparameters, print_model_parameters, log_gpu_memory_usage
import copy
from utils_main import (compute_average_activation, compute_average_activation_old, 
                        compute_stable_neurons, compute_freeze_and_drop)
from lgcl_model import LGCLWrapper 
import os
from utils.domain_unknown_config import DomainUnknownManager # 【新增】导入域未知管理器

# 设置环境变量来禁用 tokenizers 的并行化
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- 【新增】定义 collate_fn ---
# 这个函数负责将数据样本组装成一个批次，并在这里应用域未知策略
def create_collate_fn(domain_manager, ncd_config, is_training, current_task_sampler):
    
    # 提前计算好每个任务有哪些是“新”域
    seen_domains_per_task = []
    current_seen = set()
    if ncd_config:
        for i in range(len(ncd_config)):
            task_domains = set()
            for p in ncd_config[i]['data'].values():
                task_domains.update(p)
            new_domains = task_domains - current_seen
            seen_domains_per_task.append(new_domains)
            current_seen.update(task_domains)

    def collate_fn(batch):
        # batch 是一个列表，每个元素是 (data, target, person)
        data, targets, persons = zip(*batch)
        
        data_tensor = torch.stack(data, 0)
        targets_tensor = torch.tensor(targets, dtype=torch.long)
        persons_tensor = torch.tensor(persons, dtype=torch.long)
        
        masked_persons = persons_tensor.clone()
        
        # 只有在启用域未知策略时才进行处理
        if domain_manager and domain_manager.strategy != 'none':
            # 只在训练阶段，或者在 'test_only' 策略的测试阶段进行mask
            if is_training or (not is_training and domain_manager.strategy == 'test_only'):
                task_id = current_task_sampler.task # 从sampler获取当前任务ID
                new_domains_for_this_task = seen_domains_per_task[task_id] if ncd_config and task_id < len(seen_domains_per_task) else set()

                for i in range(len(persons)):
                    is_new = persons[i] in new_domains_for_this_task
                    ratio = domain_manager.get_mask_ratio_for_task(task_id, is_new_domain=is_new)
                    if domain_manager.should_mask(ratio):
                        masked_persons[i] = -1 # -1 代表未知域

        # 返回一个字典，方便后续处理
        return {
            "x": data_tensor,
            "y": targets_tensor,
            "domains": masked_persons,      # 可能包含-1的域标签
            "true_domains": persons_tensor  # 始终是真实的域标签
        }
        
    return collate_fn

def main():
    args = get_parser()
    mylogger = get_logger(args.log_path)

    log_hyperparameters(args, mylogger)
    if args.seed is not None:
        set_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.cuda_id}")
    else:
        device = torch.device("cpu")

    args.device = device
    
    # --- 【修改】get_data 函数现在返回4个值 ---
    train_loader, test_loader, train_sampler, test_sampler = get_data(args)
    
    eval_accs = []

    base_model = HARTrans(args)
    model = LGCLWrapper(args, base_model)
    model = model.to(device)

    model.train() 

    agent = Method(model, args, mylogger, device)
    print_model_parameters(model.consense_model, mylogger)
    log_gpu_memory_usage(device, mylogger, step="init model")

    eval_accs, best_accs = train(
        args,
        agent=agent,
        train_loader=train_loader,
        eval_loader=test_loader,
        train_sampler=train_sampler, # 【新增】传递 sampler
        test_sampler=test_sampler,   # 【新增】传递 sampler
        device=device,
        mylogger=mylogger
    )

    log(eval_accs, best_accs, mylogger)

def train(args, agent, train_loader, eval_loader, train_sampler, test_sampler, device, mylogger):
    eval_accs = []
    best_accs = []
    start_task = 0
    
    # 【修改】原型计算现在在循环外进行，且使用训练加载器
    if not (hasattr(args, 'use_domain_prediction') and args.use_domain_prediction):
        agent.model.consense_model.prototypes = agent.compute_prototypes(
            args.n_tasks, False, train_loader)
        mylogger.info("Prototypes computed and assigned to the model.")
    
    freeze_masks = None
    stable_indices = None
    activation_old = None
    
    ncd_config = None
    if hasattr(args, 'use_ncd') and args.use_ncd:
        from utils.ncd_config import get_mmfi_ncd_config
        ncd_config = get_mmfi_ncd_config()
   
    for task in range(start_task, args.n_tasks):
        # 获取任务信息
        if ncd_config:
            task_config = ncd_config[task]
            current_task_classes = list(task_config['data'].keys())
            new_classes = task_config['new_classes']
            task_info = {
                'classes_in_task': current_task_classes,
                'new_classes': new_classes,
                'domains_per_class': task_config['data']
            }
            mylogger.info(f"Task {task}: Classes {current_task_classes}, New classes: {new_classes}")
            mylogger.info(f"Domain distribution: {task_config['data']}")
        else:
            classes_per_task = args.n_classes // args.n_tasks
            current_task_classes = list(range(task * classes_per_task, (task + 1) * classes_per_task))
            task_info = {'classes_in_task': current_task_classes}
        
        # --- 【修改】设置当前任务 ---
        train_sampler.set_task(task)
        test_sampler.set_task(task)
        
        agent.train()
        start = time.time()
        n_epochs = args.n_epochs
        
        agent.on_task_start(task, start_task)
        mylogger.info(f"\n>>> Task #{task} --> Model Training")
        bestacc = 0.0
        epoch_best_model = None
        
        for epoch in range(n_epochs):
            # --- 【修改】训练循环现在更简洁 ---
            for i, inc_data in enumerate(train_loader):
                # 将数据字典中的所有张量移动到设备
                inc_data = {k: v.to(device, non_blocking=True) for k, v in inc_data.items()}
                inc_data['t'] = task # 添加任务ID
                
                loss = agent.observe(inc_data, task_info=task_info, freeze_masks=freeze_masks)
                print(
                    f"Epoch: {epoch + 1}/{n_epochs} | {i+1}/{len(train_loader)} - Loss: {loss:.4f}",
                    end="\r",
                )
            
            # --- 评估循环保持不变 ---
            if (epoch + 1) % 1 == 0 or (epoch + 1 == n_epochs):
                mylogger.info(f'Task {task}. Time {time.time() - start:.2f}')
                accs, acc = agent.eval_agent(eval_loader, task, start_task)
                eval_accs.append(accs) # 注意这里是 append 不是 +=
                if acc > bestacc:
                    bestacc = acc
                    epoch_best_model = copy.deepcopy(agent.model.consense_model.state_dict())  
                agent.train()

        best_accs.append(round(bestacc, 2))
        if epoch_best_model:
            agent.model.consense_model.load_state_dict(epoch_best_model)  
        
        agent.on_task_finish(task, start_task)
        log_gpu_memory_usage(device, mylogger, step=f"End training-{task}")

        # --- 稳定性计算部分保持不变 ---
        # ...
    
    return eval_accs, best_accs

def log(eval_accs, best_accs, mylogger):
    # ----- Final Results ----- #
    # 【修改】处理可能的空列表或不同长度的列表
    max_len = max(len(l) for l in eval_accs if l) if eval_accs else 0
    if max_len > 0:
        # 将每个子列表填充到最大长度
        padded_accs = [l + [0] * (max_len - len(l)) for l in eval_accs]
        accs = np.array(padded_accs).T
        avg_acc_final = accs[:, -1].mean()
        mylogger.info(f'Final Task Accuracies: {accs[:, -1]}')
        mylogger.info(f'Final Average Accuracy: {avg_acc_final:.2f}')
    
    mylogger.info('\nBest Average Accuracy per Task')
    mylogger.info(f'Acc:{best_accs}- Avg Acc:{sum(best_accs)/len(best_accs):.2f}')

# --- 【修改】get_data 函数 ---
def get_data(args):
    # 1. 初始化域未知管理器
    domain_manager = None
    if hasattr(args, 'domain_unknown_strategy') and args.domain_unknown_strategy != 'none':
        domain_manager = DomainUnknownManager(
            strategy=args.domain_unknown_strategy,
            unknown_ratio=args.domain_unknown_ratio,
            n_tasks=args.n_tasks,
            seed=args.seed
        )

    # 2. 加载NCD配置
    ncd_config = None
    if hasattr(args, 'use_ncd') and args.use_ncd:
        from utils.ncd_config import get_mmfi_ncd_config
        ncd_config = get_mmfi_ncd_config()
        # 创建完整数据集
        train_ds = MMFI27(train=True, root=args.data_root, persons=list(range(1, 6)), classes=list(range(27)))
        test_ds = MMFI27(train=False, root=args.data_root, persons=list(range(1, 6)), classes=list(range(27)))
    else:
        # 原有模式
        train_ds = MMFI27(train=True, root=args.data_root)
        test_ds = MMFI27(train=False, root=args.data_root)

    # 3. 创建 Sampler
    train_sampler = ContinualSampler(train_ds, args.n_tasks, ncd_config)
    test_sampler = ContinualSampler(test_ds, args.n_tasks, ncd_config)

    # 4. 创建 collate_fn
    train_collate_fn = create_collate_fn(domain_manager, ncd_config, is_training=True, current_task_sampler=train_sampler)
    test_collate_fn = create_collate_fn(domain_manager, ncd_config, is_training=False, current_task_sampler=test_sampler)

    # 5. 创建 DataLoader
    train_loader = DataLoader(
        train_ds,
        num_workers=args.num_workers,
        sampler=train_sampler,
        batch_size=args.batch_size,
        collate_fn=train_collate_fn, # 使用自定义的 collate_fn
    )
    test_loader = DataLoader(
        test_ds,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        sampler=test_sampler,
        collate_fn=test_collate_fn, # 使用自定义的 collate_fn
    )
    
    args.n_classes = 27
    
    # 返回 sampler 以便在 collate_fn 中获取当前任务
    return train_loader, test_loader, train_sampler, test_sampler

if __name__ == "__main__":
    main()