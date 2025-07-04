import time
import numpy as np
from collections import OrderedDict as OD
from data.base import *
from model import HARTrans
from method import Method
from utils import get_parser, set_seed,get_logger, log_hyperparameters, print_model_parameters, log_gpu_memory_usage
import copy
from model import HARTrans
from method import Method
from utils import (get_parser, set_seed, get_logger, log_hyperparameters, 
                   print_model_parameters, log_gpu_memory_usage)
from utils_main import (compute_average_activation, compute_average_activation_old, 
                        compute_stable_neurons, compute_freeze_and_drop)
from lgcl_model import LGCLWrapper 
import os
# 设置环境变量来禁用 tokenizers 的并行化，从而消除警告
# 设置为 'false' 或 'true' 都可以，目的是让它被定义
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
    train_loader, test_loader = get_data(args)

    eval_accs = []

    base_model = HARTrans(args)
    # b. 用LGCLWrapper包装它
    model = LGCLWrapper(args, base_model)
    # c. 将整个模型（包括语言模型）移动到设备
    model = model.to(device)

    model.train() 

    agent = Method(model, args,mylogger,device)
    print_model_parameters(model.consense_model, mylogger)
    log_gpu_memory_usage(device, mylogger, step="init model")

    eval_accs,best_accs = train(
        args,
        agent=agent,
        train_loader=train_loader,
        eval_loader=test_loader,
        device=device,
        mylogger=mylogger
    )


    log(eval_accs,best_accs,mylogger)


def train(args, agent, train_loader, eval_loader, device,mylogger):
    eval_accs = []
    best_accs = []
    start_task = 0
    if args.half_iid:
        start_task = (args.n_tasks // 2) - 1 
    freeze_masks = None
    stable_indices = None
    activation_old = None
    agent.model.consense_model.prototypes = agent.compute_prototypes(
        args.n_tasks, args.half_iid, train_loader)
    mylogger.info("Prototypes computed and assigned to the model.")
   
    for task in range(start_task, args.n_tasks):
        classes_per_task = args.n_classes // args.n_tasks
        current_task_classes = list(range(task * classes_per_task, (task + 1) * classes_per_task))
        task_info = {
            'classes_in_task': current_task_classes
        }
        # set task
        train_loader.sampler.set_task(task, sample_all_seen_tasks=(task == start_task),first_task=(task == start_task))
        agent.train()
        start = time.time()
        n_epochs = args.n_epochs
        if task == start_task:
            n_epochs += args.n_warmup_epochs
        agent.on_task_start(task,start_task)
        mylogger.info("\n>>> Task #{} --> Model Training".format(task+1-args.n_tasks//2))
        bestacc = 0.0
        for epoch in range(n_epochs):
            for i, (x, y) in enumerate(train_loader):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                inc_data = {"x": x, "y": y, "t": task}
                loss = agent.observe(inc_data, task_info=task_info, freeze_masks=freeze_masks)
                print(
                    f"Epoch: {epoch + 1} / {n_epochs} | {i+1} / {len(train_loader)} - Loss: {loss}",
                    end="\r",
                )
            if (epoch + 1) % 1 == 0 or (epoch + 1 == n_epochs):
                mylogger.info('Task {}. Time {:.2f}'.format(task,time.time() - start))
                accs,acc = agent.eval_agent(eval_loader, task,start_task)
                eval_accs += [accs]
                if acc > bestacc:
                    bestacc = acc
                    epoch_best_model = copy.deepcopy(agent.model.consense_model.state_dict())  
                agent.train()

        best_accs.append(round(bestacc,2))
        if epoch_best_model:
            agent.model.consense_model.load_state_dict(epoch_best_model)  
        agent.on_task_finish(task,start_task)

        log_gpu_memory_usage(device, mylogger, step=f"End training-{task}")

        # At the end of each task, calculate the stable neurons of the mlp layer and freeze them
        activation = compute_average_activation(agent.model.consense_model, train_loader,device, task=task)
        if task >=start_task+1:
            activation_old = compute_average_activation_old(agent.premodel, train_loader,device, task=task)
        np.set_printoptions(threshold=np.inf)
        stable_indices = compute_stable_neurons(activation,activation_old,in_channels=10, activation_perc = 16,stable_indices_old=stable_indices)
        for i,indices in enumerate(stable_indices):
            mylogger.info('x{}. indices {}'.format(i, indices))

        freeze_masks = compute_freeze_and_drop(stable_indices, agent.model)

    

    
    return eval_accs,best_accs



def log(eval_accs,best_accs,mylogger):
    # ----- Final Results ----- #
    accs = np.stack(eval_accs).T
    avg_acc = accs[:, -1].mean()
    mylogger.info('\nFinal Results')
    mylogger.info('Acc:{}- Avg Acc:{}'.format(best_accs,round(sum(best_accs)/len(best_accs), 2)))
    


if __name__ == "__main__":
    main()
