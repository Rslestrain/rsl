"""
简化清晰的评估代码示例
参考您提供的评估格式，使用更直观的方式
"""
import torch
import numpy as np
from torch.utils.data import DataLoader


class SimplifiedEvaluation:
    """
    简化的评估类
    参考您的代码风格，使用更清晰的逻辑
    """

    def __init__(self, model, device, logger, args):
        self.model = model
        self.device = device
        self.logger = logger
        self.args = args

    @torch.no_grad()
    def evaluate(self, loader, current_task, start_task=0):
        """
        评估模型在所有已学习任务上的表现

        参数:
            loader: 数据加载器
            current_task: 当前任务ID (0-based)
            start_task: 起始任务ID (默认0)

        返回:
            accs: 每个任务的准确率列表
            avg_acc: 平均准确率
        """
        self.model.eval()

        # 初始化准确率数组
        n_tasks = current_task + 1
        accs = np.zeros(n_tasks)

        # 评估每个已学习的任务
        for task_id in range(start_task, current_task + 1):
            # 设置当前评估的任务
            loader.sampler.set_task(task_id)

            n_correct = 0
            n_total = 0

            # 遍历该任务的所有数据
            for batch_idx, (data, target) in enumerate(loader):
                # 数据移到设备
                data = data.to(self.device)
                target = target.to(self.device)

                # 根据数据集调整标签 (如果需要)
                if self.args.get('dataset') in ['mmfi', 'mmfi_stft']:
                    target = target % 27  # MMFI有27个类
                elif self.args.get('dataset') in ['wiar', 'wiar_stft']:
                    target = target % 16  # Wiar有16个类

                # 前向传播
                logits = self.model(data, task_id=task_id, start_task=start_task)

                # 计算准确率
                pred = logits.argmax(dim=1)
                n_correct += pred.eq(target).sum().item()
                n_total += data.size(0)

            # 计算该任务的准确率
            task_acc = (n_correct / n_total) * 100 if n_total > 0 else 0
            accs[task_id - start_task] = task_acc

        # 计算平均准确率
        avg_acc = np.mean(accs)

        # 打印结果 (清晰的格式)
        self._print_results(accs, avg_acc, current_task, start_task)

        return accs.tolist(), avg_acc

    def _print_results(self, accs, avg_acc, current_task, start_task):
        """打印评估结果，格式清晰"""
        self.logger.info("\n" + "="*60)
        self.logger.info(f"Evaluation Results (Task {current_task})")
        self.logger.info("="*60)

        # 打印每个任务的准确率
        acc_strs = []
        for task_id in range(start_task, current_task + 1):
            acc = accs[task_id - start_task]
            acc_strs.append(f"Task {task_id}: {acc:.2f}%")

        self.logger.info("  " + "  |  ".join(acc_strs))
        self.logger.info("-"*60)
        self.logger.info(f"Average Accuracy: {avg_acc:.2f}%")
        self.logger.info("="*60 + "\n")


class EvenSimplerEvaluation:
    """
    更简化的版本 - 最小化代码
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def eval_all_tasks(self, loader, n_tasks):
        """
        评估所有任务

        参数:
            loader: 数据加载器
            n_tasks: 总任务数
        """
        self.model.eval()
        accs = []

        for task_id in range(n_tasks):
            loader.sampler.set_task(task_id)
            correct, total = 0, 0

            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                pred = self.model(data).argmax(1)
                correct += pred.eq(target).sum().item()
                total += data.size(0)

            acc = 100.0 * correct / total
            accs.append(acc)
            print(f"Task {task_id}: {acc:.2f}%")

        avg_acc = np.mean(accs)
        print(f"Average: {avg_acc:.2f}%")
        return accs, avg_acc


# ============ 使用示例 ============

def example_usage():
    """
    使用示例
    """
    # 假设已有的变量
    model = None  # 您的模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = None  # 您的logger
    args = {'dataset': 'mmfi_stft'}

    # 创建评估器
    evaluator = SimplifiedEvaluation(model, device, logger, args)

    # 评估
    loader = None  # 您的dataloader
    current_task = 5  # 当前任务
    start_task = 0    # 起始任务

    accs, avg_acc = evaluator.evaluate(loader, current_task, start_task)

    print(f"所有任务准确率: {accs}")
    print(f"平均准确率: {avg_acc:.2f}%")


if __name__ == '__main__':
    print(__doc__)
    print("\n" + "="*60)
    print("这是一个简化的评估示例")
    print("="*60)
    print("\n主要改进:")
    print("1. 清晰的evaluate方法，逻辑一目了然")
    print("2. 明确的变量命名 (n_correct, n_total)")
    print("3. 规范的日志输出格式")
    print("4. 分离的打印方法，便于自定义")
    print("\n您可以根据实际需求选择:")
    print("- SimplifiedEvaluation: 功能完整，日志清晰")
    print("- EvenSimplerEvaluation: 极简版本，代码最少")
