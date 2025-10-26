# utils/domain_unknown_config.py
import numpy as np

class DomainUnknownManager:
    """管理域标签未知策略的类"""
    
    def __init__(self, strategy='none', unknown_ratio=0.3, n_tasks=9, seed=42):
        """
        Args:
            strategy (str): 域未知策略。
            unknown_ratio (float): 基础的未知比例。
            n_tasks (int): 总任务数。
            seed (int): 随机种子。
        """
        self.strategy = strategy
        self.unknown_ratio = unknown_ratio
        self.n_tasks = n_tasks
        self.rng = np.random.RandomState(seed)
        
    def get_mask_ratio_for_task(self, task_id, is_new_domain=False):
        """根据策略和任务ID，返回当前应该应用的未知比例"""
        if self.strategy == 'none':
            return 0.0
            
        if self.strategy == 'partial':
            return self.unknown_ratio
            
        if self.strategy == 'progressive':
            # 比例从0线性增长到 unknown_ratio * 2
            max_ratio = min(self.unknown_ratio * 2, 0.9)
            return max_ratio * (task_id / (self.n_tasks - 1))
            
        if self.strategy == 'new_domain_unknown':
            # 新域有很大概率未知，旧域小概率未知
            return 0.8 if is_new_domain else 0.1
                
        if self.strategy == 'test_only':
            # 仅在测试时域标签未知 (这个逻辑在collate_fn中通过is_training判断)
            return self.unknown_ratio
            
        return 0.0

    def should_mask(self, ratio):
        """根据给定的比例，随机决定是否要隐藏标签"""
        return self.rng.rand() < ratio