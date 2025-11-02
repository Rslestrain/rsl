import os
import sys
import pdb
import torch
import numpy as np
from copy import deepcopy
from data import *
from data.mmfi27 import MMFI27
from torch.utils.data import DataLoader

class ContinualSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, n_tasks, ncd_config=None):
        """
        Args:
            dataset: 数据集
            n_tasks: 任务数
            ncd_config: NCD配置，包含每个任务的类别和域信息
        """
        self.first_task = False
        self.n_tasks = n_tasks
        self.dataset = dataset
        self.ncd_config = ncd_config
        
        # 如果没有提供NCD配置，使用默认的类增量模式
        if ncd_config is None:
            self.classes = np.unique(dataset.targets)
            self.n_classes = len(self.classes)
            assert self.n_classes % n_tasks == 0
            self.cpt = self.n_classes // n_tasks
        
        self.sample_all_seen_tasks = False
        self.task = None
        
        # 构建索引
        self._build_indices()
    
    def _build_indices(self):
        """构建类别和人员的索引"""
        self.class_person_indices = {}
        
        for idx in range(len(self.dataset)):
            target = self.dataset.targets[idx]
            person = self.dataset.person_labels[idx] if hasattr(self.dataset, 'person_labels') else 1
            
            key = (target, person)
            if key not in self.class_person_indices:
                self.class_person_indices[key] = []
            self.class_person_indices[key].append(idx)
        
        # 打乱每个组的索引
        for key in self.class_person_indices:
            np.random.shuffle(self.class_person_indices[key])
    
    def __iter__(self):
        if self.ncd_config is None:
            # 默认的类增量模式
            yield from self._iter_class_incremental()
        else:
            # NCD模式
            yield from self._iter_ncd()
    
    def _iter_ncd(self):
        """NCD模式的迭代器"""
        task_config = self.ncd_config[self.task]
        task_samples = []
        
        # 获取该任务应该包含的所有类别和对应的人员
        for class_id, persons in task_config['data'].items():
            for person_id in persons:
                key = (class_id, person_id)
                if key in self.class_person_indices:
                    task_samples.extend(self.class_person_indices[key])
        
        np.random.shuffle(task_samples)
        
        for idx in task_samples:
            yield idx
    
    
    def _iter_class_incremental(self):
        """原有的类增量模式"""
        task = self.task
        
        if self.sample_all_seen_tasks:
            task_classes = self.classes[: self.cpt * (task + 1)]
        else:
            task_classes = self.classes[self.cpt * task : self.cpt * (task + 1)]
        
        task_samples = []
        for class_ in task_classes:
            # 获取所有人的该类别数据
            for person in np.unique(self.dataset.person_labels) if hasattr(self.dataset, 'person_labels') else [1]:
                key = (class_, person)
                if key in self.class_person_indices:
                    task_samples.extend(self.class_person_indices[key])
        
        np.random.shuffle(task_samples)
        
        for idx in task_samples:
            yield idx
    
    def __len__(self):
        if self.ncd_config is None:
            samples_per_task = len(self.dataset) // self.n_tasks
            if self.first_task:
                return samples_per_task * (self.n_tasks // 2)
            else:
                return samples_per_task
        else:
            # NCD模式下，返回当前任务的实际样本数
            task_config = self.ncd_config[self.task]
            count = 0
            for class_id, persons in task_config['data'].items():
                for person_id in persons:
                    key = (class_id, person_id)
                    if key in self.class_person_indices:
                        count += len(self.class_person_indices[key])
            return count
    
    def set_task(self, task, sample_all_seen_tasks=False, first_task=False):
        self.task = task
        self.sample_all_seen_tasks = sample_all_seen_tasks
        self.first_task = first_task

def get_data(args):
    if hasattr(args, 'use_ncd') and args.use_ncd:
        from utils.ncd_config import get_mmfi_ncd_config
        ncd_config = get_mmfi_ncd_config()
        
        # 创建包含所有数据的完整数据集
        train_ds = MMFI27(train=True, root=args.data_root, 
                         persons=[1, 2, 3, 4, 5], 
                         classes=list(range(27)))
        test_ds = MMFI27(train=False, root=args.data_root,
                        persons=[1, 2, 3, 4, 5],
                        classes=list(range(27)))
        
        # 使用NCD配置的采样器
        train_sampler = ContinualSampler(train_ds, args.n_tasks, ncd_config)
        test_sampler = ContinualSampler(test_ds, args.n_tasks, ncd_config)
    else:
        # 原有的类增量模式
        dataset = MMFI27
        train_ds = dataset(train=True, root=args.data_root)
        test_ds = dataset(train=False, root=args.data_root)
        
        train_sampler = ContinualSampler(train_ds, args.n_tasks)
        test_sampler = ContinualSampler(test_ds, args.n_tasks)
    
    train_loader = DataLoader(
        train_ds,
        num_workers=args.num_workers,
        sampler=train_sampler,
        batch_size=args.batch_size,
    )
    
    test_loader = DataLoader(
        test_ds,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        sampler=test_sampler,
    )
    
    args.n_classes = 27
    args.n_classes_per_task = args.n_classes // args.n_tasks
    
    return train_loader, test_loader