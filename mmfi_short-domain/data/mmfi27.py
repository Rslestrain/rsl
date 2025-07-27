import torch
import torch.nn as nn
import numpy as np
import os
import os.path
import pickle
from typing import Any, Callable, Optional, Tuple, List
from torch.utils.data import Dataset
import random

class MMFI27(Dataset):
    base_folder = 'mmfi-27-python'
    
    def __init__(self, train: bool = True, root: str = '', 
                 persons: List[int] = None, 
                 classes: List[int] = None) -> None:
        """
        Args:
            train: 是否训练集
            root: 数据根目录
            persons: 要加载的人员列表，例如[1,2,3]
            classes: 要加载的类别列表，例如[0,1,2,...]
        """
        super(MMFI27, self).__init__()
        
        self.train = train
        self.persons = persons if persons is not None else [1, 2, 3, 4, 5]
        self.classes = classes if classes is not None else list(range(27))
        
        self.data = []
        self.targets = []
        self.person_labels = []  # 记录每个样本属于哪个人
        
        # 加载指定人员的数据
        for person_id in self.persons:
            file_name = f'train{person_id}' if self.train else f'test{person_id}'
            file_path = os.path.join(root, self.base_folder, file_name)
            
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    entry = pickle.load(f, encoding='latin1')
                    data = entry['data']
                    if 'labels' in entry:
                        labels = entry['labels']
                    else:
                        labels = entry['fine_labels']
                    
                    # 过滤指定类别的数据
                    for i, label in enumerate(labels):
                        if label in self.classes:
                            self.data.append(data[i])
                            self.targets.append(label)
                            self.person_labels.append(person_id)
            else:
                print(f"Warning: File {file_path} not found!")
        
        # 转换为tensor
        if len(self.data) > 0:
            self.data = torch.from_numpy(np.stack(self.data)).float()
            self.targets = np.array(self.targets)
            self.person_labels = np.array(self.person_labels)
        else:
            self.data = torch.empty(0, 10, 3, 114)
            self.targets = np.array([])
            self.person_labels = np.array([])
        
        print(f'MMFI27: Loaded {len(self.data)} samples from persons {self.persons}')
        print(f'Classes: {np.unique(self.targets) if len(self.targets) > 0 else []}')
    
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        data, target, person = self.data[index], self.targets[index], self.person_labels[index]
        return data, target, person
    
    def __len__(self) -> int:
        return len(self.data)