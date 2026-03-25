```python
# drift.py
import torch
import numpy as np

class Task1DriftTracker:
    """只追踪Task 1类别的漂移（相对于Task 0学完时）"""
    def __init__(self):
        self.task1_features = None  # Task 1的基准特征
        self.task1_data = None      # Task 1的原始数据
        
    def store_task1_features(self, model, data_loader, device):
        """存储Task 1的特征作为基准"""
        model.eval()
        features_dict = {}
        all_x = []
        all_y = []
        
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(device), y.to(device)
                features = model.get_features(x)
                
                all_x.append(x.cpu())
                all_y.append(y.cpu())
                
                for feat, label in zip(features, y):
                    label_int = label.item()
                    if label_int not in features_dict:
                        features_dict[label_int] = []
                    features_dict[label_int].append(feat.cpu())
        
        # 转换为tensor
        for label in features_dict:
            features_dict[label] = torch.stack(features_dict[label])
        
        self.task1_features = features_dict
        self.task1_data = (torch.cat(all_x, dim=0), torch.cat(all_y, dim=0))
        
        print(f"✓ Stored Task 1 baseline features ({len(features_dict)} classes)")
        for label, feats in features_dict.items():
            print(f"  Class {label}: {feats.shape[0]} samples")
        
        return features_dict
    
    def compute_drift(self, model, device, task_id):
        """计算Task 1类别相对于基准的漂移"""
        if self.task1_data is None:
            print("Warning: Task 1 data not stored")
            return 0.0, {}
        
        model.eval()
        drift_per_class = {}
        
        with torch.no_grad():
            x_task1, y_task1 = self.task1_data
            x_task1 = x_task1.to(device)
            y_task1 = y_task1.to(device)
            
            # 批量处理
            batch_size = 128
            current_features_dict = {}
            
            for i in range(0, len(x_task1), batch_size):
                x_batch = x_task1[i:i+batch_size]
                y_batch = y_task1[i:i+batch_size]
                
                features = model.get_features(x_batch)
                
                for feat, label in zip(features, y_batch):
                    label_int = label.item()
                    if label_int not in current_features_dict:
                        current_features_dict[label_int] = []
                    current_features_dict[label_int].append(feat)
            
            # 计算每个类别的漂移
            for class_label in self.task1_features.keys():
                if class_label not in current_features_dict:
                    continue
                
                baseline_feats = self.task1_features[class_label].to(device)
                current_feats = torch.stack(current_features_dict[class_label])
                
                min_samples = min(baseline_feats.shape[0], current_feats.shape[0])
                baseline_feats = baseline_feats[:min_samples]
                current_feats = current_feats[:min_samples]
                
                # Modify：对特征进行L2归一化后再计算欧氏距离平方
                baseline_feats_normalized = torch.nn.functional.normalize(baseline_feats, p=2, dim=1)
                current_feats_normalized = torch.nn.functional.normalize(current_feats, p=2, dim=1)
                squared_distances = torch.sum((current_feats_normalized - baseline_feats_normalized) ** 2, dim=1)
                drift = squared_distances.mean().item()
                drift_per_class[class_label] = drift
        
        avg_drift = np.mean(list(drift_per_class.values())) if drift_per_class else 0.0
        
        print(f"\n📊 Task 1 Drift (After Task {task_id}):")
        print(f"   Average: {avg_drift:.4f}")
        for cls in sorted(drift_per_class.keys()):
            print(f"   Class {cls}: {drift_per_class[cls]:.4f}")
        
        return avg_drift, drift_per_class
```
### 关键方法详解

#### 1. 存储基准特征 (`store_task1_features`)
**调用时机**：在Task 1训练结束后立即执行
**功能**：存储Task 1的特征作为后续漂移计算的基准

**实现步骤**：
1. **模型评估模式**：将模型设置为评估模式
2. **批量特征提取**：使用数据加载器批量处理Task 1数据
3. **特征存储**：按类别存储特征向量
4. **数据备份**：同时保存原始输入数据用于后续重新计算

```python
# 关键代码片段
features = model.get_features(x)  # 提取128维特征
for feat, label in zip(features, y):
    label_int = label.item()
    if label_int not in features_dict:
        features_dict[label_int] = []
    features_dict[label_int].append(feat.cpu())
```

#### 2. 计算漂移值 (`compute_drift`)
**调用时机**：每个任务结束后（从Task 2开始）
**功能**：计算当前模型下Task 1特征相对于基准的漂移

**计算流程**：
1. **数据准备**：加载存储的Task 1原始数据
2. **批量特征重计算**：使用当前模型重新提取特征
3. **特征对齐**：确保基准特征和当前特征样本数量一致
4. **L2归一化**：对特征向量进行归一化处理
5. **欧氏距离计算**：计算归一化后特征的差异

```python
# 关键计算步骤
baseline_feats_normalized = torch.nn.functional.normalize(baseline_feats, p=2, dim=1)
current_feats_normalized = torch.nn.functional.normalize(current_feats, p=2, dim=1)
squared_distances = torch.sum((current_feats_normalized - baseline_feats_normalized) ** 2, dim=1)
drift = squared_distances.mean().item()
```

## 漂移计算的核心数学原理

### 1. 特征归一化
```math
\hat{f} = \frac{f}{\|f\|_2}
```
其中‖f‖₂是特征的L2范数，确保所有特征向量具有单位长度。

### 2. 欧氏距离平方
```math
d^2(\hat{f}_{\text{current}}, \hat{f}_{\text{baseline}}) = \|\hat{f}_{\text{current}} - \hat{f}_{\text{baseline}}\|_2^2
```

### 3. 平均漂移值
```math
\text{Drift} = \frac{1}{N} \sum_{i=1}^N d^2(\hat{f}_{\text{current}}^{(i)}, \hat{f}_{\text{baseline}}^{(i)})
```
