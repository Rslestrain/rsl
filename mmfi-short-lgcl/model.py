import torch
import argparse
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import copy
import numpy as np
import argparse
import math
import copy

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
        / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

def normal_pdf(pos, mu, sigma):
    a = pos - mu
    log_p = -1*torch.mul(a, a)/(2*sigma) - torch.log(sigma)/2
    return F.softmax(log_p, dim=1)

class HARTrans(torch.nn.Module):
    def __init__(self,args):
        super(HARTrans, self).__init__()
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.transformer = HARTransformer(342, 9)
        self.kernel_num = 128
        self.kernel_num_v = 16
        self.filter_sizes = [10, 10]
        self.pos_encoding = Gaussian_Position(342, 10, 10,args.device)

        self.dropout_rate = 0.5
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.encoders = []
        self.device = args.device
        self.prototypes = None

        for i, filter_size in enumerate(self.filter_sizes):
            enc_attr_name = "encoder_%d" % i
            self.__setattr__(enc_attr_name,
                             torch.nn.Conv1d(in_channels=342,
                                             out_channels=self.kernel_num,
                                             kernel_size=filter_size)
                             )
            self.encoders.append(self.__getattr__(enc_attr_name))


        self.feats = nn.ModuleList([self._create_feat() for _ in range(6)])

        self.feats_status = [0]
        
    def _create_feat(self):
        return nn.Linear(342, 27).to(self.device)


    def _aggregate(self, o, v=None):
        enc_outs = []
        for encoder in self.encoders:
            f_map = encoder(o.transpose(-1, -2))
            enc_ = F.relu(f_map)
            k_h = enc_.size()[-1]
            enc_ = F.max_pool1d(enc_, kernel_size=k_h)
            enc_ = enc_.squeeze(dim=-1)
            enc_outs.append(enc_)
        encoding = self.dropout(torch.cat(enc_outs, 1))
        q_re = F.relu(encoding)
        return q_re
    def forward(self, data, task=None):
        """
        这是新的、统一的 forward 方法。
        它整合了 return_hidden 和 forward_pre 的逻辑。
        """
        # --- 1. 特征提取部分 (来自旧的 return_hidden) ---
        x = data.reshape(-1, 10, 342)
        d1 = x.size(dim=0)
        d3 = x.size(2)
        x_reshaped = x.view(d1, -1, 1, d3)
        x_summed = torch.sum(x_reshaped, dim=-2).squeeze(-2)
        x_div = torch.div(x_summed, 1)
        x_pos = self.pos_encoding(x_div)
        
        # 从Transformer获取输出和前缀
        transformed_x, task_prefix = self.transformer(x_pos, task)
        
        # 池化得到高级特征
        class_feature = torch.mean(transformed_x, dim=1)

        # --- 2. 分类器部分 (来自旧的 forward_pre) ---
        # 注意：现在 forward_pre 的逻辑直接在这里实现
        if len(self.feats_status) == 1 or self.prototypes is None:
            # 如果是初始任务，或者没有原型，直接用第一个分类头
            logits = self.feats[0](class_feature)
        else:
            # 增量阶段，使用原型来选择正确的分类头
            batch_size = class_feature.size(0)
            selected_logits = []
            
            for i in range(batch_size):
                sample_origin = data[i] # 使用原始输入数据计算距离
                min_dist = float('inf')
                best_group_idx = 0
                
                # 遍历所有任务组的原型
                for group_idx, group_prototypes in enumerate(self.prototypes):
                    # 将原型移动到正确的设备
                    group_prototypes = group_prototypes.to(self.device)
                    # 计算样本与该组所有原型的距离的平均值或最小值
                    # 这里用平均距离作为例子
                    distances = torch.norm(sample_origin.unsqueeze(0) - group_prototypes, p=2, dim=(1,2,3))
                    avg_dist = distances.mean()
                    
                    if avg_dist < min_dist:
                        min_dist = avg_dist
                        best_group_idx = group_idx
                
                # 使用找到的最佳任务组的分类头进行预测
                logits_sample = self.feats[best_group_idx](class_feature[i].unsqueeze(0))
                selected_logits.append(logits_sample)

            logits = torch.cat(selected_logits, dim=0)

        return logits, class_feature, task_prefix

    def return_hidden(self, data, task=None):
        """
        这个函数现在可以被安全地废弃，或者为了兼容性，让它只返回特征。
        """
        # 调用新的forward，但只取第二个返回值
        _, class_feature, _ = self.forward(data, task)
        return class_feature
        
    def forward_classifier(self, x):
        """
        这个函数也被新的forward方法取代了，保留以防万一。
        """
        for i, status in enumerate(self.feats_status):
            if status == 1:  
                feat = self.feats[i](x)
                return feat
        # 如果没有找到状态为1的，返回最后一个作为默认
        return self.feats[-1](x)
        
    def forward_pre(self, x, origin=None):
        """
        这个函数的功能已经被完全整合到新的 forward 方法中，可以删掉或保留为空。
        """
        # 为了避免其他地方调用它时出错，我们可以让它调用新的forward
        logits, _, _ = self.forward(origin, task=None) # task=None因为这里不知道task id
        return logits


    def forward_activations(self, data, task=None):
        data = data.reshape(-1, 10, 342)
        d1 = data.size(dim=0)
        d3 = data.size(2)
        x = data.view(d1, -1, 1, d3)
        x = torch.sum(x, dim=-2).squeeze(-2)
        x = torch.div(x, 1)
        x = self.pos_encoding(x)
        return self.transformer.forward_activations(x,task)

        
class ThatLinear(nn.Module):
    def __init__(self,head):
        super(ThatLinear, self).__init__()
        self.dense = torch.nn.Linear(256, head)
        self.softmax = torch.nn.LogSoftmax(dim=1)
    def forward(self, x):
        de = self.dense(x)
        predict = self.softmax(de)
        return predict

class HARTransformer(nn.Module):

    def __init__(self, hidden_dim, H, filters=[1, 3, 5]):
        super(HARTransformer, self).__init__()
        self.model = Encoder(
            EncoderLayer(hidden_dim, Attention(hidden_dim , H),
                         HAR_Linear(input_dim=342, output_dim=10), 0.1)   
        )

    def forward(self, x, task=None):
        out, task_prefix = self.model(x, task)
        return out, task_prefix
    
    def forward_activations(self, x, task=None):
        return self.model.forward_activations(x, task)

class Encoder(nn.Module):
    def __init__(self, layer):
        super(Encoder, self).__init__()
        self.layers = layer
        self.norm = LayerNorm(layer.size)

    def forward(self, x, task=None):
        """
        这是最终修正版。
        """
        # 1. 调用 EncoderLayer，它会返回一个元组
        #    (包含经过注意力层和前馈网络后的张量, 从注意力层透传出来的前缀张量)
        output_tensor, prefix_tensor = self.layers(x, task)

        # 2. 只对返回的张量部分进行层归一化(Layer Normalization)
        normalized_output = self.norm(output_tensor)

        # 3. 将归一化后的张量和我们一直保留的前缀张量一起返回
        return normalized_output, prefix_tensor
    
    def forward_activations(self, x, task=None):
        return self.layers.forward_activations(x, task)
    
    def forward_activations(self, x, task=None):
        # 这里的逻辑也需要确认，确保它返回的是正确的激活值
        return self.layers.forward_activations(x, task)

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, task=None):
        # 从 attention 层获取 task_prefix
        x_attn, task_prefix = self.sublayer[0](x, lambda x: self.self_attn(x, task))
        out = self.sublayer[1](x_attn, self.feed_forward)
        return out, task_prefix
    
    def forward_activations(self, x, task=None):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, task))
        x = self.sublayer[1].norm(x)
        return self.feed_forward.forward_activations(x)
    
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # sublayer现在可能返回一个元组 (output, prefix)
        result = sublayer(self.norm(x))
        if isinstance(result, tuple):
            output, prefix = result
            return x + self.dropout(output), prefix
        else:
            return x + self.dropout(result), None # 如果没有prefix，返回None

class ConPrefix(nn.Module):
    def __init__(self, dim, mid_dim=256, act_fn='tanh'):
        super().__init__()

        self.dim = dim
        self.mid_dim = mid_dim

        self.down = nn.Linear(dim, mid_dim)
        if act_fn == 'tanh':
            self.act_fn = nn.Tanh()
        else:
            raise NotImplementedError
        self.up = nn.Linear(mid_dim, dim * 2)

    def forward(self,x):
        # Output the prefixes to be added to the qkv
        kv = self.up(self.act_fn(self.down(x)))
        q = torch.zeros(size=(kv.shape[0], kv.shape[1], self.dim),device=kv.device)

        return torch.cat((q,kv), dim=-1)
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=9, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # conPerfix
        self.conPerfix = ConPrefix(dim,mid_dim=256)
        self.prev_prefix = None
        self.last_prefix = None
        self.prev_conPerfix = ConPrefix(dim,mid_dim=256)

    def forward(self, x, task=None):
        

        B, N, C = x.shape
        task_prefix_tensor = None
        if task <= 3:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        elif task == 4:
            prefixs = self.conPerfix(x)
            task_prefix_tensor = self.conPerfix.down.weight # 我们可以取down层的权重作为前缀的代表
            qkv = (self.qkv(x)+prefixs).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            prev_prefix = self.prev_conPerfix(x)
            prefixs = self.conPerfix(x)
            task_prefix_tensor = self.conPerfix.down.weight # 我们可以取down层的权重作为前缀的代表
            qkv = (self.qkv(x)+prev_prefix+prefixs).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(0)  

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_out= (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out, task_prefix_tensor

class Gaussian_Position(nn.Module):
    def __init__(self, d_model, total_size, K=10,device=None):
        super(Gaussian_Position, self).__init__()
        self.embedding = nn.Parameter(torch.zeros(
            [K, d_model], dtype=torch.float), requires_grad=True)
        nn.init.xavier_uniform_(self.embedding, gain=1)
        self.positions = torch.tensor([i for i in range(
            total_size)], requires_grad=False).unsqueeze(1).repeat(1, K).to(device)
        s = 0.0
        interval = total_size / K
        mu = []
        for _ in range(K):
            mu.append(nn.Parameter(torch.tensor(
                s, dtype=torch.float), requires_grad=True))
            s = s + interval
        self.mu = nn.Parameter(torch.tensor(
            mu, dtype=torch.float).unsqueeze(0), requires_grad=True)
        self.sigma = nn.Parameter(torch.tensor([torch.tensor(
            [50.0], dtype=torch.float, requires_grad=True) for _ in range(K)]).unsqueeze(0))

    def forward(self, x):
        M = normal_pdf(self.positions, self.mu, self.sigma)
        pos_enc = torch.matmul(M, self.embedding)
        return x + pos_enc.unsqueeze(0).repeat(x.size(0), 1, 1)


class HAR_Linear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(HAR_Linear, self).__init__()
        self.layer1 = MaskedLinearDynamic(10, 10*4)
        self.layer2 = MaskedLinearDynamic(10*4, 10*4)
        self.layer3 = MaskedLinearDynamic(10*4, 10*4)
        self.layer4 = MaskedLinearDynamic(10*4, 10)
        self.output_dim = output_dim
        self._initialize_weights()   
        
    def forward_activations(self, x):
        x = x.transpose(-1, -2) #16, 342, 10
        x1 = F.relu(self.layer1(x))
        x2 = F.relu(self.layer2(x1))
        x3 = F.relu(self.layer3(x2))
        x4 = self.layer4(x3)
        x = x.transpose(-1, -2)
            
        return x4, [x.transpose(-1, -2), x1, x2, x3, x4]
        
    def forward(self, x):
        x = x.transpose(-1, -2) #16, 342, 10
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        x = x.transpose(-1, -2)
        return x
    
    def re_initialize(self, freeze_masks):
        i = 0
        for m in self.modules():
            if isinstance(m, MaskedLinearDynamic):
                old_weights = m.weight.data.clone().detach()
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                m.weight.data[torch.tensor(freeze_masks[i][0]).to(torch.bool)] = old_weights[torch.tensor(freeze_masks[i][0]).to(torch.bool)]
                
                old_bias = m.bias.clone().detach()
                nn.init.constant_(m.bias.data, 0)
                m.bias.data[torch.tensor(freeze_masks[i][1]).to(torch.bool)] =  old_bias[torch.tensor(freeze_masks[i][1]).to(torch.bool)]
                i += 1
                
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, MaskedLinearDynamic):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def set_masks(self, weight_mask,bias_mask):
       i = 0
       for module in self.modules():
           if isinstance(module, MaskedLinearDynamic):
               module.set_mask(weight_mask[i],bias_mask[i])
               i = i + 1
    
    

def to_var(x, requires_grad = False, volatile = False):
    if isinstance(x, np.ndarray):
        x = torch.tensor(x)
    if torch.cuda.is_available():
        x = x.to(torch.device("cuda:0"))
    return Variable(x, requires_grad = requires_grad, volatile = volatile)

class MaskedLinearDynamic(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinearDynamic, self).__init__(in_features, out_features, bias)
        self.mask_flag = False
        self.bias_flag = bias
        self.sparse_grads = True
        self.weight_mask = None
        self.bias_mask = None
        
    def set_mask(self, weight_mask, bias_mask):
        self.weight_mask = to_var(weight_mask, requires_grad=False)
        self.weight.data = self.weight.data * self.weight_mask.data
        if self.bias_flag == True:
            self.bias_mask = to_var(bias_mask, requires_grad=False)
            self.bias.data = self.bias.data * self.bias_mask.data
        self.mask_flag = True
        
    def get_mask(self):
        return self.weight_mask, self.bias_mask

    def forward(self, x):
        if self.mask_flag == True and self.sparse_grads:
            weight = self.weight * self.weight_mask
            if self.bias_flag == True:
                bias = self.bias * self.bias_mask
            else:
                bias = self.bias
            return F.linear(x, weight, bias)
        else:
            return F.linear(x, self.weight, self.bias)

def model_init(model):
    weight_masks = []
    bias_masks = []
    for module in model.modules():
        if isinstance(module, MaskedLinearDynamic):
            weight_mask = torch.from_numpy(np.ones(module.weight.shape, dtype=int))
            weight_masks.append(weight_mask)
            bias_mask = torch.from_numpy(np.ones(module.bias.shape, dtype=int))
            bias_masks.append(bias_mask)

    model.transformer.model.layers.feed_forward.set_masks(weight_masks, bias_masks)

    return model