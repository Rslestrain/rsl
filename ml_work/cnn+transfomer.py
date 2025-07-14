import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

#多尺度CNN特征提取器
class MultiScaleCNN(nn.Module):
    def __init__(self, input_dim, d_model):
        super(MultiScaleCNN, self).__init__()
        # 定义不同大小的卷积核
        self.kernel_sizes = [3, 5, 7]
        num_kernels = len(self.kernel_sizes)
        assert d_model % num_kernels == 0, "d_model must be divisible by the number of kernel sizes."
        self.out_channels = d_model // num_kernels


        self.conv_blocks = nn.ModuleList()
        for kernel_size in self.kernel_sizes:
            # 使用'same' padding来确保输出序列长度不变
            conv_layer = nn.Sequential(
                nn.Conv1d(in_channels=input_dim, out_channels=self.out_channels, kernel_size=kernel_size,
                          padding='same'),
                nn.ReLU()
            )
            self.conv_blocks.append(conv_layer)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        outputs = []
        for conv_block in self.conv_blocks:
            outputs.append(conv_block(x))
        concatenated_features = torch.cat(outputs, dim=1)
        return concatenated_features.permute(0, 2, 1)


##注意力池化层
class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super(AttentionPooling, self).__init__()
        # 一个简单的线性层来计算注意力权重
        self.attention_net = nn.Linear(d_model, 1)

    def forward(self, x):

        # 计算注意力分数
        attention_scores = self.attention_net(x)
        attention_weights = F.softmax(attention_scores, dim=1)
        weighted_features = x * attention_weights

        # 沿时间维度求和，得到加权平均的序列表示
        pooled_output = torch.sum(weighted_features, dim=1)

        return pooled_output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return self.dropout(x)


class InnovativeConvT(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, n_heads, n_layers, dropout=0.1):
        super(InnovativeConvT, self).__init__()

        # 多尺度CNN模块
        self.multi_scale_cnn = MultiScaleCNN(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers=n_layers)
        self.attention_pooling = AttentionPooling(d_model)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, output_dim)
        )

    def forward(self, src):
        cnn_out = self.multi_scale_cnn(src)
        pos_encoded_out = self.pos_encoder(cnn_out)
        transformer_out = self.transformer_encoder(pos_encoded_out)
        pooled_out = self.attention_pooling(transformer_out)
        output = self.decoder(pooled_out)

        return output

def create_sequences_encoder_only(data, target_col_idx, input_len, output_len):
    sequences = []
    for i in range(len(data) - input_len - output_len + 1):
        input_seq = data[i:i + input_len]
        target_seq_y = data[i + input_len:i + input_len + output_len, target_col_idx]
        sequences.append((input_seq, target_seq_y))
    return sequences


def sequences_to_tensors_encoder_only(sequences):
    src_list, y_list = [], []
    for src, y in sequences:
        src_list.append(torch.tensor(src, dtype=torch.float32))
        y_list.append(torch.tensor(y, dtype=torch.float32))
    return torch.stack(src_list), torch.stack(y_list)


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for src, y in dataloader:
        src, y = src.to(device), y.to(device)
        optimizer.zero_grad()
        prediction = model(src)
        loss = criterion(prediction, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate_model(model, dataloader, criterion, device, scaler, target_col_idx):
    model.eval()
    total_loss = 0
    all_predictions, all_truths = [], []
    with torch.no_grad():
        for src, y_true in dataloader:
            src, y_true = src.to(device), y_true.to(device)
            prediction = model(src)
            loss = criterion(prediction, y_true)
            total_loss += loss.item()
            all_predictions.append(prediction.cpu().numpy())
            all_truths.append(y_true.cpu().numpy())

    predictions_scaled = np.concatenate(all_predictions, axis=0)
    truths_scaled = np.concatenate(all_truths, axis=0)

    target_mean = scaler.mean_[target_col_idx]
    target_std = scaler.scale_[target_col_idx]

    predictions_inversed = (predictions_scaled * target_std) + target_mean
    truths_inversed = (truths_scaled * target_std) + target_mean

    mse_inversed = mean_squared_error(truths_inversed.flatten(), predictions_inversed.flatten())
    mae_inversed = mean_absolute_error(truths_inversed.flatten(), predictions_inversed.flatten())

    return total_loss / len(dataloader), mse_inversed, mae_inversed


def plot_predictions(model, dataloader, scaler, target_col_idx, device, title, str1):
    print(f"\n正在生成图表: {title}...")
    model.eval()
    src, y_true = next(iter(dataloader))
    src, y_true = src.to(device), y_true.to(device)

    with torch.no_grad():
        prediction = model(src)

    y_true_np = y_true[0].cpu().numpy().flatten()
    prediction_np = prediction[0].cpu().numpy().flatten()

    target_mean = scaler.mean_[target_col_idx]
    target_std = scaler.scale_[target_col_idx]
    y_true_inversed = (y_true_np * target_std) + target_mean
    prediction_inversed = (prediction_np * target_std) + target_mean

    plt.figure(figsize=(15, 6))
    plt.plot(y_true_inversed, label=' (Actual Values)', color='orange', marker='.')
    plt.plot(prediction_inversed, label=' (Predicted Values)', color='blue', linestyle='--', marker='x')
    plt.title(title, fontsize=16)
    plt.xlabel('(Future Time Step)', fontsize=12)
    plt.ylabel('(Global Active Power - kW)', fontsize=12)
    plt.legend()
    plt.grid(True)

    if str1 == "01":
        plt.savefig('convT_prediction_90_days.png')
    else:
        plt.savefig('convT_prediction_365_days.png')
    plt.close()


# 主执行流程
def main():
    print("开始加载和预处理数据...")
    try:
        train_daily = pd.read_csv('train_daily.csv', parse_dates=['DateTime']).set_index('DateTime')
        test_daily = pd.read_csv('test_daily.csv', parse_dates=['DateTime']).set_index('DateTime')
    except FileNotFoundError:
        print("找不到 train_daily.csv 或 test_daily.csv 文件")
        return

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_daily)
    test_scaled = scaler.transform(test_daily)
    print("特征缩放完成。")

    target_col_idx = train_daily.columns.get_loc('global_active_power')
    input_dim = train_scaled.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"输入特征维度: {input_dim}")

    #90天预测
    print("\n" + "=" * 20 + " 90天预测 (CNN+Transformer Model) " + "=" * 20)
    seq_len_in_90, seq_len_out_90 = 90, 90
    train_seq_90 = create_sequences_encoder_only(train_scaled, target_col_idx, seq_len_in_90, seq_len_out_90)
    test_seq_90 = create_sequences_encoder_only(test_scaled, target_col_idx, seq_len_in_90, seq_len_out_90)
    src_train_90, y_train_90 = sequences_to_tensors_encoder_only(train_seq_90)
    src_test_90, y_test_90 = sequences_to_tensors_encoder_only(test_seq_90)
    train_loader_90 = DataLoader(TensorDataset(src_train_90, y_train_90), batch_size=32, shuffle=True)
    test_loader_90 = DataLoader(TensorDataset(src_test_90, y_test_90), batch_size=32, shuffle=False)

    mse_scores_90, mae_scores_90 = [], []
    model_90 = None
    d_model_90 = 120
    for i in range(5):
        print(f"\n--- 第 {i + 1}/5 轮实验 (90天预测) ---")
        model_90 = InnovativeConvT(input_dim=input_dim, output_dim=seq_len_out_90, d_model=d_model_90, n_heads=6,
                                   n_layers=3).to(device)
        optimizer = torch.optim.Adam(model_90.parameters(), lr=0.0001)
        criterion = nn.MSELoss()
        for epoch in range(20):
            train_loss = train_epoch(model_90, train_loader_90, optimizer, criterion, device)
            if (epoch + 1) % 5 == 0:
                print(f'Epoch {epoch + 1}/20, Train Loss: {train_loss:.6f}')
        _, mse, mae = evaluate_model(model_90, test_loader_90, criterion, device, scaler, target_col_idx)
        print(f"评估结果 (逆归一化后) - MSE: {mse:.4f}, MAE: {mae:.4f}")
        mse_scores_90.append(mse)
        mae_scores_90.append(mae)

    print("\n" + "---" * 10)
    print("--- 90天预测任务最终结果 ---")
    print(f"--- 平均MSE: {np.mean(mse_scores_90):.4f} ± {np.std(mse_scores_90):.4f}")
    print(f"--- 平均MAE: {np.mean(mae_scores_90):.4f} ± {np.std(mae_scores_90):.4f}")
    print("---" * 10)

    if model_90:
        plot_predictions(model_90, test_loader_90, scaler, target_col_idx, device,
                         "Short-term Forecast (90 Days) - Actual Values vs. Forecasted Values ( CNN+Transformer Model)", "01")

    #365天预测
    print("\n" + "=" * 20 + "  365天预测 (CNN+Transformer Model) " + "=" * 20)
    seq_len_in_365, seq_len_out_365 = 90, 365
    train_seq_365 = create_sequences_encoder_only(train_scaled, target_col_idx, seq_len_in_365, seq_len_out_365)
    test_seq_365 = create_sequences_encoder_only(test_scaled, target_col_idx, seq_len_in_365, seq_len_out_365)
    src_train_365, y_train_365 = sequences_to_tensors_encoder_only(train_seq_365)
    src_test_365, y_test_365 = sequences_to_tensors_encoder_only(test_seq_365)
    train_loader_365 = DataLoader(TensorDataset(src_train_365, y_train_365), batch_size=16, shuffle=True)
    test_loader_365 = DataLoader(TensorDataset(src_test_365, y_test_365), batch_size=16, shuffle=False)

    mse_scores_365, mae_scores_365 = [], []
    model_365 = None
    #模型超参数d_model应为CNN分支数的倍数，这里设为240 (3*80)
    d_model_365 = 240
    for i in range(5):
        print(f"\n--- 第 {i + 1}/5 轮实验 (365天预测) ---")
        model_365 = InnovativeConvT(input_dim=input_dim, output_dim=seq_len_out_365, d_model=d_model_365, n_heads=8,
                                    n_layers=4).to(device)
        optimizer = torch.optim.Adam(model_365.parameters(), lr=0.0001)
        criterion = nn.MSELoss()
        for epoch in range(30):
            train_loss = train_epoch(model_365, train_loader_365, optimizer, criterion, device)
            if (epoch + 1) % 5 == 0:
                print(f'Epoch {epoch + 1}/30, Train Loss: {train_loss:.6f}')
        _, mse, mae = evaluate_model(model_365, test_loader_365, criterion, device, scaler, target_col_idx)
        print(f"评估结果 (逆归一化后) - MSE: {mse:.4f}, MAE: {mae:.4f}")
        mse_scores_365.append(mse)
        mae_scores_365.append(mae)


    print("--- 365天预测任务最终结果 ---")
    print(f"--- 平均MSE: {np.mean(mse_scores_365):.4f} ± {np.std(mse_scores_365):.4f}")
    print(f"--- 平均MAE: {np.mean(mae_scores_365):.4f} ± {np.std(mae_scores_365):.4f}")


    if model_365:
        plot_predictions(model_365, test_loader_365, scaler, target_col_idx, device,
                         "Long-term Forecast (365 Days) - Actual Value vs. Forecast Value (CNN+Transformer Model)", "02")


if __name__ == '__main__':
    main()