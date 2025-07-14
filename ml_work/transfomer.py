import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

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


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.d_model = d_model
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.output_layer = nn.Linear(d_model, 1)

    def _generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt):
        src_embedded = self.input_embedding(src) * math.sqrt(self.d_model)
        src_pos = self.pos_encoder(src_embedded)

        tgt_embedded = self.input_embedding(tgt) * math.sqrt(self.d_model)
        tgt_pos = self.pos_encoder(tgt_embedded)

        tgt_mask = self._generate_square_subsequent_mask(tgt.size(1), src.device)

        output = self.transformer(src_pos, tgt_pos, tgt_mask=tgt_mask)
        prediction = self.output_layer(output)
        return prediction

def create_sequences(data, target_col_idx, input_len, output_len):
    sequences = []
    for i in range(len(data) - input_len - output_len + 1):
        input_seq = data[i:i + input_len]
        target_seq_y = data[i + input_len:i + input_len + output_len, target_col_idx]
        decoder_input = data[i + input_len - 1:i + input_len + output_len - 1]
        sequences.append((input_seq, decoder_input, target_seq_y))
    return sequences


def sequences_to_tensors(sequences):
    src_list, tgt_list, y_list = [], [], []
    for src, tgt, y in sequences:
        src_list.append(torch.tensor(src, dtype=torch.float32))
        tgt_list.append(torch.tensor(tgt, dtype=torch.float32))
        y_list.append(torch.tensor(y, dtype=torch.float32).unsqueeze(-1))
    return torch.stack(src_list), torch.stack(tgt_list), torch.stack(y_list)


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for src, tgt, y in dataloader:
        src, tgt, y = src.to(device), tgt.to(device), y.to(device)
        optimizer.zero_grad()
        prediction = model(src, tgt)
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
        for src, tgt, y_true in dataloader:
            src, y_true = src.to(device), y_true.to(device)
            prediction = model(src, tgt.to(device))
            # 损失仍然在归一化数据上计算
            loss = criterion(prediction, y_true)
            total_loss += loss.item()
            all_predictions.append(prediction.cpu().numpy())
            all_truths.append(y_true.cpu().numpy())

    # 将所有批次的预测和真实值连接起来
    predictions_scaled = np.concatenate(all_predictions, axis=0)
    truths_scaled = np.concatenate(all_truths, axis=0)

    target_mean = scaler.mean_[target_col_idx]
    target_std = scaler.scale_[target_col_idx]

    # 应用逆归一化
    predictions_inversed = (predictions_scaled * target_std) + target_mean
    truths_inversed = (truths_scaled * target_std) + target_mean

    mse_inversed = mean_squared_error(truths_inversed.flatten(), predictions_inversed.flatten())
    mae_inversed = mean_absolute_error(truths_inversed.flatten(), predictions_inversed.flatten())


    return total_loss / len(dataloader), mse_inversed, mae_inversed


def plot_predictions(model, dataloader, scaler, target_col_idx, device, title, str1):
    print(f"\n正在生成图表: {title}...")
    model.eval()
    src, tgt, y_true = next(iter(dataloader))
    src, y_true = src.to(device), y_true.to(device)

    with torch.no_grad():
        prediction = model(src, tgt.to(device))

    # 取第一个样本进行可视化
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
    plt.xlabel(' (Future Time Step)', fontsize=12)
    plt.ylabel(' (Global Active Power - kW)', fontsize=12)
    plt.legend()
    plt.grid(True)

    # 根据传入的字符串保存为不同文件名
    if str1 == "01":
        plt.savefig('prediction_90_days.png')
    else:
        plt.savefig('prediction_365_days.png')
    plt.close()


#主执行流程
def main():
    print("开始加载和预处理数据...")
    try:
        train_daily = pd.read_csv('train_daily.csv', parse_dates=['DateTime']).set_index('DateTime')
        test_daily = pd.read_csv('test_daily.csv', parse_dates=['DateTime']).set_index('DateTime')
    except FileNotFoundError:
        print(" 找不到 train_daily.csv 或 test_daily.csv 文件")
        return

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_daily)
    test_scaled = scaler.transform(test_daily)
    print("特征缩放完成。")

    target_col_idx = train_daily.columns.get_loc('global_active_power')
    input_dim = train_scaled.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"输入特征维度: {input_dim}")

    # 90天预测
    print("\n" + "=" * 20 + " 90天预测 " + "=" * 20)
    seq_len_in_90, seq_len_out_90 = 90, 90
    train_seq_90 = create_sequences(train_scaled, target_col_idx, seq_len_in_90, seq_len_out_90)
    test_seq_90 = create_sequences(test_scaled, target_col_idx, seq_len_in_90, seq_len_out_90)
    src_train_90, tgt_train_90, y_train_90 = sequences_to_tensors(train_seq_90)
    src_test_90, tgt_test_90, y_test_90 = sequences_to_tensors(test_seq_90)
    train_loader_90 = DataLoader(TensorDataset(src_train_90, tgt_train_90, y_train_90), batch_size=32, shuffle=True)
    test_loader_90 = DataLoader(TensorDataset(src_test_90, tgt_test_90, y_test_90), batch_size=32, shuffle=False)

    mse_scores_90, mae_scores_90 = [], []
    model_90 = None
    for i in range(5):
        print(f"\n--- 第 {i + 1}/5 轮实验 (90天预测) ---")
        model_90 = TimeSeriesTransformer(input_dim=input_dim, d_model=128, nhead=8, num_encoder_layers=3,
                                         num_decoder_layers=3, dim_feedforward=512).to(device)
        optimizer = torch.optim.Adam(model_90.parameters(), lr=0.0001)
        criterion = nn.MSELoss()
        for epoch in range(20):
            train_loss = train_epoch(model_90, train_loader_90, optimizer, criterion, device)
            if (epoch + 1) % 5 == 0:
                print(f'Epoch {epoch + 1}/20, Train Loss (Normalized): {train_loss:.6f}')

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
                         "Short-term Forecast (90 Days) - Actual Values vs. Forecasted Values", "01")

    # 365天预测
    print("\n" + "=" * 20 + "  365天预测 " + "=" * 20)
    seq_len_in_365, seq_len_out_365 = 90, 365
    train_seq_365 = create_sequences(train_scaled, target_col_idx, seq_len_in_365, seq_len_out_365)
    test_seq_365 = create_sequences(test_scaled, target_col_idx, seq_len_in_365, seq_len_out_365)
    src_train_365, tgt_train_365, y_train_365 = sequences_to_tensors(train_seq_365)
    src_test_365, tgt_test_365, y_test_365 = sequences_to_tensors(test_seq_365)
    train_loader_365 = DataLoader(TensorDataset(src_train_365, tgt_train_365, y_train_365), batch_size=16, shuffle=True)
    test_loader_365 = DataLoader(TensorDataset(src_test_365, tgt_test_365, y_test_365), batch_size=16, shuffle=False)

    mse_scores_365, mae_scores_365 = [], []
    model_365 = None
    for i in range(5):
        print(f"\n--- 第 {i + 1}/5 轮实验 (365天预测) ---")
        model_365 = TimeSeriesTransformer(input_dim=input_dim, d_model=256, nhead=8, num_encoder_layers=4,
                                          num_decoder_layers=4, dim_feedforward=1024).to(device)
        optimizer = torch.optim.Adam(model_365.parameters(), lr=0.0001)
        criterion = nn.MSELoss()
        for epoch in range(30):
            train_loss = train_epoch(model_365, train_loader_365, optimizer, criterion, device)
            if (epoch + 1) % 5 == 0:
                print(f'Epoch {epoch + 1}/30, Train Loss (Normalized): {train_loss:.6f}')
                
        _, mse, mae = evaluate_model(model_365, test_loader_365, criterion, device, scaler, target_col_idx)
        print(f"评估结果 (逆归一化后) - MSE: {mse:.4f}, MAE: {mae:.4f}")
        mse_scores_365.append(mse)
        mae_scores_365.append(mae)


    print("--- 365天预测任务最终结果 ---")
    print(f"--- 平均MSE: {np.mean(mse_scores_365):.4f} ± {np.std(mse_scores_365):.4f}")
    print(f"--- 平均MAE: {np.mean(mae_scores_365):.4f} ± {np.std(mae_scores_365):.4f}")


    if model_365:
        plot_predictions(model_365, test_loader_365, scaler, target_col_idx, device,
                         "Long-term Forecast (365 Days) - Actual Value vs. Forecast Value", "02")


if __name__ == '__main__':
    main()