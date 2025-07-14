import numpy as np
import pandas as pd
import os


def preprocess_power_data(file_path, output_path):

    print(f"正在处理文件: {file_path}")
    df = pd.read_csv(file_path, parse_dates=['DateTime'])
    df = df.set_index('DateTime')

    #  缺失值处理
    # 将 中文'？'以及空字符 替换为 NumPy 的 NaN
    df.replace(['?', '', ' '], np.nan, inplace=True)

    # 将所有列的数据类型转换为数值型（float）
    df = df.astype(float)

    # 检查缺失值的数量
    print("填充前的缺失值数量:\n", df.isnull().sum())

    # 使用前向填充方法处理缺失值
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    # 再次检查以确认缺失值已被处理
    print("\n填充后的缺失值数量:\n", df.isnull().sum())

    # 按天聚合
    daily = pd.DataFrame()

    # 每日总和特征
    daily['global_active_power'] = df['Global_active_power'].resample('D').sum()
    daily['global_reactive_power'] = df['Global_reactive_power'].resample('D').sum()
    daily['sub_metering_1'] = df['Sub_metering_1'].resample('D').sum()
    daily['sub_metering_2'] = df['Sub_metering_2'].resample('D').sum()
    daily['sub_metering_3'] = df['Sub_metering_3'].resample('D').sum()

    # 每日平均特征
    daily['voltage'] = df['Voltage'].resample('D').mean()
    daily['global_intensity'] = df['Global_intensity'].resample('D').mean()

    # 气象特征
    daily['RR'] = df['RR'].resample('D').first()
    daily['NBJRR1'] = df['NBJRR1'].resample('D').first()
    daily['NBJRR5'] = df['NBJRR5'].resample('D').first()
    daily['NBJRR10'] = df['NBJRR10'].resample('D').first()
    daily['NBJBROU'] = df['NBJBROU'].resample('D').first()

    # 计算剩余能耗
    daily['Sub_metering_remainder'] = (daily['global_active_power'] * 1000 / 60) - \
                                      (daily['sub_metering_1'] + daily['sub_metering_2'] + daily['sub_metering_3'])


    daily = daily.dropna()  # 移除聚合后仍为空的行
    daily = daily.round(2)  # 所有数值保留2位小数
    daily.to_csv(output_path)
    print(f"保存日度数据到: {output_path}")
    print(f"共 {daily.shape[0]} 天的数据，{daily.shape[1]} 个特征\n")



if __name__ == '__main__':
    # 读取 train.csv，保留列名
    train = pd.read_csv('train.csv', sep=',', low_memory=False)
    column_names = train.columns.tolist()

    # 读取 test.csv（无列名）, 然后保存新的 test 文件
    test = pd.read_csv('test.csv', sep=',', low_memory=False)
    test.columns = column_names
    test.to_csv("test_with_header.csv", index=False)

    preprocess_power_data('train.csv', 'train_daily.csv')

    preprocess_power_data('test_with_header.csv', 'test_daily.csv')

