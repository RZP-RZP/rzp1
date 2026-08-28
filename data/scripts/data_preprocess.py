"""
冲压数据集预处理脚本 阶段3
输入：data/raw/stamping_raw_sample.csv
输出：data/processed/stamping_train.csv 、 data/processed/stamping_test.csv
处理步骤：缺失值检查、异常值过滤、特征归一化、划分训练集测试集
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os

def main():
    # 文件路径
    raw_path = "./data/raw/stamping_raw_sample.csv"
    out_dir = "./data/processed"
    os.makedirs(out_dir, exist_ok=True)

    # 1读取原始数据
    df_raw = pd.read_csv(raw_path, encoding="utf-8")
    print(f"原始数据集行数：{len(df_raw)}")

    # 2缺失值处理
    df = df_raw.dropna()
    print(f"删除缺失值后行数：{len(df)}")

    # 3简单异常值过滤
    df = df[(df["sheet_thickness"] >=0.8) & (df["sheet_thickness"] <=2.0)]
    df = df[(df["blank_holder_force"] >=150) & (df["blank_holder_force"] <=450)]
    print(f"过滤异常值后行数：{len(df)}")

    # 4分离特征与标签
    feature_cols = ["sheet_thickness","blank_holder_force","stamping_speed","friction_coeff","die_gap"]
    label_cols = ["crack","wrinkle","springback"]
    X = df[feature_cols]
    y = df[label_cols]

    # 5特征归一化 MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

    # 拼接特征+标签
    df_all = pd.concat([X_scaled_df.reset_index(drop=True), y.reset_index(drop=True)],axis=1)

    # 6划分训练集测试集 7:3
    df_train, df_test = train_test_split(df_all, test_size=0.3, random_state=42)

    # 7保存输出文件
    train_out = os.path.join(out_dir,"stamping_train.csv")
    test_out = os.path.join(out_dir,"stamping_test.csv")
    df_train.to_csv(train_out,index=False,encoding="utf-8")
    df_test.to_csv(test_out,index=False,encoding="utf-8")

    print(f"训练集已输出:{train_out} 行数{len(df_train)}")
    print(f"测试集已输出:{test_out} 行数{len(df_test)}")
    print("====数据预处理全部完成====")

if __name__ == "__main__":
    main()
