"""
数据库初始化脚本
创建SQLite数据库，导入原始数据集，建立预测历史记录表
"""
import os
import sqlite3
import pandas as pd
from datetime import datetime

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "stamping_raw_sample.csv")
DB_DIR = os.path.join(BASE_DIR, "db")
DB_PATH = os.path.join(DB_DIR, "stamping.db")


def main():
    os.makedirs(DB_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. 创建原始数据表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS raw_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sheet_thickness REAL,
            blank_holder_force REAL,
            stamping_speed REAL,
            friction_coeff REAL,
            die_gap REAL,
            crack INTEGER,
            wrinkle INTEGER,
            springback INTEGER
        )
    """)
    print("原始数据表 raw_data 创建完成")

    # 2. 导入原始数据
    df = pd.read_csv(RAW_PATH)
    df = df.dropna()  # 去掉缺失值行再入库
    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO raw_data (sheet_thickness, blank_holder_force, stamping_speed,
                                  friction_coeff, die_gap, crack, wrinkle, springback)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row["sheet_thickness"], row["blank_holder_force"],
            row["stamping_speed"], row["friction_coeff"], row["die_gap"],
            int(row["crack"]), int(row["wrinkle"]), int(row["springback"])
        ))
    print(f"已导入 {len(df)} 条原始数据")

    # 3. 创建预测历史记录表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sheet_thickness REAL,
            blank_holder_force REAL,
            stamping_speed REAL,
            friction_coeff REAL,
            die_gap REAL,
            pred_crack INTEGER,
            pred_wrinkle INTEGER,
            pred_springback INTEGER,
            crack_prob REAL,
            wrinkle_prob REAL,
            springback_prob REAL,
            created_at TEXT
        )
    """)
    print("预测历史记录表 prediction_history 创建完成")

    conn.commit()
    conn.close()
    print(f"\n数据库已创建: {DB_PATH}")
    print("=== 数据库初始化全部完成 ===")


if __name__ == "__main__":
    main()
