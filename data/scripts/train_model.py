"""
冲压质量预测模型训练脚本
输入: data/data/processed/stamping_train.csv, stamping_test.csv
输出: data/models/stamping_model.pkl
模型: RandomForestClassifier 多标签分类
"""
import os
import json
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, hamming_loss

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, "data", "processed", "stamping_train.csv")
TEST_PATH = os.path.join(BASE_DIR, "data", "processed", "stamping_test.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "stamping_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.json")

FEATURE_COLS = ["sheet_thickness", "blank_holder_force", "stamping_speed",
                "friction_coeff", "die_gap"]
LABEL_COLS = ["crack", "wrinkle", "springback"]
LABEL_NAMES = {"crack": "开裂", "wrinkle": "起皱", "springback": "回弹"}


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. 读取训练集和测试集
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)
    print(f"训练集: {len(df_train)} 条, 测试集: {len(df_test)} 条")

    X_train = df_train[FEATURE_COLS]
    y_train = df_train[LABEL_COLS]
    X_test = df_test[FEATURE_COLS]
    y_test = df_test[LABEL_COLS]

    # 2. 训练随机森林多标签分类模型
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        random_state=42
    )
    model.fit(X_train, y_train)
    print("模型训练完成")

    # 3. 模型评估
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns=LABEL_COLS)

    # 计算各标签准确率
    metrics = {}
    total_acc = 0
    for col in LABEL_COLS:
        acc = accuracy_score(y_test[col], y_pred_df[col])
        total_acc += acc
        metrics[col] = {
            "name": LABEL_NAMES[col],
            "accuracy": round(acc, 4)
        }
    metrics["average_accuracy"] = round(total_acc / len(LABEL_COLS), 4)
    metrics["hamming_loss"] = round(hamming_loss(y_test, y_pred), 4)
    metrics["sample_count"] = {"train": len(df_train), "test": len(df_test)}

    # 特征重要性
    importance = dict(zip(FEATURE_COLS, model.feature_importances_.round(4).tolist()))
    metrics["feature_importance"] = importance

    print("\n=== 模型评估结果 ===")
    print(f"平均准确率: {metrics['average_accuracy']}")
    print(f"Hamming Loss: {metrics['hamming_loss']}")
    print("各标签准确率:")
    for col in LABEL_COLS:
        print(f"  {LABEL_NAMES[col]}({col}): {metrics[col]['accuracy']}")
    print("特征重要性:")
    for k, v in importance.items():
        print(f"  {k}: {v}")

    # 4. 保存模型和评估指标
    joblib.dump(model, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\n模型已保存: {MODEL_PATH}")
    print(f"评估指标已保存: {METRICS_PATH}")
    print("=== 模型训练全部完成 ===")


if __name__ == "__main__":
    main()
