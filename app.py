"""
冲压质量智能预测系统 - Flask后端服务
功能：质量预测、数据分析、历史记录、异常检测
"""
import os
import json
import sqlite3
from datetime import datetime
import pandas as pd
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "data", "models", "stamping_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "data", "models", "scaler.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "data", "models", "model_metrics.json")
DB_PATH = os.path.join(BASE_DIR, "data", "db", "stamping.db")
TRAIN_PATH = os.path.join(BASE_DIR, "data", "data", "processed", "stamping_train.csv")

FEATURE_COLS = ["sheet_thickness", "blank_holder_force", "stamping_speed",
                "friction_coeff", "die_gap"]
LABEL_COLS = ["crack", "wrinkle", "springback"]
LABEL_NAMES = {"crack": "开裂", "wrinkle": "起皱", "springback": "回弹"}
FEATURE_NAMES = {
    "sheet_thickness": "板料厚度(mm)",
    "blank_holder_force": "压边力(kN)",
    "stamping_speed": "冲压速度(mm/s)",
    "friction_coeff": "摩擦系数",
    "die_gap": "模具间隙(mm)"
}

# 工艺合理范围（用于异常检测）
PARAM_RANGES = {
    "sheet_thickness": (0.8, 2.0),
    "blank_holder_force": (150, 450),
    "stamping_speed": (80, 220),
    "friction_coeff": (0.05, 0.25),
    "die_gap": (0.8, 1.6)
}

# 加载模型和归一化器
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def anomaly_detect(params):
    """工艺参数异常检测：检查参数是否在合理范围内"""
    warnings = []
    for key, (low, high) in PARAM_RANGES.items():
        val = params.get(key)
        if val is not None:
            if val < low:
                warnings.append(f"{FEATURE_NAMES[key]}={val} 低于合理下限{low}")
            elif val > high:
                warnings.append(f"{FEATURE_NAMES[key]}={val} 超出合理上限{high}")
    return warnings


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analysis")
def analysis():
    return render_template("analysis.html")


@app.route("/history")
def history():
    return render_template("history.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    """质量预测接口"""
    try:
        data = request.get_json()
        params = {col: float(data.get(col, 0)) for col in FEATURE_COLS}

        # 异常检测
        warnings = anomaly_detect(params)

        # 模型预测（先归一化，再用每个标签单独的逻辑回归模型预测）
        X_raw = pd.DataFrame([params], columns=FEATURE_COLS)
        X = pd.DataFrame(scaler.transform(X_raw), columns=FEATURE_COLS)
        pred_list = []
        probs = {}
        for col in LABEL_COLS:
            m = model[col]
            p = int(m.predict(X)[0])
            prob = float(m.predict_proba(X)[0][1]) if len(m.classes_) > 1 else 0.0
            pred_list.append(p)
            probs[col] = round(prob, 4)

        result = {
            "predictions": {
                col: {"name": LABEL_NAMES[col], "value": int(pred_list[i]), "prob": probs[col]}
                for i, col in enumerate(LABEL_COLS)
            },
            "warnings": warnings,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # 保存预测记录到数据库
        try:
            conn = get_db()
            conn.execute("""
                INSERT INTO prediction_history
                (sheet_thickness, blank_holder_force, stamping_speed, friction_coeff, die_gap,
                 pred_crack, pred_wrinkle, pred_springback, crack_prob, wrinkle_prob, springback_prob, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                params["sheet_thickness"], params["blank_holder_force"],
                params["stamping_speed"], params["friction_coeff"], params["die_gap"],
                int(pred_list[0]), int(pred_list[1]), int(pred_list[2]),
                probs["crack"], probs["wrinkle"], probs["springback"],
                result["timestamp"]
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"保存历史记录失败: {e}")

        return jsonify({"success": True, "data": result})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/analysis")
def analysis_data():
    """数据分析接口：返回数据分布、特征相关性、标签分布等"""
    try:
        df = pd.read_csv(TRAIN_PATH)

        # 标签分布
        label_dist = {}
        for col in LABEL_COLS:
            label_dist[col] = {
                "name": LABEL_NAMES[col],
                "positive": int(df[col].sum()),
                "negative": int(len(df) - df[col].sum())
            }

        # 特征统计
        feature_stats = {}
        for col in FEATURE_COLS:
            feature_stats[col] = {
                "name": FEATURE_NAMES[col],
                "mean": round(float(df[col].mean()), 4),
                "std": round(float(df[col].std()), 4),
                "min": round(float(df[col].min()), 4),
                "max": round(float(df[col].max()), 4)
            }

        # 特征分布直方图数据
        feature_hist = {}
        for col in FEATURE_COLS:
            hist, bins = np.histogram(df[col], bins=10)
            feature_hist[col] = {
                "name": FEATURE_NAMES[col],
                "bins": [round(float(b), 3) for b in bins[1:]],
                "counts": hist.tolist()
            }

        # 模型评估指标
        metrics = {}
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                metrics = json.load(f)

        return jsonify({
            "success": True,
            "data": {
                "total_samples": len(df),
                "label_dist": label_dist,
                "feature_stats": feature_stats,
                "feature_hist": feature_hist,
                "metrics": metrics
            }
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/history")
def history_data():
    """预测历史记录接口"""
    try:
        page = int(request.args.get("page", 1))
        size = int(request.args.get("size", 20))
        offset = (page - 1) * size

        conn = get_db()
        total = conn.execute("SELECT COUNT(*) as cnt FROM prediction_history").fetchone()["cnt"]
        rows = conn.execute("""
            SELECT * FROM prediction_history
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """, (size, offset)).fetchall()
        conn.close()

        records = []
        for r in rows:
            records.append({
                "id": r["id"],
                "params": {
                    "sheet_thickness": r["sheet_thickness"],
                    "blank_holder_force": r["blank_holder_force"],
                    "stamping_speed": r["stamping_speed"],
                    "friction_coeff": r["friction_coeff"],
                    "die_gap": r["die_gap"]
                },
                "predictions": {
                    "crack": {"value": r["pred_crack"], "prob": r["crack_prob"]},
                    "wrinkle": {"value": r["pred_wrinkle"], "prob": r["wrinkle_prob"]},
                    "springback": {"value": r["pred_springback"], "prob": r["springback_prob"]}
                },
                "created_at": r["created_at"]
            })

        return jsonify({
            "success": True,
            "data": {
                "total": total,
                "page": page,
                "size": size,
                "records": records
            }
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/history/<int:record_id>", methods=["DELETE"])
def delete_history(record_id):
    """删除单条历史记录"""
    try:
        conn = get_db()
        conn.execute("DELETE FROM prediction_history WHERE id = ?", (record_id,))
        conn.commit()
        conn.close()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    print("=" * 50)
    print("冲压质量智能预测系统启动中...")
    print(f"模型路径: {MODEL_PATH}")
    print(f"数据库路径: {DB_PATH}")
    print("访问地址: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=True, host="127.0.0.1", port=5000)
