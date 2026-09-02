"""
自动化测试脚本
测试内容：模型加载、预测接口、数据分析接口、历史记录接口、数据库读写
"""
import os
import sys
import json
import sqlite3
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from app import app, FEATURE_COLS, LABEL_COLS, MODEL_PATH, SCALER_PATH, DB_PATH

PASSED = 0
FAILED = 0


def test_case(name):
    def decorator(func):
        def wrapper():
            global PASSED, FAILED
            try:
                func()
                print(f"  [PASS] {name}")
                PASSED += 1
            except Exception as e:
                print(f"  [FAIL] {name} - {e}")
                FAILED += 1
        return wrapper
    return decorator


@test_case("模型文件存在")
def test_model_exists():
    assert os.path.exists(MODEL_PATH), f"模型文件不存在: {MODEL_PATH}"


@test_case("归一化器文件存在")
def test_scaler_exists():
    assert os.path.exists(SCALER_PATH), f"归一化器文件不存在: {SCALER_PATH}"


@test_case("模型可加载并预测")
def test_model_predict():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    assert isinstance(model, dict), "模型应为字典格式（每个标签一个模型）"
    for col in LABEL_COLS:
        assert col in model, f"缺少标签模型: {col}"
    X_raw = pd.DataFrame([[1.2, 300, 150, 0.12, 1.1]], columns=FEATURE_COLS)
    X = pd.DataFrame(scaler.transform(X_raw), columns=FEATURE_COLS)
    for col in LABEL_COLS:
        pred = model[col].predict(X)
        assert pred[0] in [0, 1], f"{col}预测值非0/1: {pred[0]}"


@test_case("数据库文件存在")
def test_db_exists():
    assert os.path.exists(DB_PATH), f"数据库文件不存在: {DB_PATH}"


@test_case("数据库表结构正确")
def test_db_tables():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cursor.fetchall()]
    assert "raw_data" in tables, "缺少 raw_data 表"
    assert "prediction_history" in tables, "缺少 prediction_history 表"
    cursor.execute("SELECT COUNT(*) FROM raw_data")
    count = cursor.fetchone()[0]
    assert count > 0, "raw_data 表为空"
    conn.close()


@test_case("预测接口 - 正常参数")
def test_api_predict_normal():
    client = app.test_client()
    resp = client.post("/api/predict", json={
        "sheet_thickness": 1.2,
        "blank_holder_force": 300,
        "stamping_speed": 150,
        "friction_coeff": 0.12,
        "die_gap": 1.1
    })
    assert resp.status_code == 200, f"状态码错误: {resp.status_code}"
    data = resp.get_json()
    assert data["success"], "接口返回失败"
    preds = data["data"]["predictions"]
    for col in LABEL_COLS:
        assert col in preds, f"缺少预测标签: {col}"
        assert preds[col]["value"] in [0, 1], "预测值非0/1"
        assert 0 <= preds[col]["prob"] <= 1, "概率超出范围"


@test_case("预测接口 - 异常参数检测")
def test_api_predict_anomaly():
    client = app.test_client()
    resp = client.post("/api/predict", json={
        "sheet_thickness": 0.3,  # 低于下限0.8
        "blank_holder_force": 600,  # 高于上限450
        "stamping_speed": 150,
        "friction_coeff": 0.12,
        "die_gap": 1.1
    })
    data = resp.get_json()
    assert data["success"], "接口返回失败"
    warnings = data["data"]["warnings"]
    assert len(warnings) >= 2, f"应检测到至少2个异常，实际: {len(warnings)}"


@test_case("预测接口 - 参数缺失处理")
def test_api_predict_missing():
    client = app.test_client()
    resp = client.post("/api/predict", json={
        "sheet_thickness": 1.2
        # 缺少其他参数
    })
    # 后端用默认0填充，不应崩溃
    assert resp.status_code == 200, f"状态码错误: {resp.status_code}"


@test_case("数据分析接口")
def test_api_analysis():
    client = app.test_client()
    resp = client.get("/api/analysis")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"]
    d = data["data"]
    assert d["total_samples"] > 0
    assert "label_dist" in d
    assert "feature_stats" in d
    assert "feature_hist" in d
    assert "metrics" in d
    assert "average_accuracy" in d["metrics"]


@test_case("历史记录接口 - 查询")
def test_api_history_query():
    client = app.test_client()
    resp = client.get("/api/history?page=1&size=10")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"]
    assert "total" in data["data"]
    assert "records" in data["data"]
    assert isinstance(data["data"]["records"], list)


@test_case("历史记录接口 - 写入后可查询")
def test_api_history_write_read():
    client = app.test_client()
    # 先预测一条（会写入历史）
    client.post("/api/predict", json={
        "sheet_thickness": 1.5,
        "blank_holder_force": 350,
        "stamping_speed": 160,
        "friction_coeff": 0.15,
        "die_gap": 1.2
    })
    # 查询历史
    resp = client.get("/api/history?page=1&size=10")
    data = resp.get_json()
    assert data["data"]["total"] >= 1, "历史记录应至少1条"


@test_case("前端页面 - 预测页可访问")
def test_page_index():
    client = app.test_client()
    resp = client.get("/")
    assert resp.status_code == 200
    assert "冲压" in resp.get_data(as_text=True)


@test_case("前端页面 - 分析页可访问")
def test_page_analysis():
    client = app.test_client()
    resp = client.get("/analysis")
    assert resp.status_code == 200


@test_case("前端页面 - 历史页可访问")
def test_page_history():
    client = app.test_client()
    resp = client.get("/history")
    assert resp.status_code == 200


def main():
    print("=" * 50)
    print("  冲压质量智能预测系统 - 自动化测试")
    print("=" * 50)
    print()

    print("[模型与数据]")
    test_model_exists()
    test_scaler_exists()
    test_model_predict()
    test_db_exists()
    test_db_tables()

    print("\n[API接口]")
    test_api_predict_normal()
    test_api_predict_anomaly()
    test_api_predict_missing()
    test_api_analysis()
    test_api_history_query()
    test_api_history_write_read()

    print("\n[前端页面]")
    test_page_index()
    test_page_analysis()
    test_page_history()

    print("\n" + "=" * 50)
    print(f"  测试结果: 通过 {PASSED} 项, 失败 {FAILED} 项")
    if FAILED == 0:
        print("  状态: 全部通过 ✓")
    else:
        print("  状态: 存在失败项 ✗")
    print("=" * 50)

    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
