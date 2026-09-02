@echo off
chcp 65001 >nul
echo ================================================
echo   冲压质量智能预测系统 - 一键启动
echo ================================================
echo.

REM 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

REM 检查依赖
echo [1/3] 检查依赖包...
python -c "import pandas, sklearn, flask, joblib" 2>nul
if errorlevel 1 (
    echo 正在安装依赖...
    pip install pandas numpy scikit-learn joblib flask -q
)
echo 依赖检查完成

REM 检查模型文件
echo [2/3] 检查模型与数据库...
if not exist "data\models\stamping_model.pkl" (
    echo 模型不存在，开始训练...
    python data\scripts\train_model.py
)
if not exist "data\db\stamping.db" (
    echo 数据库不存在，开始初始化...
    python data\scripts\init_db.py
)
echo 模型与数据库就绪

REM 启动服务
echo [3/3] 启动Web服务...
echo.
echo ================================================
echo   系统启动成功！
echo   访问地址: http://127.0.0.1:5000
echo   按 Ctrl+C 停止服务
echo ================================================
echo.
python app.py

pause
