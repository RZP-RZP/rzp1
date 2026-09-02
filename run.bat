@echo off
echo ================================================
echo   Stamping Quality Prediction System - Launch
echo ================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

REM Install dependencies
echo [1/3] Checking dependencies...
python -c "import pandas, sklearn, flask, joblib" 2>nul
if errorlevel 1 (
    echo Installing dependencies...
    python -m pip install pandas numpy scikit-learn joblib flask -q
)
echo Dependencies OK

REM Check scaler, model and database
echo [2/3] Checking scaler, model and database...
if not exist "data\models\scaler.pkl" (
    echo Scaler not found, preprocessing...
    python data\scripts\data_preprocess.py
)
if not exist "data\models\stamping_model.pkl" (
    echo Model not found, training...
    python data\scripts\train_model.py
)
if not exist "data\db\stamping.db" (
    echo Database not found, initializing...
    python data\scripts\init_db.py
)
echo Scaler, model and database OK

REM Start server
echo [3/3] Starting web server...
echo.
echo ================================================
echo   System started!
echo   URL: http://127.0.0.1:5000
echo   Press Ctrl+C to stop
echo ================================================
echo.
python app.py

pause
