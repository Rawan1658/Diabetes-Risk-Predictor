@echo off
:: Diabetes Risk Predictor — Rawan
:: Double-click to install dependencies and start the Flask app.

title Diabetes Risk Predictor :: Rawan
pushd "%~dp0"

echo --- step 1: ensuring requirements ---
pip install -q -r requirements.txt || (
    echo.
    echo Could not install requirements. Aborting.
    popd
    exit /b 1
)

echo.
echo --- step 2: launching Flask (http://127.0.0.1:5000) ---
python app.py

popd
pause
