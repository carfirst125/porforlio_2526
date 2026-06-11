@echo off
echo ============================================
echo  VIB Chatbot V2 - Khoi dong Streamlit UI
echo ============================================
echo.

cd /d "%~dp0"

python -m streamlit --version >nul 2>&1
if errorlevel 1 (
    echo [!] Chua cai streamlit. Dang cai dat...
    pip install streamlit httpx
)

echo [OK] Mo Streamlit tai http://localhost:8501
echo      Dam bao API dang chay truoc tai http://localhost:8000
echo      Nhan Ctrl+C de dung.
echo.

python -m streamlit run frontend/app.py

pause
