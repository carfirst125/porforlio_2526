@echo off
echo ============================================
echo  VIB Chatbot V2 - Khoi dong API Server
echo ============================================
echo.

cd /d "%~dp0"

:: Kiem tra .env
if not exist .env (
    echo [!] Chua co file .env, dang tao tu .env.example...
    copy .env.example .env
    echo [OK] Tao .env thanh cong. Ban co the chinh sua truoc khi chay.
    echo.
)

:: Kiem tra Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [LOI] Khong tim thay Python.
    pause & exit /b 1
)

:: Kiem tra uvicorn
python -m uvicorn --version >nul 2>&1
if errorlevel 1 (
    echo [!] Chua cai dependencies. Dang cai dat...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [LOI] Cai dat that bai.
        pause & exit /b 1
    )
)

:: Kiem tra Ollama
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo [CANH BAO] Ollama chua chay tren localhost:11434
    echo           Hay chay Ollama va pull model truoc:
    echo           ollama pull deepseek-r1:8b
    echo           ollama pull bge-m3
    echo.
)

echo [OK] Khoi dong API tai http://localhost:8000
echo      API docs: http://localhost:8000/docs
echo      Nhan Ctrl+C de dung.
echo.

python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

pause
