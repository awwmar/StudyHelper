@echo off
cd /d "%~dp0"
echo Starting Study Assistant...
echo Activating Virtual Environment...
call venv\Scripts\activate
if %errorlevel% neq 0 (
    echo Failed to activate venv.
    pause
    exit /b %errorlevel%
)

echo Checking dependencies...
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install dependencies.
    pause
    exit /b 1
)

echo Starting Flask App...
python -m flask --app app run --host 127.0.0.1 --port 5000
pause
