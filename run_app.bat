@echo off
echo Starting Study Assistant...
echo Activating Virtual Environment...
call venv\Scripts\activate
if %errorlevel% neq 0 (
    echo Failed to activate venv.
    pause
    exit /b %errorlevel%
)

echo Checking dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Warning: Failed to verify dependencies.
)

echo Starting Flask App...
python App.py
pause
