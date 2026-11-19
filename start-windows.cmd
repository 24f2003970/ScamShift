@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 goto failed
)
if not exist ".venv\scamshift-ready" (
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto failed
    type nul > ".venv\scamshift-ready"
)
echo Open http://127.0.0.1:8765 after the server starts.
".venv\Scripts\python.exe" -m uvicorn scamshift.app:app --host 127.0.0.1 --port 8765
if errorlevel 1 goto failed
exit /b 0
:failed
echo ScamShift could not start. See the error above.
pause
exit /b 1
