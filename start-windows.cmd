@echo off
setlocal
cd /d "%~dp0"
set "VENV_PYTHON=.venv\Scripts\python.exe"
if exist "%VENV_PYTHON%" goto check_venv
for %%V in (3.12 3.13 3.14) do (
    py -%%V -c "import sys" >nul 2>&1
    if not errorlevel 1 (
        py -%%V -m venv .venv
        if errorlevel 1 goto failed
        goto check_venv
    )
)
for %%V in (312 313 314) do (
    if exist "%LocalAppData%\Programs\Python\Python%%V\python.exe" (
        "%LocalAppData%\Programs\Python\Python%%V\python.exe" -m venv .venv
        if errorlevel 1 goto failed
        goto check_venv
    )
)
python -c "import sys; raise SystemExit(not ((3,12) <= sys.version_info[:2] <= (3,14)))" >nul 2>&1
if errorlevel 1 (
    echo Install Python 3.12, 3.13, or 3.14 to start ScamShift.
    goto failed
)
python -m venv .venv
if errorlevel 1 goto failed

:check_venv
"%VENV_PYTHON%" -c "import sys; raise SystemExit(not ((3,12) <= sys.version_info[:2] <= (3,14)))" >nul 2>&1
if errorlevel 1 (
    echo The existing .venv is incompatible or broken. Rename .venv and run this launcher again.
    goto failed
)
"%VENV_PYTHON%" -c "from pathlib import Path; import sys; marker=Path('.venv/scamshift-requirements.txt'); raise SystemExit(not (marker.is_file() and marker.read_bytes() == Path('requirements.txt').read_bytes()))"
if errorlevel 1 goto install
"%VENV_PYTHON%" -c "import fastapi, mlflow, sklearn, uvicorn" >nul 2>&1
if errorlevel 1 goto install
goto serve

:install
"%VENV_PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 goto failed
copy /y requirements.txt ".venv\scamshift-requirements.txt" >nul
if errorlevel 1 goto failed

:serve
if /i "%~1"=="--check" (
    echo ScamShift interpreter and dependencies are ready.
    exit /b 0
)
echo Open http://127.0.0.1:8765 after the server starts.
"%VENV_PYTHON%" -m uvicorn scamshift.app:app --host 127.0.0.1 --port 8765
if errorlevel 1 goto failed
exit /b 0

:failed
echo ScamShift could not start. See the error above.
if /i not "%~1"=="--no-pause" if /i not "%~1"=="--check" pause
exit /b 1
