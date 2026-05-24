@echo off
cd /d "%~dp0"
echo ============ DIAGNOSTICS ============
echo.
echo [1] Working directory:
echo %CD%
echo.
echo [2] Python:
python --version 2>&1
echo.
echo [3] Python location:
where python 2>&1
echo.
echo [4] Streamlit installed:
python -c "import streamlit; print('YES - version', streamlit.__version__)" 2>&1
echo.
echo [5] App file exists:
if exist app.py (echo YES - app.py found) else (echo NO - app.py missing!)
echo.
echo [6] Data folder:
if exist data\raw\ (dir data\raw\ /b) else (echo data\raw\ folder missing!)
echo.
echo =====================================
pause
