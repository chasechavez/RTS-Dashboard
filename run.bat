@echo off
cd /d "%~dp0"
title Football Hip Strength Monitor

echo ============================================
echo   Football Hip Strength Monitor
echo ============================================
echo.
echo Working folder: %CD%
echo.

:: ── Python check ─────────────────────────────────────────────────────────────
python --version
if errorlevel 1 (
    echo.
    echo ERROR: Python not found in PATH.
    echo Download from https://python.org ^(tick "Add to PATH"^)
    goto :end
)

:: ── Suppress Streamlit email prompt ──────────────────────────────────────────
if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
(echo [general] & echo email = "") > "%USERPROFILE%\.streamlit\credentials.toml"

:: ── Install packages if missing ───────────────────────────────────────────────
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo Installing packages ^(one-time, ~1 minute^)...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo ERROR: pip install failed.
        goto :end
    )
)

:: ── Show network address ──────────────────────────────────────────────────────
echo App will open at:
echo   http://localhost:8501
echo.
echo To share with others on same WiFi, use your IP address instead.
echo ^(run "ipconfig" in another window to find it^)
echo.
echo Press Ctrl+C here to stop the app.
echo ============================================
echo.

:: ── Launch ───────────────────────────────────────────────────────────────────
python -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --browser.gatherUsageStats false

:end
echo.
echo ============================================
echo  Window will close when you press any key.
echo ============================================
pause >nul
