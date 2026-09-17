@echo off
chcp 65001 >nul
title Sydney Web UI (Gradio)
echo ============================================================
echo   正在启动 Sydney 网页交互界面 (Web UI)...
echo ============================================================

set SCRIPT_DIR=%~dp0

if exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
    "%SCRIPT_DIR%.venv\Scripts\python.exe" "%SCRIPT_DIR%web_ui.py" %*
) else if exist "%SCRIPT_DIR%venv\Scripts\python.exe" (
    "%SCRIPT_DIR%venv\Scripts\python.exe" "%SCRIPT_DIR%web_ui.py" %*
) else if exist "%SCRIPT_DIR%env\Scripts\python.exe" (
    "%SCRIPT_DIR%env\Scripts\python.exe" "%SCRIPT_DIR%web_ui.py" %*
) else if defined PYTHON_EXEC (
    "%PYTHON_EXEC%" "%SCRIPT_DIR%web_ui.py" %*
) else (
    python "%SCRIPT_DIR%web_ui.py" %*
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo 运行退出或发生异常。按任意键退出...
    pause >nul
)
