@echo off
chcp 65001 >nul
title Sydney (MiniCPM5-2B + LoRA v2 Core)
echo ============================================================
echo   正在启动 Sydney 官方交互终端...
echo ============================================================

set SCRIPT_DIR=%~dp0

REM 优先按顺序检索标准相对虚拟环境路径与环境变量，绝不硬编码任何机器专属绝对路径
if exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
    "%SCRIPT_DIR%.venv\Scripts\python.exe" "%SCRIPT_DIR%run_sydney.py" %*
) else if exist "%SCRIPT_DIR%venv\Scripts\python.exe" (
    "%SCRIPT_DIR%venv\Scripts\python.exe" "%SCRIPT_DIR%run_sydney.py" %*
) else if exist "%SCRIPT_DIR%env\Scripts\python.exe" (
    "%SCRIPT_DIR%env\Scripts\python.exe" "%SCRIPT_DIR%run_sydney.py" %*
) else if defined PYTHON_EXEC (
    "%PYTHON_EXEC%" "%SCRIPT_DIR%run_sydney.py" %*
) else (
    python "%SCRIPT_DIR%run_sydney.py" %*
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo 运行退出或发生异常。按任意键退出...
    pause >nul
)
