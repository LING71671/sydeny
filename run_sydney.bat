@echo off
chcp 65001 >nul
title Sydney (MiniCPM5-2B + LoRA v2 Core)
echo ============================================================
echo   正在启动 Sydney 官方交互终端...
echo ============================================================

if exist "A:\DevEnv\Envs\lf-minicpm5\Scripts\python.exe" (
    "A:\DevEnv\Envs\lf-minicpm5\Scripts\python.exe" "%~dp0run_sydney.py" %*
) else (
    python "%~dp0run_sydney.py" %*
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo 运行退出或发生异常。按任意键退出...
    pause >nul
)
