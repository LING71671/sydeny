# ==============================================================================
# Sydney (MiniCPM5-2B + LoRA v2 Core) PowerShell 一键启动脚本
# ==============================================================================
$ErrorActionPreference = 'Continue'
$scriptPath = Join-Path $PSScriptRoot "run_sydney.py"
$localPython = "A:\DevEnv\Envs\lf-minicpm5\Scripts\python.exe"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  正在启动 Sydney 官方交互终端..." -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

if (Test-Path -LiteralPath $localPython) {
    & $localPython $scriptPath @args
} else {
    python $scriptPath @args
}
