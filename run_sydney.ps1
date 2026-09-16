# ==============================================================================
# Sydney (MiniCPM5-2B + LoRA) PowerShell 一键启动脚本
# ==============================================================================
$ErrorActionPreference = 'Continue'
$scriptPath = Join-Path $PSScriptRoot "run_sydney.py"

# 检测候选 Python 解释器（按标准优先级检索标准相对路径与环境变量）
$candidates = @(
    $env:PYTHON_EXEC,
    (Join-Path $PSScriptRoot ".venv\Scripts\python.exe"),
    (Join-Path $PSScriptRoot "venv\Scripts\python.exe"),
    (Join-Path $PSScriptRoot "env\Scripts\python.exe")
)

$pythonExe = "python"
foreach ($cand in $candidates) {
    if ($cand -and (Test-Path -LiteralPath $cand)) {
        $pythonExe = $cand
        break
    }
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  正在启动 Sydney 官方交互终端..." -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

& $pythonExe $scriptPath @args
