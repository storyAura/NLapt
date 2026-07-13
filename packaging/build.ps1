# NLapt GUI Windows build script (PyInstaller onedir, windowed).
# Run from the repository root:  powershell -ExecutionPolicy Bypass -File packaging/build.ps1

$ErrorActionPreference = "Stop"

# 1. Ensure PyInstaller is available.
python -m PyInstaller --version *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "PyInstaller 未安装。请先运行:" -ForegroundColor Yellow
    Write-Host "    pip install pyinstaller" -ForegroundColor Yellow
    exit 1
}

# 2. Build from the spec (onedir, console=False, name NLapt).
python -m PyInstaller packaging/nlapt.spec --noconfirm
if ($LASTEXITCODE -ne 0) {
    Write-Host "构建失败,请检查上方 PyInstaller 输出。" -ForegroundColor Red
    exit $LASTEXITCODE
}

# 3. Report the output location.
$outDir = Join-Path (Get-Location) "dist\NLapt"
Write-Host ""
Write-Host "构建完成。输出目录: $outDir" -ForegroundColor Green
Write-Host "可执行文件: $outDir\NLapt.exe"
