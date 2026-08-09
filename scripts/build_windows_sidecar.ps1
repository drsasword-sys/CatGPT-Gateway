$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

python -m pip install --upgrade pip
python -m pip install -r requirements.txt pyinstaller

$browserRuntime = Join-Path $repoRoot "build_browser_runtime"
$env:PLAYWRIGHT_BROWSERS_PATH = $browserRuntime
python -m patchright install chromium

python -m PyInstaller catgpt-sidecar.spec --noconfirm --clean

$sidecarDir = Join-Path $repoRoot "dist\CatGPTGateway"
$bundledBrowser = Join-Path $sidecarDir "browser-runtime"
Copy-Item -Path $browserRuntime -Destination $bundledBrowser -Recurse -Force

$archive = Join-Path $repoRoot "dist\CatGPTGateway-Windows-x64-beta.zip"
if (Test-Path $archive) {
    Remove-Item $archive -Force
}
Compress-Archive -Path "$sidecarDir\*" -DestinationPath $archive -CompressionLevel Optimal
Write-Host "Built $archive"
