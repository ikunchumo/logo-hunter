# 推送到 GitHub（PowerShell）
#
# 前置：先在网页上创建一个空仓库（不要勾选 Add a README）
#       https://github.com/new?name=logo-hunter
#
# 用法：
#   .\push_to_github.ps1 -Token github_pat_xxxx
#   .\push_to_github.ps1 -Token github_pat_xxxx -Repo logo-hunter
#
# 说明：脚本会先用 token 查用户名，然后配置 remote 并推送，
#       推送完成后自动把 token 从 .git/config 里抹掉。

param(
    [Parameter(Mandatory = $true)][string]$Token,
    [string]$Repo = "logo-hunter",
    [string]$User = "",
    [string]$Email = "zhengsiyuanwangyi@163.com"
)

$ErrorActionPreference = "Stop"
if ($PSScriptRoot) { Set-Location $PSScriptRoot }

$headers = @{
    Authorization = "Bearer $Token"
    "User-Agent"  = "logo-hunter-deploy"
    Accept        = "application/vnd.github+json"
}

Write-Host "1/4 校验 token ..." -ForegroundColor Cyan
if (-not $User) {
    try {
        $me = Invoke-RestMethod -Uri "https://api.github.com/user" -Headers $headers -TimeoutSec 40
        $User = $me.login
    }
    catch {
        Write-Host "   token 无效或已过期" -ForegroundColor Red
        exit 1
    }
}
Write-Host "   用户：$User" -ForegroundColor Green

Write-Host "2/4 检查仓库是否存在 ..." -ForegroundColor Cyan
$exists = $false
try {
    Invoke-RestMethod -Uri "https://api.github.com/repos/$User/$Repo" -Headers $headers -TimeoutSec 40 | Out-Null
    $exists = $true
    Write-Host "   仓库已存在" -ForegroundColor Green
}
catch {
    Write-Host "   仓库不存在" -ForegroundColor Yellow
}

if (-not $exists) {
    Write-Host "   尝试自动创建 ..." -ForegroundColor Cyan
    $body = @{
        name        = $Repo
        description = "图片目标检测器：SIFT 特征匹配 + 可视化训练页"
        private     = $false
    } | ConvertTo-Json
    try {
        Invoke-RestMethod -Uri "https://api.github.com/user/repos" -Method Post `
            -Headers $headers -Body $body -ContentType "application/json" | Out-Null
        Write-Host "   创建成功" -ForegroundColor Green
    }
    catch {
        Write-Host ""
        Write-Host "   自动创建被拒（token 缺 Administration 权限）。" -ForegroundColor Yellow
        Write-Host "   请手动创建空仓库（30 秒）：" -ForegroundColor Yellow
        Write-Host "     https://github.com/new?name=$Repo" -ForegroundColor White
        Write-Host "   注意：不要勾选 Add a README / .gitignore / license" -ForegroundColor Yellow
        Write-Host "   建好后重新运行本脚本即可。" -ForegroundColor Yellow
        exit 1
    }
}

Write-Host "3/4 配置 remote ..." -ForegroundColor Cyan
if (-not (Test-Path ".git")) { git init -b main | Out-Null }
git config user.name $User
git config user.email $Email
git remote remove origin 2>$null
git remote add origin "https://${User}:${Token}@github.com/${User}/${Repo}.git"
git branch -M main

Write-Host "4/4 推送 ..." -ForegroundColor Cyan
$env:GIT_TERMINAL_PROMPT = "0"
git push -u origin main 2>&1 | ForEach-Object { Write-Host "   $_" }

if ($LASTEXITCODE -eq 0) {
    git remote set-url origin "https://github.com/${User}/${Repo}.git"
    Write-Host ""
    Write-Host "完成：https://github.com/$User/$Repo" -ForegroundColor Green
    Write-Host "token 已从 .git/config 移除" -ForegroundColor DarkGray
}
else {
    Write-Host "推送失败：检查网络，或确认仓库是空的" -ForegroundColor Red
    git remote set-url origin "https://github.com/${User}/${Repo}.git"
}
