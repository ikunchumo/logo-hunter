# 推送到 GitHub（PowerShell）
#
# 用法：
#   .\push_to_github.ps1 -User <你的GitHub用户名> -Token <PAT>
#
# 例：
#   .\push_to_github.ps1 -User zhengsiyuanwangyi -Token ghp_xxxxxxxxxxxx
#
# 注意：GitHub 从 2021 年起不再接受账号密码，必须用 Personal Access Token。
# 生成方法见 README 的「发布到 GitHub」一节。

param(
    [Parameter(Mandatory = $true)][string]$User,
    [Parameter(Mandatory = $true)][string]$Token,
    [string]$Repo = "logo-hunter",
    [string]$Email = "zhengsiyuanwangyi@163.com"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".git")) {
    Write-Host "当前目录不是 git 仓库，先运行 git init" -ForegroundColor Red
    exit 1
}

Write-Host "1/4 检查仓库 ..." -ForegroundColor Cyan
git rev-parse --verify HEAD | Out-Null
if ($LASTEXITCODE -ne 0) {
    git config user.name $User
    git config user.email $Email
    git add -A
    git commit -m "init: 图片目标检测器"
}

Write-Host "2/4 创建远程仓库（已存在则跳过）..." -ForegroundColor Cyan
$headers = @{
    Authorization          = "token $Token"
    "User-Agent"           = "logo-hunter-deploy"
    Accept                 = "application/vnd.github+json"
}
$body = @{
    name        = $Repo
    description = "图片目标检测器：SIFT 特征匹配 + 可视化训练页"
    private     = $false
    has_issues  = $true
} | ConvertTo-Json

try {
    Invoke-RestMethod -Uri "https://api.github.com/user/repos" -Method Post `
        -Headers $headers -Body $body -ContentType "application/json" | Out-Null
    Write-Host "   已创建 https://github.com/$User/$Repo" -ForegroundColor Green
}
catch {
    $msg = $_.Exception.Message
    if ($msg -match "422") {
        Write-Host "   仓库已存在，继续推送" -ForegroundColor Yellow
    }
    else {
        Write-Host "   创建失败：$msg" -ForegroundColor Red
        Write-Host "   检查 token 是否有 repo 权限" -ForegroundColor Red
        exit 1
    }
}

Write-Host "3/4 配置 remote ..." -ForegroundColor Cyan
$remote = "https://$User`:$Token@github.com/$User/$Repo.git"
git remote remove origin 2>$null
git remote add origin $remote
git branch -M main

Write-Host "4/4 推送 ..." -ForegroundColor Cyan
git push -u origin main 2>&1 | ForEach-Object { Write-Host "   $_" }

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "完成：https://github.com/$User/$Repo" -ForegroundColor Green
    # 把 token 从 remote URL 里抹掉，避免留在配置里
    git remote set-url origin "https://github.com/$User/$Repo.git"
    Write-Host "已把 token 从 git 配置里移除" -ForegroundColor DarkGray
}
else {
    Write-Host "推送失败，检查网络或 token 权限" -ForegroundColor Red
}
