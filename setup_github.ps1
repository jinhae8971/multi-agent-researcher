[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding            = [System.Text.Encoding]::UTF8
chcp 65001 | Out-Null

$GH_USER    = "jinhae8971"
$GH_REPO    = "multi-agent-researcher"
$GH_TOKEN   = "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"
$REMOTE_URL = "https://$GH_TOKEN@github.com/$GH_USER/$GH_REPO.git"
$API_HDR    = @{
    "Authorization" = "token $GH_TOKEN"
    "Accept"        = "application/vnd.github+json"
    "User-Agent"    = "MultiAgentResearcher"
}
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# [1] Git safe dir + init + remote
git config --global --add safe.directory ($ScriptDir -replace '\\','/') 2>$null
if (-not (Test-Path ".git")) { git init | Out-Null }
$prev = $ErrorActionPreference; $ErrorActionPreference = "SilentlyContinue"
git remote remove origin 2>$null | Out-Null
$ErrorActionPreference = $prev
git remote add origin $REMOTE_URL
git config user.name $GH_USER; git config user.email "jinhae8971@gmail.com"
Write-Host "[1] Git OK" -ForegroundColor Green

# [2] GitHub repo (private)
try {
    Invoke-RestMethod -Uri "https://api.github.com/repos/$GH_USER/$GH_REPO" -Headers $API_HDR | Out-Null
    Write-Host "[2] Repo exists" -ForegroundColor Green
} catch {
    try {
        Invoke-RestMethod -Method Post -Uri "https://api.github.com/user/repos" -Headers $API_HDR `
            -Body (@{name=$GH_REPO;private=$true;auto_init=$false} | ConvertTo-Json) `
            -ContentType "application/json" | Out-Null
        Write-Host "[2] Repo created (private)" -ForegroundColor Green; Start-Sleep -Seconds 2
    } catch {
        Write-Host "[2] Create manually: https://github.com/new (name: $GH_REPO, private)" -ForegroundColor Red
        Read-Host "Press Enter after creating repo"
    }
}

# [3] Commit & Push
$ErrorActionPreference = "SilentlyContinue"
git add .; git commit -m "feat: multi-agent researcher initial deploy" 2>$null
if ($LASTEXITCODE -ne 0) { git commit --allow-empty -m "chore: update" 2>$null }
git branch -M main; git push -u origin main --force 2>$null
$pushCode = $LASTEXITCODE; $ErrorActionPreference = "Stop"
if ($pushCode -ne 0) {
    Write-Host "PUSH FAILED. Token needs 'repo' scope: https://github.com/settings/tokens/new" -ForegroundColor Red
    exit 1
}
Write-Host "[3] Push OK" -ForegroundColor Green

# [4] GitHub Secrets
$secrets = @{
    ANTHROPIC_API_KEY = "sk-ant-YOUR-KEY-HERE"
    TAVILY_API_KEY    = "tvly-YOUR-KEY-HERE"
    TELEGRAM_TOKEN    = "8481005106:AAESmINZyjDHrbno69EVB6kSMSjWyG_dyCU"
    TELEGRAM_CHAT_ID  = "954137156"
}
if (Get-Command gh -ErrorAction SilentlyContinue) {
    $env:GH_TOKEN = $GH_TOKEN
    foreach ($s in $secrets.GetEnumerator()) {
        gh secret set $s.Key --body $s.Value --repo "$GH_USER/$GH_REPO" 2>$null
    }
    Write-Host "[4] Secrets set via gh CLI" -ForegroundColor Green
} else {
    Write-Host "[4] Set Secrets manually:" -ForegroundColor Red
    Write-Host "  https://github.com/$GH_USER/$GH_REPO/settings/secrets/actions" -ForegroundColor White
    foreach ($s in $secrets.GetEnumerator()) {
        Write-Host "  $($s.Key) = $($s.Value)" -ForegroundColor Cyan
    }
    Read-Host "Press Enter after adding Secrets"
}

# [5] Enable GitHub Pages (docs/ folder on main branch)
try {
    Invoke-RestMethod -Method Post `
        -Uri "https://api.github.com/repos/$GH_USER/$GH_REPO/pages" `
        -Headers $API_HDR `
        -Body '{"source":{"branch":"main","path":"/docs"}}' `
        -ContentType "application/json" | Out-Null
    Write-Host "[5] GitHub Pages enabled (docs/)" -ForegroundColor Green
} catch {
    Write-Host "[5] Pages setup: https://github.com/$GH_USER/$GH_REPO/settings/pages" -ForegroundColor White
    Write-Host "     Source: Deploy from branch > main > /docs" -ForegroundColor White
}

# [6] Trigger test
try {
    Invoke-RestMethod -Method Post `
        -Uri "https://api.github.com/repos/$GH_USER/$GH_REPO/actions/workflows/research.yml/dispatches" `
        -Headers $API_HDR `
        -Body '{"ref":"main","inputs":{"topic":"AI semiconductor market outlook","domain":"tech"}}' `
        -ContentType "application/json" | Out-Null
    Write-Host "[6] Test triggered! Check in ~5 min" -ForegroundColor Green
} catch {
    Write-Host "[6] Manual test: https://github.com/$GH_USER/$GH_REPO/actions" -ForegroundColor White
}

Write-Host ""
Write-Host "=== SETUP COMPLETE ===" -ForegroundColor Cyan
Write-Host "Dashboard: https://$GH_USER.github.io/$GH_REPO/" -ForegroundColor White
Write-Host ""
Write-Host "IMPORTANT: Update these Secrets with real values:" -ForegroundColor Yellow
Write-Host "  1. ANTHROPIC_API_KEY -> Your Anthropic API key" -ForegroundColor Yellow
Write-Host "  2. TAVILY_API_KEY    -> Get free key at https://tavily.com" -ForegroundColor Yellow
