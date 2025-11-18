<#
PowerShell helper to backup and push `data/` folder to git with Git LFS support.
Run from repository root.
Usage:
  .\scripts\push_data_to_git.ps1 -CommitMessage "Add data and metadata files"
#>
param(
    [string]$CommitMessage = "Add data and metadata files",
    [switch]$DryRun
)

Write-Host "Running push helper for data/ - ensure you're in the repo root" -ForegroundColor Cyan

# 1) Backup data folder to zip
$timestamp = (Get-Date).ToString('yyyyMMdd_HHmmss')
$backupName = "data_backup_$timestamp.zip"
Write-Host "Creating backup archive: $backupName" -ForegroundColor Yellow
Compress-Archive -Path "data\*" -DestinationPath $backupName -Force

# 2) Ensure git-lfs is installed and initialized
Write-Host "Initializing Git LFS (git lfs install)" -ForegroundColor Yellow
git lfs install

# 3) Show git status
Write-Host "Current git status:" -ForegroundColor Cyan
git status --porcelain

if ($DryRun) {
    Write-Host "DryRun mode - no changes staged or pushed." -ForegroundColor Magenta
    exit 0
}

# 4) Stage .gitattributes and .gitignore (we edited these to allow tracking data/vectors and metadata)
Write-Host "Staging .gitattributes and .gitignore" -ForegroundColor Yellow
git add .gitattributes .gitignore

# 5) Stage metadata, batch summaries and vectors (LFS handles large files via .gitattributes)
Write-Host "Staging data/ metadata and batch summaries" -ForegroundColor Yellow
# Stage metadata and batch_summary explicitly
git add data/metadata -v
git add data/batch_summary_*.json -v

# Stage vectors and embeddings (may be large; tracked by LFS per .gitattributes)
Write-Host "Staging data/vectors and data/embeddings (may be large)" -ForegroundColor Yellow
git add data/vectors -v
git add data/embeddings -v

# 6) Commit
Write-Host "Committing changes with message: $CommitMessage" -ForegroundColor Cyan
git commit -m "$CommitMessage"

# 7) Push to current branch
Write-Host "Pushing to remote (current branch)" -ForegroundColor Cyan
git push

Write-Host "Done. Backup stored as $backupName" -ForegroundColor Green
