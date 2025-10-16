# RAG Pipeline Runner PowerShell Script
# Chạy pipeline RAG để xử lý PDF và tạo embeddings

param(
    [switch]$Help,
    [string]$Model = "GEMMA"
)

if ($Help) {
    Write-Host "RAG Pipeline Runner" -ForegroundColor Cyan
    Write-Host "==================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Chạy pipeline RAG để xử lý PDF và tạo embeddings" -ForegroundColor White
    Write-Host ""
    Write-Host "Usage:" -ForegroundColor Yellow
    Write-Host "  .\scripts\run_pipeline.ps1" -ForegroundColor White
    Write-Host "  .\scripts\run_pipeline.ps1 -Model BGE_M3" -ForegroundColor White
    Write-Host ""
    Write-Host "Parameters:" -ForegroundColor Yellow
    Write-Host "  -Model    Embedding model (GEMMA hoặc BGE_M3, mặc định: GEMMA)" -ForegroundColor White
    Write-Host "  -Help     Hiển thị trợ giúp này" -ForegroundColor White
    exit 0
}

Write-Host "🚀 Chạy RAG Pipeline - Xử lý PDF và tạo Embeddings" -ForegroundColor Magenta
Write-Host ("=" * 70) -ForegroundColor Magenta

# Kiểm tra virtual environment
$venvPath = ".venv\Scripts\activate.bat"
if (-not (Test-Path $venvPath)) {
    Write-Host "❌ Không tìm thấy virtual environment tại .venv" -ForegroundColor Red
    Write-Host "Vui lòng chạy: python -m venv .venv" -ForegroundColor Yellow
    Read-Host "Nhấn Enter để thoát"
    exit 1
}

# Kích hoạt virtual environment
Write-Host "🔧 Kích hoạt virtual environment..." -ForegroundColor Yellow
& $venvPath

# Thiết lập PYTHONPATH
$env:PYTHONPATH = Get-Location

# Chạy pipeline
Write-Host "📁 Đang chạy pipeline..." -ForegroundColor Green
try {
    & python scripts/run_pipeline.py
    Write-Host "`n✅ Pipeline hoàn thành thành công!" -ForegroundColor Green
} catch {
    Write-Host "`n❌ Lỗi khi chạy pipeline: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
} finally {
    # Deactivate virtual environment
    & deactivate
}

Write-Host "`n🎉 Đã hoàn thành!" -ForegroundColor Cyan
Read-Host "Nhấn Enter để thoát"