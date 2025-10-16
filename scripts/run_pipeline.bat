@echo off
REM RAG Pipeline Runner Batch Script
REM Chạy pipeline RAG để xử lý PDF và tạo embeddings

echo 🚀 Chạy RAG Pipeline - Xử lý PDF và tạo Embeddings
echo ====================================================

REM Kiểm tra virtual environment
if not exist ".venv\Scripts\activate.bat" (
    echo ❌ Không tìm thấy virtual environment tại .venv
    echo Vui lòng chạy: python -m venv .venv
    pause
    exit /b 1
)

REM Kích hoạt virtual environment
call .venv\Scripts\activate.bat

REM Thiết lập PYTHONPATH
set PYTHONPATH=%CD%

REM Chạy pipeline
echo 🔧 Đang khởi tạo pipeline...
python scripts\run_pipeline.py

REM Deactivate virtual environment
call deactivate

echo.
echo ✅ Đã hoàn thành!
pause