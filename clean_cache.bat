@echo off
echo ================================
echo  XOA SACH __pycache__ va *.pyc
echo ================================

:: Xoa tat ca file .pyc
del /s /q *.pyc

:: Xoa tat ca thu muc __pycache__
for /d /r . %%d in (__pycache__) do (
    if exist "%%d" (
        echo Xoa thu muc: %%d
        rd /s /q "%%d"
    )
)

echo ================================
echo  DA XOA XONG TOAN BO CACHE
echo ================================
pause
