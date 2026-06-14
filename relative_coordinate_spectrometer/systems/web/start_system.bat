@echo off
cd /d "%~dp0"
where py >nul 2>&1
if %errorlevel% equ 0 (
  py -3 "%~dp0start_system.py" %*
) else (
  python "%~dp0start_system.py" %*
)
pause
