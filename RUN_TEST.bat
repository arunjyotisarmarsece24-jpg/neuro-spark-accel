@echo off
cd /d "%~dp0fpga_accelerator"
echo ========================================================
echo   Running Arty S7-25 Hardware Verification Test
echo ========================================================
python host\test_arty_s7.py
echo.
pause
