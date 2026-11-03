@echo off
cd /d "%~dp0"
set VIVADO_BIN=C:\AMDDesignTools\2025.2\Vivado\bin\vivado.bat

if not exist "%VIVADO_BIN%" (
    echo [ERROR] Vivado not found at %VIVADO_BIN%
    pause
    exit /b 1
)

echo ==========================================================
echo   PROGRAMMING ARTY S7-25 OVER JTAG
echo ==========================================================
call "%VIVADO_BIN%" -mode batch -notrace -source "%~dp0scripts\program_fpga.tcl"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] JTAG programming failed.
    echo Please make sure Arty S7 is plugged into USB and powered on.
    pause
    exit /b %ERRORLEVEL%
)

echo [SUCCESS] Arty S7-25 is running!
