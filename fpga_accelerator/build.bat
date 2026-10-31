@echo off
set VIVADO_BIN=C:\AMDDesignTools\2025.2\Vivado\bin\vivado.bat

if not exist "%VIVADO_BIN%" (
    echo [ERROR] Vivado not found at %VIVADO_BIN%
    echo Please verify AMD Vivado 2025.2 installation path.
    pause
    exit /b 1
)

echo ==========================================================
echo   BUILDING BITGEMM-S7 BITSTREAM (VIVADO BATCH MODE)
echo ==========================================================
call "%VIVADO_BIN%" -mode batch -notrace -source "%~dp0scripts\build_bitstream.tcl"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Synthesis or implementation failed. See logs above.
    pause
    exit /b %ERRORLEVEL%
)

echo [SUCCESS] Bitstream built at: %~dp0build\bitgemm_s7_arty.bit
