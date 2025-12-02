@echo off
setlocal EnableDelayedExpansion

:: --- RUN THIS SCRIPT TO FIND THE PATH ERROR ---

if "%~1"=="" (
    echo DRAG AND DROP YOUR DATA FOLDER ONTO THIS SCRIPT
    pause
    exit /b
)

set "ROOT_DIR=%~1"
:: REMOVE QUOTES if the user added them in the argument
set "ROOT_DIR=%ROOT_DIR:"=%"
:: REMOVE TRAILING SLASH if present
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"

echo ==========================================
echo DIAGNOSTIC MODE
echo ROOT: [%ROOT_DIR%]
echo ==========================================

for /d %%D in ("%ROOT_DIR%\ip*") do (
    set "FULL_IP=%%~fD"
    set "DIR_NAME=%%~nD"
    
    :: Perform the replacement
    set "NUM=!DIR_NAME:ip=!"
    
    :: Construct the OP path
    set "EXPECTED_OP=%ROOT_DIR%\op!NUM!"
    
    echo.
    echo ---------------------------------------
    echo FOUND INPUT FOLDER:  [!DIR_NAME!]
    echo EXTRACTED NUMBER:    [!NUM!]
    echo.
    echo THE COMPUTER IS LOOKING FOR EXACTLY THIS:
    echo "[!EXPECTED_OP!]"
    echo.
    
    if exist "!EXPECTED_OP!" (
        echo STATUS: [FOUND] - The path is valid.
    ) else (
        echo STATUS: [NOT FOUND] - Windows cannot see this folder.
        echo.
        echo CHECK THESE COMMON ERRORS:
        echo 1. Does [!EXPECTED_OP!] actually exist?
        echo 2. Is there a space in the input name? (e.g. "ip 1" vs "ip1")
        echo 3. Is there a leading zero? (e.g. "ip01" needs "op01")
    )
)

echo.
echo ---------------------------------------
pause