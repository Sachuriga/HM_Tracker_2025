@echo off
setlocal EnableDelayedExpansion

:: ================= CONFIGURATION =================
:: UPDATE: Ensure these paths exist on your machine
set "TRODES_BIN=C:\Users\gl_pc\Desktop\Track\trodes\trodesexport.exe"
set "ONNX_WEIGHTS_PATH=C:\Users\gl_pc\Desktop\track\yolov3_training_best.onnx"

set MAX_CPU_LOAD=95
set MAX_GPU_LOAD=95
set FREQ=30000
set RAMP_UP_DELAY=10
:: =================================================

:: 1. Check if Root Directory is provided
if "%~1"=="" (
    echo Usage: runner_windows.bat "C:\Users\gl_pc\Desktop\Track_2021"
    exit /b 1
)

:: Clean the path: Remove quotes and trailing backslashes if present
set "ROOT_DIR=%~1"
if "!ROOT_DIR:~-1!"=="\" set "ROOT_DIR=!ROOT_DIR:~0,-1!"

echo Scanning: "%ROOT_DIR%"
echo ---------------------------------------------------

:: ================= MAIN LOOP =================
:: Loop through all directories starting with "ip"
for /d %%D in ("%ROOT_DIR%\ip*") do (
    set "IP_PATH=%%~fD"
    set "DIR_NAME=%%~nD"
    
    :: Logic: Take folder name "ip1", remove "ip", get "1"
    set "NUM=!DIR_NAME:ip=!"
    
    :: Construct the expected Output Path
    set "OP_PATH=!ROOT_DIR!\op!NUM!"

    :: === DEBUG PRINT: SHOW EXACTLY WHAT WE FOUND ===
    echo Found Input: [!DIR_NAME!]
    echo Looking for: [!OP_PATH!]

    if exist "!OP_PATH!" (
        echo - Status: FOUND PAIR. Queuing job...
        
        :: === PRE-LAUNCH RESOURCE CHECK LOOP ===
        :CHECK_RESOURCES
        call :GET_CPU_USAGE CURRENT_CPU
        call :GET_GPU_USAGE CURRENT_GPU
        
        cls
        echo ---------------------------------------------------
        echo Monitor Active on: "%ROOT_DIR%"
        echo Job Queue: !DIR_NAME! -^> op!NUM!
        echo System Load: CPU !CURRENT_CPU!%% ^| GPU !CURRENT_GPU!%%
        echo ---------------------------------------------------

        if !CURRENT_CPU! LSS %MAX_CPU_LOAD% (
            if !CURRENT_GPU! LSS %MAX_GPU_LOAD% (
                goto :LAUNCH_JOB
            )
        )
        
        timeout /t 10 /nobreak >nul
        goto :CHECK_RESOURCES

        :LAUNCH_JOB
        start "Job: !DIR_NAME!" cmd /c "%~f0" :WORKER "!IP_PATH!" "!OP_PATH!"
        
        :: Ramp up delay to let the new window start using resources
        timeout /t %RAMP_UP_DELAY% /nobreak >nul
        
    ) else (
        echo - Status: NOT FOUND.
        echo   (Make sure '!OP_PATH!' matches the number format of '!DIR_NAME!')
        echo ---------------------------------------------------
    )
)

echo.
echo All valid pairs queued.
pause
exit /b 0

:: =======================================================
::                   WORKER PROCESS
:: =======================================================
:WORKER
if "%1"==":WORKER" (
    set "IP=%~2"
    set "OP=%~3"
    set "LOG_FILE=%~3\pipeline_log.txt"

    if not exist "%~3" mkdir "%~3"
    
    (
        echo ================= STARTING PIPELINE %DATE% %TIME% =================
        echo Processing: %IP%

        :: === Extract DIO ===
        for /r "%IP%" %%f in (*.rec) do (
            echo --- RUNNING TRODES: %%~nxf ---
            "%TRODES_BIN%" -dio -rec "%%f"
        )

        :: === Run Sync Script ===
        echo --- RUNNING LED SYNC ---
        python -u ./src/Video_LED_Sync_using_ICA.py -i "%IP%" -o "%OP%" -f %FREQ%
        if errorlevel 1 goto :ErrorExit

        :: === Stitch Step ===
        echo --- RUNNING STITCHING ---
        python -u ./src/join_views.py "%IP%"

        :: === Tracking ===
        if exist "%IP%\stitched.mp4" (
            echo --- RUNNING YOLO TRACKER ---
            python -u ./src/TrackerYolov.py --input_folder "%IP%\stitched.mp4" --output_folder "%OP%" --onnx_weight "%ONNX_WEIGHTS_PATH%"
            if errorlevel 1 goto :ErrorExit
        ) else (
            echo [Warning] stitched.mp4 not found. Skipping tracking.
        )

        :: === Plotting ===
        echo --- RUNNING PLOTTING ---
        python -u plot_trials.py -o "%OP%"
        if errorlevel 1 goto :ErrorExit

        :: === Compression ===
        :: FIX: Use %%v instead of undefined !VIDEO_FILE!
        for %%v in ("%OP%\*.mp4") do (
            echo --- RUNNING COMPRESSION on %%~nxv ---
            ffmpeg -y -i "%%v" -vcodec libx264 -crf 28 "%OP%\__temp_compressed.mp4"
            if exist "%OP%\__temp_compressed.mp4" (
                move /y "%OP%\__temp_compressed.mp4" "%%v"
            )
            goto :CompDone
        )
        :CompDone

        echo ================= FINISHED %DATE% %TIME% =================
        echo You may close this window now.
        exit
        
        :ErrorExit
        echo !!! PIPELINE FAILED !!!
        echo Check the log above for details.

    ) >> "!LOG_FILE!" 2>&1

    :: Display Log to Screen
    type "!LOG_FILE!"
    pause
    exit
)

:: ================= HELPER FUNCTIONS =================
:GET_CPU_USAGE
for /f "skip=1" %%p in ('wmic cpu get loadpercentage') do ( 
    set %1=%%p
    goto :eof
)
goto :eof

:GET_GPU_USAGE
where nvidia-smi >nul 2>nul
if %errorlevel% neq 0 (
    set %1=0
    goto :eof
)
for /f "tokens=1 delims=," %%g in ('nvidia-smi --query-gpu^=utilization.gpu --format^=csv^,noheader^,nounits') do (
    set %1=%%g
    goto :eof
)
set %1=0
goto :eof