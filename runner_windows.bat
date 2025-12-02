@echo off
setlocal EnableDelayedExpansion

:: ================= CONFIGURATION =================
:: SET YOUR PATHS HERE (No spaces around the = sign)
set "TRODES_BIN=C:\Path\To\Trodes_2-3-2\trodesexport.exe"
set "ONNX_WEIGHTS_PATH=C:\Users\genzel\Desktop\Documents\Param\yolov3_training_best.onnx"

set MAX_CPU_LOAD=95
set MAX_GPU_LOAD=95
set FREQ=30000
set RAMP_UP_DELAY=10
:: =================================================

:: 1. Check if Root Directory is provided
if "%~1"=="" (
    echo Usage: batch_runner.bat "path_to_data_folder"
    exit /b 1
)
set "ROOT_DIR=%~1"

echo Scanning %ROOT_DIR% for ip/op pairs...
echo Main Monitor Running (Job details will appear in pop-up windows)...

:: ================= MAIN LOOP =================
:: Loop through all directories starting with "ip"
for /d %%D in ("%ROOT_DIR%\ip*") do (
    set "IP_PATH=%%~fD"
    set "DIR_NAME=%%~nD"
    
    :: Extract the number from "ipX" to find "opX"
    set "NUM=!DIR_NAME:ip=!"
    set "OP_PATH=%ROOT_DIR%\op!NUM!"

    if exist "!OP_PATH!" (
        
        :: === PRE-LAUNCH RESOURCE CHECK LOOP ===
        :CHECK_RESOURCES
        call :GET_CPU_USAGE CURRENT_CPU
        call :GET_GPU_USAGE CURRENT_GPU
        
        :: Print Status (uses <nul set /p to print on same line, sort of)
        cls
        echo Processing: !DIR_NAME!
        echo System Status: CPU !CURRENT_CPU!%% ^| GPU !CURRENT_GPU!%%
        echo Waiting for resources...

        if !CURRENT_CPU! LSS %MAX_CPU_LOAD% (
            if !CURRENT_GPU! LSS %MAX_GPU_LOAD% (
                goto :LAUNCH_JOB
            )
        )
        
        :: Wait 10 seconds before checking again
        timeout /t 10 /nobreak >nul
        goto :CHECK_RESOURCES

        :LAUNCH_JOB
        echo Launching Job: !DIR_NAME!
        
        :: LAUNCH NEW WINDOW
        :: "start" opens a new window. 
        :: We pass arguments to THIS script but jump to the :WORKER label.
        start "Job: !DIR_NAME!" cmd /c "%~f0" :WORKER "!IP_PATH!" "!OP_PATH!"
        
        :: Ramp up delay
        timeout /t %RAMP_UP_DELAY% /nobreak >nul
        
    ) else (
        echo Skipping !DIR_NAME!: Corresponding !OP_PATH! not found.
    )
)

echo All pairs queued.
pause
exit /b 0

:: =======================================================
::                   WORKER PROCESS
:: This section runs inside the POP-UP window
:: =======================================================
:WORKER
if "%1"==":WORKER" (
    set "IP=%~2"
    set "OP=%~3"
    set "LOG_FILE=%~3\pipeline_log.txt"

    if not exist "%~3" mkdir "%~3"

    echo ^>^>^> PREPARING JOB: Pair !IP! -^> !OP! > "!LOG_FILE!"
    
    :: We use a block (parentheses) to redirect ALL output to log
    (
        echo ================= STARTING PIPELINE %DATE% %TIME% =================

        :: === Extract DIO ===
        for /r "%IP%" %%f in (*.rec) do (
            echo --- RUNNING TRODES: %%~nxf ---
            "%TRODES_BIN%" -dio -rec "%%f"
        )

        :: === Run Sync Script ===
        echo --- RUNNING LED SYNC ---
        python -u ./src/Video_LED_Sync_using_ICA.py -i "%IP%" -o "%OP%" -f %FREQ%
        if errorlevel 1 (
            echo !!! ERROR: LED SYNC FAILED !!!
            goto :EndWorker
        )

        :: === Stitch Step ===
        echo --- RUNNING STITCHING ---
        python -u ./src/join_views.py "%IP%"

        :: === Tracking ===
        if exist "%IP%\stitched.mp4" (
            echo --- RUNNING YOLO TRACKER ---
            python -u ./src/TrackerYolov.py --input_folder "%IP%\stitched.mp4" --output_folder "%OP%" --onnx_weight "%ONNX_WEIGHTS_PATH%"
            if errorlevel 1 (
                echo !!! ERROR: TRACKER FAILED !!!
                goto :EndWorker
            )
        ) else (
            echo [Warning] stitched.mp4 not found. Skipping tracking.
        )

        :: === Plotting ===
        echo --- RUNNING PLOTTING ---
        python -u plot_trials.py -o "%OP%"
        if errorlevel 1 (
            echo !!! ERROR: PLOTTING FAILED !!!
            goto :EndWorker
        )

        :: === Compression ===
        for %%v in ("%OP%\*.mp4") do (
            echo --- RUNNING COMPRESSION ---
            ffmpeg -y -hide_banner -loglevel warning -i "%%v" -vcodec h264_nvenc -qp 30 "%OP%\__temp_compressed.mp4"
            if exist "%OP%\__temp_compressed.mp4" (
                move /y "%OP%\__temp_compressed.mp4" "%%v"
            )
            :: Break after first video found (matches original script logic)
            goto :CompDone
        )
        :CompDone

        echo ================= FINISHED %DATE% %TIME% =================
        echo You may close this window now.

    ) >> "!LOG_FILE!" 2>&1

    :: Also tail the log to the screen so user sees it
    type "!LOG_FILE!"
    
    :EndWorker
    pause
    exit
)

:: ================= HELPER FUNCTIONS =================

:GET_CPU_USAGE
:: Parse WMIC output to get CPU Load
for /f "skip=1" %%p in ('wmic cpu get loadpercentage') do ( 
    set %1=%%p
    goto :eof
)
goto :eof

:GET_GPU_USAGE
:: Parse NVIDIA-SMI output
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