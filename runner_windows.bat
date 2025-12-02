@echo off
setlocal EnableDelayedExpansion

:: ================= CONFIGURATION =================
set "TRODES_BIN=C:\Users\gl_pc\Desktop\Track\trodes\trodesexport.exe"
set "ONNX_WEIGHTS_PATH=C:\Users\gl_pc\Desktop\track\yolov3_training_best.onnx"

set MAX_CPU_LOAD=95
set MAX_GPU_LOAD=95
set FREQ=30000
set RAMP_UP_DELAY=10
:: =================================================

if "%~1"=="" (
    echo Usage: runner.bat "C:\Your\Data\Path"
    exit /b 1
)

:: Clean path and remove trailing backslash if present
set "ROOT_DIR=%~1"
if "!ROOT_DIR:~-1!"=="\" set "ROOT_DIR=!ROOT_DIR:~0,-1!"

echo Scanning: "%ROOT_DIR%"

:: ================= MAIN LOOP =================
for /d %%D in ("%ROOT_DIR%\ip*") do (
    set "IP_PATH=%%~fD"
    set "DIR_NAME=%%~nD"
    set "NUM=!DIR_NAME:ip=!"
    set "OP_PATH=!ROOT_DIR!\op!NUM!"

    :: Method: Try to actually enter the directory to prove it exists
    pushd "!OP_PATH!" 2>nul
    if not errorlevel 1 (
        popd
        
        :: === FOLDER FOUND ===
        :CHECK_RESOURCES
        :: Reset variables to 0 to prevent syntax crashes
        set CURRENT_CPU=0
        set CURRENT_GPU=0
        
        call :GET_CPU_USAGE CURRENT_CPU
        call :GET_GPU_USAGE CURRENT_GPU
        
        cls
        echo ---------------------------------------------------
        echo Processing: !DIR_NAME!
        echo Path: !OP_PATH!
        echo System Status: CPU !CURRENT_CPU!%% ^| GPU !CURRENT_GPU!%%
        echo ---------------------------------------------------

        :: Math Check (Safe against empty vars)
        if !CURRENT_CPU! LSS %MAX_CPU_LOAD% (
            if !CURRENT_GPU! LSS %MAX_GPU_LOAD% (
                goto :LAUNCH_JOB
            )
        )
        
        timeout /t 5 /nobreak >nul
        goto :CHECK_RESOURCES

        :LAUNCH_JOB
        echo Launching Job...
        start "Job: !DIR_NAME!" cmd /c "%~f0" :WORKER "!IP_PATH!" "!OP_PATH!"
        timeout /t %RAMP_UP_DELAY% /nobreak >nul

    ) else (
        :: === FOLDER NOT FOUND ===
        echo [Skipping] !DIR_NAME!: Could not find folder "!OP_PATH!"
    )
)

echo All pairs queued.
pause
exit /b 0

:: ================= WORKER PROCESS =================
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
        for %%v in ("%OP%\*.mp4") do (
            echo --- RUNNING COMPRESSION on %%~nxv ---
            ffmpeg -y -hide_banner -loglevel warning -i "%%v" -vcodec libx264 -crf 28 "%OP%\__temp_compressed.mp4"
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
    
    type "!LOG_FILE!"
    pause
    exit
)

:: ================= HELPER FUNCTIONS =================
:GET_CPU_USAGE
:: Initialize to 0 so we never return empty
set %1=0
for /f "skip=1" %%p in ('wmic cpu get loadpercentage 2^>nul') do ( 
    set %1=%%p
    goto :eof
)
goto :eof

:GET_GPU_USAGE
set %1=0
where nvidia-smi >nul 2>nul
if %errorlevel% neq 0 goto :eof
for /f "tokens=1 delims=," %%g in ('nvidia-smi --query-gpu^=utilization.gpu --format^=csv^,noheader^,nounits') do (
    set %1=%%g
    goto :eof
)
goto :eof