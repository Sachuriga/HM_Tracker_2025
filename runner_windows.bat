@echo off
setlocal EnableDelayedExpansion

:: ================= CONFIGURATION =================
set "TRODES_BIN=C:\Users\gl_pc\Desktop\Track\trodes\trodesexport.exe"
set "ONNX_WEIGHTS_PATH=C:\Users\gl_pc\Desktop\track\yolov3_training_best.onnx"
set FREQ=30000
:: =================================================

:: Check arguments
if "%~1"=="" (
    echo Usage: runner_direct.bat "C:\Users\gl_pc\Desktop\Track_2021"
    exit /b 1
)

:: Clean path
set "ROOT_DIR=%~1"
if "!ROOT_DIR:~-1!"=="\" set "ROOT_DIR=!ROOT_DIR:~0,-1!"

echo Scanning: "%ROOT_DIR%"

:: ================= MAIN LOOP =================
for /d %%D in ("%ROOT_DIR%\ip*") do (
    set "IP_PATH=%%~fD"
    set "DIR_NAME=%%~nD"
    set "NUM=!DIR_NAME:ip=!"
    set "OP_PATH=!ROOT_DIR!\op!NUM!"

    :: Check if OP folder exists using pushd (safest method)
    pushd "!OP_PATH!" 2>nul
    if not errorlevel 1 (
        popd
        echo [MATCH] Found pair: !DIR_NAME! -^> op!NUM!
        
        :: LAUNCH IMMEDIATELY (No CPU Check)
        :: We use 'start' to open a new window.
        start "Job: !DIR_NAME!" cmd /c "%~f0" :WORKER "!IP_PATH!" "!OP_PATH!"
        
        :: Wait 5 seconds to prevent opening 50 windows at once
        timeout /t 5 /nobreak >nul
        
    ) else (
        echo [SKIP]  !DIR_NAME! (No matching '!OP_PATH!')
    )
)

echo.
echo All jobs triggered.
pause
exit /b 0

:: ================= WORKER PROCESS =================
:WORKER
if "%1"==":WORKER" (
    set "IP=%~2"
    set "OP=%~3"
    set "LOG_FILE=%~3\pipeline_log.txt"
    
    if not exist "%~3" mkdir "%~3"

    :: Redirect all output to log, but also keep window open
    (
        echo ================= STARTING PIPELINE =================
        echo Time: %DATE% %TIME%
        echo Input: %IP%
        echo Output: %OP%
        echo.
        
        :: 1. CHECK PYTHON (Important for Conda)
        python --version
        if errorlevel 1 (
            echo !!! ERROR: Python not found. Anaconda environment might not be active in this window.
            goto :ErrorExit
        )

        :: 2. RUN TRODES
        echo --- RUNNING TRODES ---
        for /r "%IP%" %%f in (*.rec) do (
            echo Processing: %%~nxf
            "%TRODES_BIN%" -dio -rec "%%f"
        )

        :: 3. RUN SYNC
        echo --- RUNNING LED SYNC ---
        python -u ./src/Video_LED_Sync_using_ICA.py -i "%IP%" -o "%OP%" -f %FREQ%
        if errorlevel 1 goto :ErrorExit

        :: 4. RUN STITCHING
        echo --- RUNNING STITCHING ---
        python -u ./src/join_views.py "%IP%"

        :: 5. RUN TRACKER
        if exist "%IP%\stitched.mp4" (
            echo --- RUNNING YOLO TRACKER ---
            python -u ./src/TrackerYolov.py --input_folder "%IP%\stitched.mp4" --output_folder "%OP%" --onnx_weight "%ONNX_WEIGHTS_PATH%"
            if errorlevel 1 goto :ErrorExit
        ) else (
            echo [Warning] stitched.mp4 not found. Skipping tracking.
        )

        :: 6. PLOTTING
        echo --- RUNNING PLOTTING ---
        python -u plot_trials.py -o "%OP%"
        if errorlevel 1 goto :ErrorExit

        :: 7. COMPRESSION
        for %%v in ("%OP%\*.mp4") do (
            echo --- RUNNING COMPRESSION: %%~nxv ---
            ffmpeg -y -hide_banner -loglevel warning -i "%%v" -vcodec libx264 -crf 28 "%OP%\__temp_compressed.mp4"
            if exist "%OP%\__temp_compressed.mp4" (
                move /y "%OP%\__temp_compressed.mp4" "%%v"
            )
            goto :CompDone
        )
        :CompDone

        echo.
        echo ================= FINISHED =================
        echo You may close this window.
        
    ) >> "!LOG_FILE!" 2>&1

    :: Display the log on the screen so you can see what happened
    type "!LOG_FILE!"
    pause
    exit
    
    :ErrorExit
    echo.
    echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    echo !!!       JOB FAILED         !!!
    echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    echo.
    echo See log details above.
    pause
    exit
)