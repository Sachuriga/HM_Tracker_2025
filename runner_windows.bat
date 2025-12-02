<#
.SYNOPSIS
    Windows PowerShell equivalent of the batch runner pipeline.
.DESCRIPTION
    Scans for ip/op folders, monitors CPU/GPU usage, and launches jobs in new windows.
.PARAMETER RootDir
    The path to the data folder containing ip* directories.
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$RootDir
)

# ================= CONFIGURATION =================
# CRITICAL: Update these paths for your Windows machine!
$TRODES_BIN = "C:\Path\To\Trodes_2-3-2\trodesexport.exe" 
$ONNX_WEIGHTS_PATH = "C:\Users\genzel\Desktop\Documents\Param\yolov3_training_best.onnx"

$MAX_CPU_LOAD = 80
$MAX_GPU_LOAD = 80
$FREQ = 20000
$RAMP_UP_DELAY = 10 
# =================================================

# Ensure RootDir is valid
if (-not (Test-Path -Path $RootDir)) {
    Write-Host "Error: Directory '$RootDir' does not exist." -ForegroundColor Red
    exit 1
}

# ================= FUNCTIONS =================

function Get-CpuUsage {
    # Uses WMI to get current CPU load percentage
    $cpu = Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average
    return [math]::Round($cpu.Average)
}

function Get-GpuUsage {
    if (Get-Command "nvidia-smi" -ErrorAction SilentlyContinue) {
        try {
            # Query NVIDIA SMI, parse CSV output
            $gpu = nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits
            # Take the first GPU found
            $gpu = $gpu -split "`r`n" | Select-Object -First 1
            return [int]$gpu
        }
        catch {
            return 0
        }
    }
    return 0
}

# This block defines the code that will run inside the pop-up windows
$WorkerScriptBlock = {
    param($Ip, $Op, $Freq, $TrodesBin, $OnnxPath)

    $ErrorActionPreference = "Stop"
    $LogFile = Join-Path -Path $Op -ChildPath "pipeline_log.txt"
    
    # Create Output Directory
    New-Item -ItemType Directory -Force -Path $Op | Out-Null

    # Start logging to file AND console (Transcripts are PowerShell's "tail -f" equivalent)
    Start-Transcript -Path $LogFile -Force

    Write-Host ">>> PREPARING JOB: Pair $Ip -> $Op" -ForegroundColor Cyan
    Write-Host "================= STARTING PIPELINE $(Get-Date) ================="

    try {
        # === Extract DIO (Trodes) ===
        $recFiles = Get-ChildItem -Path $Ip -Filter "*.rec" -Recurse
        foreach ($file in $recFiles) {
            Write-Host "--- RUNNING TRODES: $($file.Name) ---" -ForegroundColor Yellow
            # Call executable using call operator &
            & $TrodesBin -dio -rec "$($file.FullName)"
        }

        # === Run Sync Script ===
        Write-Host "--- RUNNING LED SYNC ---" -ForegroundColor Yellow
        # Using python -u for unbuffered output
        python -u ./src/Video_LED_Sync_using_ICA.py -i "$Ip" -o "$Op" -f "$Freq"
        if ($LASTEXITCODE -ne 0) { throw "LED SYNC FAILED" }

        # === Stitch step ===
        Write-Host "--- RUNNING STITCHING ---" -ForegroundColor Yellow
        python -u ./src/join_views.py "$Ip"

        # === Tracking ===
        $StitchedVideo = Join-Path $Ip "stitched.mp4"
        if (Test-Path $StitchedVideo) {
            Write-Host "--- RUNNING YOLO TRACKER ---" -ForegroundColor Yellow
            python -u ./src/TrackerYolov.py `
                --input_folder "$StitchedVideo" `
                --output_folder "$Op" `
                --onnx_weight "$OnnxPath"
            
            if ($LASTEXITCODE -ne 0) { throw "TRACKER FAILED" }
        } else {
            Write-Host "   [Warning] stitched.mp4 not found. Skipping tracking." -ForegroundColor Magenta
        }

        # === Plotting ===
        Write-Host "--- RUNNING PLOTTING ---" -ForegroundColor Yellow
        python -u plot_trials.py -o "$Op"
        if ($LASTEXITCODE -ne 0) { throw "PLOTTING FAILED" }

        # === Compression (FFmpeg) ===
        $VideoFile = Get-ChildItem -Path $Op -Filter "*.mp4" | Select-Object -First 1
        if ($VideoFile) {
            Write-Host "--- RUNNING COMPRESSION ---" -ForegroundColor Yellow
            $TempFile = Join-Path $Op "__temp_compressed.mp4"
            
            # FFmpeg call
            ffmpeg -y -hide_banner -loglevel warning -i "$($VideoFile.FullName)" -vcodec h264_nvenc -qp 30 "$TempFile"
            
            if (Test-Path $TempFile) {
                Move-Item -Path $TempFile -Destination "$($VideoFile.FullName)" -Force
            }
        }

        Write-Host "================= FINISHED $(Get-Date) =================" -ForegroundColor Green
        Write-Host "You may close this window now."
    }
    catch {
        Write-Host "!!! ERROR: $($_.Exception.Message) !!!" -ForegroundColor Red
        Write-Host "Check log file: $LogFile"
    }
    finally {
        Stop-Transcript
        # Keep window open for review, similar to the bash script behavior
        Read-Host "Press Enter to exit..."
    }
}

# ================= MAIN MANAGER LOOP =================

Write-Host "Scanning $RootDir for ip/op pairs..." -ForegroundColor Cyan
Write-Host "Main Monitor Running (Job details will appear in pop-up windows)..."

# Find directory pairs
$ipDirs = Get-ChildItem -Path $RootDir -Directory -Filter "ip*" | Sort-Object Name

# List to keep track of active jobs
$ActiveProcesses = @()

foreach ($ipDir in $ipDirs) {
    $DirName = $ipDir.Name
    # Extract number (remove 'ip' prefix)
    $Num = $DirName -replace "ip",""
    $OpPath = Join-Path $RootDir ("op" + $Num)

    if (Test-Path $OpPath) {
        
        # === PRE-LAUNCH RESOURCE CHECK LOOP ===
        while ($true) {
            $CurrentCpu = Get-CpuUsage
            $CurrentGpu = Get-GpuUsage
            
            # Clean up finished processes from our tracking list
            $ActiveProcesses = $ActiveProcesses | Where-Object { -not $_.HasExited }
            $JobCount = $ActiveProcesses.Count

            # Update Status Line (Write-Host with NoNewline to update in place)
            Write-Host -NoNewline "`rSystem Status: CPU ${CurrentCpu}% | GPU ${CurrentGpu}% | Jobs Active: ${JobCount}   "

            if (($CurrentCpu -lt $MAX_CPU_LOAD) -and ($CurrentGpu -lt $MAX_GPU_LOAD)) {
                Write-Host "" # New line to clear buffer
                break
            } else {
                # If resources busy or max jobs, wait
                if ($JobCount -gt 0) {
                    Start-Sleep -Seconds 5
                } else {
                    Start-Sleep -Seconds 10
                }
            }
        }

        # === LAUNCH JOB ===
        Write-Host "Launching Job: $DirName..." -ForegroundColor Green
        
        # We convert the ScriptBlock to a string encoded for the CLI to pass it cleanly to the new window
        $EncodedCommand = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($WorkerScriptBlock.ToString()))
        
        # Prepare arguments list
        # Note: We must pass the variables explicitly into the script block
        $CommandStr = "& { $WorkerScriptBlock } -Ip '$($ipDir.FullName)' -Op '$OpPath' -Freq $FREQ -TrodesBin '$TRODES_BIN' -OnnxPath '$ONNX_WEIGHTS_PATH'"

        # Launch new PowerShell window
        $p = Start-Process powershell -ArgumentList "-NoExit", "-Command", $CommandStr -PassThru
        
        # Add to tracking list
        $ActiveProcesses += $p

        Start-Sleep -Seconds $RAMP_UP_DELAY

    } else {
        Write-Host "Skipping $DirName: Corresponding $OpPath not found." -ForegroundColor DarkGray
    }
}

Write-Host "All pairs queued. Waiting for remaining jobs to finish..." -ForegroundColor Cyan
# Wait for all tracked processes to exit
while (($ActiveProcesses | Where-Object { -not $_.HasExited }).Count -gt 0) {
    Start-Sleep -Seconds 2
}
Write-Host "All Done." -ForegroundColor Green