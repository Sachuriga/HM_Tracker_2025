#!/bin/bash

# =================CONFIGURATION=================
ROOT_DIR="$1"
MAX_CPU_LOAD=80
MAX_GPU_LOAD=80
FREQ=20000

ONNX_WEIGHTS_PATH="/home/genzel/Desktop/Documents/Param/Track_sachi/yolov3_training_best.onnx"

# Initial cooldown to allow a process to ramp up resources
RAMP_UP_DELAY=10 
# ===============================================

# 1. Check if Root Directory is provided
if [[ -z "$ROOT_DIR" ]]; then
    echo "Usage: ./batch_runner.sh <path_to_data_folder>"
    exit 1
fi

# ================= FUNCTIONS =================
get_cpu_usage() {
    if command -v vmstat &> /dev/null; then
        local idle=$(LC_ALL=C vmstat 1 2 | tail -n 1 | awk '{print $15}')
        echo "$idle" | awk '{print 100 - $1}'
    else
        LC_ALL=C top -bn2 -d 0.5 | grep "Cpu(s)" | tail -n 1 | \
        sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | \
        awk '{print 100 - $1}' | awk '{printf "%.0f", $1}'
    fi
}

get_gpu_usage() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n 1
    else
        echo "0"
    fi
}

# --- NEW: CYCLING DASHBOARD FUNCTION ---
live_dashboard() {
    # Give the main script a moment to start jobs before we start looking
    sleep 5 
    
    while true; do
        # 1. Find logs modified in the last 2 minutes (-mmin -2)
        # This ensures we only watch jobs that are currently running.
        ACTIVE_LOGS=$(find "$ROOT_DIR" -name "pipeline_log.txt" -mmin -2)

        # 2. If no logs are active, just show stats and wait
        if [[ -z "$ACTIVE_LOGS" ]]; then
            local cpu=$(get_cpu_usage)
            local gpu=$(get_gpu_usage)
            echo "   [System Idle] CPU: ${cpu}% | GPU: ${gpu}% (Waiting for active jobs...)"
            sleep 10
            continue
        fi

        # 3. Cycle through each active log
        for logfile in $ACTIVE_LOGS; do
            # Get folder name for display (e.g., op1)
            PARENT_DIR=$(dirname "$logfile")
            JOB_NAME=$(basename "$PARENT_DIR")
            
            local cpu=$(get_cpu_usage)
            local gpu=$(get_gpu_usage)
            local timestamp=$(date '+%H:%M:%S')

            # Visual Separator
            echo "------------------------------------------------------------"
            echo " [WATCHDOG $timestamp] CPU: ${cpu}% | GPU: ${gpu}%"
            echo " >>> STREAMING OUTPUT: $JOB_NAME (Switching in 10s)"
            echo "------------------------------------------------------------"

            # 4. Tail the log for 10 seconds, then kill the tail command
            # 'timeout' runs the command for X duration.
            # 'tail -f' follows the file as it grows.
            timeout 10s tail -n 5 -f "$logfile"

            echo "" # New line for cleanliness
        done
    done
}
# =============================================

# =================THE PIPELINE WORKER=================
run_pipeline() {
    local IP="$1"
    local OP="$2"
    local LOG_FILE="$OP/pipeline_log.txt"
    
    mkdir -p "$OP"
    
    # We output to the LOG_FILE, but we do NOT pipe to stdout here.
    # The 'live_dashboard' will handle showing us the content.
    
    echo ">>> STARTING JOB: Pair $IP -> $OP"
    echo "================= STARTING PIPELINE $(date) =================" > "$LOG_FILE"

    # === Extract DIO ===
    for i in $(find "$IP" -name "*.rec" -type f); do
        echo "--- RUNNING TRODES: $i ---" >> "$LOG_FILE"
        ./Trodes_2-3-2_Ubuntu2004/trodesexport -dio -rec "$i" >> "$LOG_FILE" 2>&1
    done

    # === Run Sync Script ===
    echo "--- RUNNING LED SYNC ---" >> "$LOG_FILE"
    python3 -u ./src/Video_LED_Sync_using_ICA.py -i "$IP" -o "$OP" -f "$FREQ" >> "$LOG_FILE" 2>&1
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "!!! ERROR: LED SYNC FAILED ($OP) !!!" >> "$LOG_FILE"
        return 
    fi

    # === Stitch step ===
    python3 -u ./src/join_views.py "$IP" >> "$LOG_FILE" 2>&1

    # ==== Tracking ====
    if [[ -f "$IP/stitched.mp4" ]]; then
        echo "--- RUNNING YOLO TRACKER ---" >> "$LOG_FILE"

        python3 -u ./src/TrackerYolov.py \
            --input_folder "$IP/stitched.mp4" \
            --output_folder "$OP" \
            --onnx_weight "$ONNX_WEIGHTS_PATH" >> "$LOG_FILE" 2>&1

        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            echo "!!! ERROR: TRACKER FAILED ($OP) !!!" >> "$LOG_FILE"
            return
        fi
    else
        echo "   [Warning] $IP/stitched.mp4 not found. Skipping tracking." >> "$LOG_FILE"
    fi

    # ==== Plotting ====
    echo "--- RUNNING PLOTTING ---" >> "$LOG_FILE"
    python3 -u plot_trials.py -o "$OP" >> "$LOG_FILE" 2>&1
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "!!! ERROR: PLOTTING FAILED ($OP) !!!" >> "$LOG_FILE"
        return
    fi

    # ==== Compression ====
    VIDEO_FILE=$(ls "$OP"/*.mp4 2>/dev/null | head -n 1)
    if [[ -n "$VIDEO_FILE" ]]; then
        echo "--- RUNNING COMPRESSION ---" >> "$LOG_FILE"
        TEMP_FILE="$OP/__temp_compressed.mp4"
        ffmpeg -y -hide_banner -loglevel warning -i "$VIDEO_FILE" -vcodec h264_nvenc -qp 30 "$TEMP_FILE" >> "$LOG_FILE" 2>&1
        if [[ -f "$TEMP_FILE" ]]; then
            mv -f "$TEMP_FILE" "$VIDEO_FILE"
        fi
    fi

    echo "================= FINISHED $(date) =================" >> "$LOG_FILE"
}

# =================MAIN MANAGER LOOP=================

echo "Scanning $ROOT_DIR for ip/op pairs..."

# --- START BACKGROUND DASHBOARD ---
echo "Starting Live Log Switcher..."
live_dashboard &          # Run in background
MONITOR_PID=$!            # Save the Process ID
trap "kill $MONITOR_PID" EXIT # Kill it when script ends
# --------------------------------

# Find directory pairs
find "$ROOT_DIR" -maxdepth 1 -type d -name "ip*" | sort | while read -r IP_PATH; do
    
    DIR_NAME=$(basename "$IP_PATH")
    NUM="${DIR_NAME#ip}"
    OP_PATH="$ROOT_DIR/op$NUM"

    if [[ -d "$OP_PATH" ]]; then
        
        # === PRE-LAUNCH RESOURCE CHECK LOOP ===
        while true; do
            CURRENT_CPU=$(get_cpu_usage)
            CURRENT_GPU=$(get_gpu_usage)
            JOBS_RUNNING=$(jobs -r | wc -l)

            # Note: We removed the print statement here because the dashboard is handling output.
            # Only print if we are actually blocked.
            
            if (( CURRENT_CPU < MAX_CPU_LOAD )) && (( CURRENT_GPU < MAX_GPU_LOAD )); then
                break
            else
                # If busy, just wait silently (dashboard will show high CPU)
                if [[ "$JOBS_RUNNING" -gt 0 ]]; then
                    wait -n
                else
                    sleep 10
                fi
            fi
        done

        # === LAUNCH JOB ===
        # We don't print "Found Next Candidate" to stdout to keep the dashboard clean
        # The dashboard will pick it up once the log file is created.
        run_pipeline "$IP_PATH" "$OP_PATH" &
        
        sleep "$RAMP_UP_DELAY"
        
    else
        # Log silent errors to a global error log if needed, or stderr
        echo "Skipping $DIR_NAME: Corresponding $OP_PATH not found." >&2
    fi
done

echo "All pairs queued. Waiting for remaining jobs to finish..."
wait
echo "All Done."