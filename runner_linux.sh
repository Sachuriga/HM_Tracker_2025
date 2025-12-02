#!/bin/bash

# =================CONFIGURATION=================
ROOT_DIR="$1"
MAX_CPU_LOAD=95
MAX_GPU_LOAD=95
FREQ=20000

ONNX_WEIGHTS_PATH="/home/genzel/Desktop/Documents/Param/yolov3_training_best.onnx"

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

# =================THE PIPELINE WORKER=================
run_pipeline() {
    local IP="$1"
    local OP="$2"
    local JOB_NAME=$(basename "$IP")
    local LOG_FILE="$OP/pipeline_log.txt"
    
    mkdir -p "$OP"
    
    # Initialize the log file
    echo ">>> PREPARING JOB: Pair $IP -> $OP" > "$LOG_FILE"

    # --- NEW: POP UP WINDOW ---
    # This opens a new terminal window that monitors the log file created above.
    # It uses 'tail -f' to stream output.
    if command -v gnome-terminal &> /dev/null; then
        # Opens GNOME terminal. The window stays open until you close it.
        gnome-terminal --title="Running: $JOB_NAME" --geometry=100x24 -- bash -c "tail -f \"$LOG_FILE\"" &
    elif command -v xterm &> /dev/null; then
        # Fallback for xterm
        xterm -T "Running: $JOB_NAME" -e "tail -f \"$LOG_FILE\"" &
    else
        echo "Warning: No terminal emulator found (gnome-terminal or xterm). Logs will strictly be in files."
    fi
    # --------------------------

    # Redirect ALL output below this line to the log file
    {
        echo "================= STARTING PIPELINE $(date) =================" 

        # === Extract DIO ===
        for i in $(find "$IP" -name "*.rec" -type f); do
            echo "--- RUNNING TRODES: $i ---" 
            /home/genzel/Desktop/Documents/Param/Trodes_2-3-2_Ubuntu2004/trodesexport -dio -rec "$i" 
        done

        # === Run Sync Script ===
        echo "--- RUNNING LED SYNC ---" 
        python3 -u ./src/Video_LED_Sync_using_ICA.py -i "$IP" -o "$OP" -f "$FREQ" 
        
        # Check Sync Status
        if [ $? -ne 0 ]; then
            echo "!!! ERROR: LED SYNC FAILED ($OP) !!!" 
            echo "Check log above for details."
            return 
        fi

        # === Stitch step ===
        echo "--- RUNNING STITCHING ---"
        python3 -u ./src/join_views.py "$IP" 

        # ==== Tracking ====
        if [[ -f "$IP/stitched.mp4" ]]; then
            echo "--- RUNNING YOLO TRACKER ---" 

            python3 -u ./src/TrackerYolov.py \
                --input_folder "$IP/stitched.mp4" \
                --output_folder "$OP" \
                --onnx_weight "$ONNX_WEIGHTS_PATH" 

            if [ $? -ne 0 ]; then
                echo "!!! ERROR: TRACKER FAILED ($OP) !!!" 
                return
            fi
        else
            echo "   [Warning] $IP/stitched.mp4 not found. Skipping tracking." 
        fi

        # ==== Plotting ====
        echo "--- RUNNING PLOTTING ---" 
        python3 -u plot_trials.py -o "$OP" 
        
        if [ $? -ne 0 ]; then
            echo "!!! ERROR: PLOTTING FAILED ($OP) !!!" 
            return
        fi

        # ==== Compression ====
        VIDEO_FILE=$(ls "$OP"/*.mp4 2>/dev/null | head -n 1)
        if [[ -n "$VIDEO_FILE" ]]; then
            echo "--- RUNNING COMPRESSION ---" 
            TEMP_FILE="$OP/__temp_compressed.mp4"
            ffmpeg -y -hide_banner -loglevel warning -i "$VIDEO_FILE" -vcodec h264_nvenc -qp 30 "$TEMP_FILE" 
            if [[ -f "$TEMP_FILE" ]]; then
                mv -f "$TEMP_FILE" "$VIDEO_FILE"
            fi
        fi

        echo "================= FINISHED $(date) =================" 
        echo " You may close this window now."

    } >> "$LOG_FILE" 2>&1
}

# =================MAIN MANAGER LOOP=================

echo "Scanning $ROOT_DIR for ip/op pairs..."
echo "Main Monitor Running (Job details will appear in pop-up windows)..."

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

            # Print status to main terminal
            echo -ne "System Status: CPU ${CURRENT_CPU}% | GPU ${CURRENT_GPU}% | Jobs Active: ${JOBS_RUNNING}\r"
            
            if (( CURRENT_CPU < MAX_CPU_LOAD )) && (( CURRENT_GPU < MAX_GPU_LOAD )); then
                echo "" # New line to clear the carriage return
                break
            else
                # If busy, wait
                if [[ "$JOBS_RUNNING" -gt 0 ]]; then
                    wait -n
                else
                    sleep 10
                fi
            fi
        done

        # === LAUNCH JOB ===
        echo "Launching Job: $DIR_NAME..."
        run_pipeline "$IP_PATH" "$OP_PATH" &
        
        sleep "$RAMP_UP_DELAY"
        
    else
        echo "Skipping $DIR_NAME: Corresponding $OP_PATH not found." >&2
    fi
done

echo "All pairs queued. Waiting for remaining jobs to finish..."
wait
echo "All Done."