#!/bin/bash

# =================CONFIGURATION=================
ROOT_DIR="$1"
MAX_CPU_LOAD=80
MAX_GPU_LOAD=80
FREQ=20000

ONNX_WEIGHTS_PATH="/home/genzel/Desktop/Documents/Param/Track_sachi/yolov3_training_best.onnx"

# Initial cooldown to allow a process to ramp up resources before checking again
RAMP_UP_DELAY=10 
# ===============================================

# 1. Check if Root Directory is provided
if [[ -z "$ROOT_DIR" ]]; then
    echo "Usage: ./batch_runner.sh <path_to_data_folder>"
    exit 1
fi

# ================= FUNCTIONS =================
get_cpu_usage() {
    # Method 1: Try using vmstat (Most reliable, low overhead)
    if command -v vmstat &> /dev/null; then
        # Run vmstat for 1 second, 2 times. Tail gets the last line.
        # Column 15 is usually 'id' (idle). We calculate 100 - idle.
        local idle=$(LC_ALL=C vmstat 1 2 | tail -n 1 | awk '{print $15}')
        echo "$idle" | awk '{print 100 - $1}'
        
    # Method 2: Fallback to TOP if vmstat is missing
    else
        # grep the line with "id", extract the number before "id"
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

# --- NEW: CONSTANT MONITOR FUNCTION ---
monitor_resources() {
    while true; do
        sleep 30
        local cpu=$(get_cpu_usage)
        local gpu=$(get_gpu_usage)
        local timestamp=$(date '+%H:%M:%S')
        
        # Print to stderr to ensure it shows up even if stdout is redirected
        echo "   [WATCHDOG $timestamp] CPU: ${cpu}% | GPU: ${gpu}%" >&2
    done
}
# =============================================

# =================THE PIPELINE WORKER=================
run_pipeline() {
    local IP="$1"
    local OP="$2"
    local LOG_FILE="$OP/pipeline_log.txt"
    
    mkdir -p "$OP"
    
    echo ">>> STARTING JOB: Pair $IP -> $OP"
    echo "================= STARTING PIPELINE $(date) =================" > "$LOG_FILE"

    # === Extract DIO ===
    for i in $(find "$IP" -name "*.rec" -type f); do
        echo "   [Trodes] Processing: $i"
        echo "--- RUNNING TRODES: $i ---" >> "$LOG_FILE"
        ./Trodes_2-3-2_Ubuntu2004/trodesexport -dio -rec "$i" 2>&1 | tee -a "$LOG_FILE"
    done

    # === Run Sync Script ===
    echo "   [Sync] Running Video_LED_Sync..."
    echo "--- RUNNING LED SYNC ---" >> "$LOG_FILE"
    python3 -u ./src/Video_LED_Sync_using_ICA.py -i "$IP" -o "$OP" -f "$FREQ" 2>&1 | tee -a "$LOG_FILE"
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "!!! ERROR in LED SYNC for $IP !!!"
        echo "    Check log: $LOG_FILE"
        return 
    fi

    # === Stitch step ===
    python3 -u ./src/join_views.py "$IP" 2>&1 | tee -a "$LOG_FILE"

    # ==== Tracking ====
    if [[ -f "$IP/stitched.mp4" ]]; then
        echo "   [Tracker] Running YOLO..."
        echo "--- RUNNING YOLO TRACKER ---" >> "$LOG_FILE"

        python3 -u ./src/TrackerYolov.py \
            --input_folder "$IP/stitched.mp4" \
            --output_folder "$OP" \
            --onnx_weight "$ONNX_WEIGHTS_PATH" 2>&1 | tee -a "$LOG_FILE"

        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            echo "!!! ERROR in YOLO TRACKER for $IP !!!"
            echo "    Check log: $LOG_FILE"
            return
        fi
    else
        echo "   [Warning] $IP/stitched.mp4 not found. Skipping tracking." | tee -a "$LOG_FILE"
    fi

    # ==== Plotting ====
    echo "   [Plotting] Running plot_trials..."
    echo "--- RUNNING PLOTTING ---" >> "$LOG_FILE"
    python3 -u plot_trials.py -o "$OP" 2>&1 | tee -a "$LOG_FILE"
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "!!! ERROR in PLOTTING for $IP !!!"
        return
    fi

    # ==== Compression ====
    VIDEO_FILE=$(ls "$OP"/*.mp4 2>/dev/null | head -n 1)
    if [[ -n "$VIDEO_FILE" ]]; then
        echo "   [Compress] Compressing $VIDEO_FILE..."
        echo "--- RUNNING COMPRESSION ---" >> "$LOG_FILE"
        TEMP_FILE="$OP/__temp_compressed.mp4"
        ffmpeg -y -hide_banner -loglevel warning -i "$VIDEO_FILE" -vcodec h264_nvenc -qp 30 "$TEMP_FILE" 2>&1 | tee -a "$LOG_FILE"
        if [[ -f "$TEMP_FILE" ]]; then
            mv -f "$TEMP_FILE" "$VIDEO_FILE"
        fi
    fi

    echo ">>> FINISHED JOB: $IP"
    echo "================= FINISHED $(date) =================" >> "$LOG_FILE"
}

# =================MAIN MANAGER LOOP=================

echo "Scanning $ROOT_DIR for ip/op pairs..."

# --- START BACKGROUND MONITOR ---
echo "Starting background resource monitor (Updates every 30s)..."
monitor_resources &       # Run in background
MONITOR_PID=$!            # Save the Process ID
trap "kill $MONITOR_PID" EXIT # Kill it when script ends or crashes
# --------------------------------

# Find directory pairs
find "$ROOT_DIR" -maxdepth 1 -type d -name "ip*" | sort | while read -r IP_PATH; do
    
    DIR_NAME=$(basename "$IP_PATH")
    NUM="${DIR_NAME#ip}"
    OP_PATH="$ROOT_DIR/op$NUM"

    if [[ -d "$OP_PATH" ]]; then
        echo "--------------------------------------------------------"
        echo "Found Next Candidate: $DIR_NAME -> op$NUM"
        
        # === PRE-LAUNCH RESOURCE CHECK LOOP ===
        # This loop ensures we don't start if system is ALREADY overloaded
        while true; do
            CURRENT_CPU=$(get_cpu_usage)
            CURRENT_GPU=$(get_gpu_usage)
            JOBS_RUNNING=$(jobs -r | wc -l)

            echo "   [MAIN CHECK] Active Jobs: $JOBS_RUNNING | CPU: $CURRENT_CPU% | GPU: $CURRENT_GPU%"
            
            if (( CURRENT_CPU < MAX_CPU_LOAD )) && (( CURRENT_GPU < MAX_GPU_LOAD )); then
                break
            else
                if [[ "$JOBS_RUNNING" -gt 0 ]]; then
                    echo "   Resources busy. Waiting for a task to finish..."
                    wait -n
                else
                    echo "   Resources busy (external load). Sleeping 10s..."
                    sleep 10
                fi
            fi
        done

        # === LAUNCH JOB ===
        run_pipeline "$IP_PATH" "$OP_PATH" &
        
        sleep "$RAMP_UP_DELAY"
        
    else
        echo "Skipping $DIR_NAME: Corresponding $OP_PATH not found."
    fi
done

echo "All pairs queued. Waiting for remaining jobs to finish..."
wait
echo "All Done."