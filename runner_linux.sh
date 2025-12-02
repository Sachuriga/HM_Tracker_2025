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
    # LC_ALL=C ensures 'top' output is in standard English with dot decimals
    LC_ALL=C top -bn2 -d 0.5 | grep "Cpu(s)" | tail -n 1 | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}' | awk '{printf "%.0f", $1}'
}

get_gpu_usage() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n 1
    else
        echo "0"
    fi
}
# =============================================

# =================THE PIPELINE WORKER=================
run_pipeline() {
    local IP="$1"
    local OP="$2"
    local LOG_FILE="$OP/pipeline_log.txt"
    
    # Create the output directory if it doesn't exist so we can write the log
    mkdir -p "$OP"
    
    echo ">>> STARTING JOB: Pair $IP -> $OP"
    echo "================= STARTING PIPELINE $(date) =================" > "$LOG_FILE"

    # === Extract DIO using Trodes ===
    for i in $(find "$IP" -name "*.rec" -type f); do
        echo "   [Trodes] Processing: $i"
        echo "--- RUNNING TRODES: $i ---" >> "$LOG_FILE"
        
        # Using tee to show output on screen AND save to file
        ./Trodes_2-3-2_Ubuntu2004/trodesexport -dio -rec "$i" 2>&1 | tee -a "$LOG_FILE"
    done

    # === Run Sync Script ===
    echo "   [Sync] Running Video_LED_Sync..."
    echo "--- RUNNING LED SYNC ---" >> "$LOG_FILE"
    
    # -u ensures python output is not buffered (crucial for progress bars)
    python3 -u ./src/Video_LED_Sync_using_ICA.py -i "$IP" -o "$OP" -f "$FREQ" 2>&1 | tee -a "$LOG_FILE"
    
    # Capture the exit status of the FIRST command in the pipe (python), not tee
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "!!! ERROR in LED SYNC for $IP !!!"
        echo "    Check log: $LOG_FILE"
        return # Stop this specific pipeline
    fi

    # === (Optional) Stitch step ===
    # Using -u + tee
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
        echo "    Check log: $LOG_FILE"
        return
    fi

    # ==== Compression ====
    VIDEO_FILE=$(ls "$OP"/*.mp4 2>/dev/null | head -n 1)

    if [[ -n "$VIDEO_FILE" ]]; then
        echo "   [Compress] Compressing $VIDEO_FILE..."
        echo "--- RUNNING COMPRESSION ---" >> "$LOG_FILE"
        TEMP_FILE="$OP/__temp_compressed.mp4"
        
        # FFmpeg output is stderr usually, 2>&1 captures it.
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

# Find directory pairs
find "$ROOT_DIR" -maxdepth 1 -type d -name "ip*" | sort | while read -r IP_PATH; do
    
    DIR_NAME=$(basename "$IP_PATH")
    NUM="${DIR_NAME#ip}"
    OP_PATH="$ROOT_DIR/op$NUM"

    if [[ -d "$OP_PATH" ]]; then
        echo "--------------------------------------------------------"
        echo "Found Next Candidate: $DIR_NAME -> op$NUM"
        
        # === RESOURCE CHECK LOOP ===
        while true; do
            CURRENT_CPU=$(get_cpu_usage)
            CURRENT_GPU=$(get_gpu_usage)
            
            # Count running background jobs
            JOBS_RUNNING=$(jobs -r | wc -l)

            echo "   [SYSTEM STATUS]"
            echo "   Resources: CPU: $CURRENT_CPU% | GPU: $CURRENT_GPU% (Limit: $MAX_CPU_LOAD%)"
            echo "   Active Jobs: $JOBS_RUNNING"
            
            if [[ "$JOBS_RUNNING" -gt 0 ]]; then
                echo "   Currently Running Programs:"
                jobs -r | sed 's/.*run_pipeline//' | tr -d '&' | awk '{print "     >> IP: " $1 " | OP: " $2}'
            fi
            echo "   ---------------------------"

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

        # === LAUNCH JOB IN BACKGROUND ===
        # NOTE: Because we are using tee now, if multiple jobs run at once,
        # their output will mix together on the screen.
        run_pipeline "$IP_PATH" "$OP_PATH" &
        
        sleep "$RAMP_UP_DELAY"
        
    else
        echo "Skipping $DIR_NAME: Corresponding $OP_PATH not found."
    fi
done

echo "All pairs queued. Waiting for remaining jobs to finish..."
wait
echo "All Done."