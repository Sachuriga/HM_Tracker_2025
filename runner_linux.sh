#!/bin/bash

# =================CONFIGURATION=================
ROOT_DIR="$1"
MAX_CPU_LOAD=80
MAX_GPU_LOAD=80
FREQ=30000

ONNX_WEIGHTS_PATH="/home/genzel/Desktop/Documents/Param/Track_sachi/yolov3_training_best.onnx"

# Initial cooldown to allow a process to ramp up resources before checking again
RAMP_UP_DELAY=10 
# ===============================================

# 1. Check if Root Directory is provided
if [[ -z "$ROOT_DIR" ]]; then
    echo "Usage: ./batch_runner.sh <path_to_data_folder>"
    exit 1
fi

# Function to get current CPU usage
get_cpu_usage() {
    # Greps the idle cpu percentage and subtracts it from 100
    top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}' | awk '{printf "%.0f", $1}'
}

# Function to get current GPU usage (returns 0 if no nvidia-smi)
get_gpu_usage() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n 1
    else
        echo "0"
    fi
}

# =================THE PIPELINE WORKER=================
# This function contains your original logic
run_pipeline() {
    local IP="$1"
    local OP="$2"
    
    echo ">>> STARTING JOB: Pair $IP -> $OP"

    # === Extract DIO using Trodes ===
    for i in $(find "$IP" -name "*.rec" -type f); do
        echo "   [Trodes] Processing: $i"
        ./Trodes_2-3-2_Ubuntu2004/trodesexport -dio -rec "$i" > /dev/null 2>&1
    done

    # === Run Sync Script ===
    echo "   [Sync] Running Video_LED_Sync..."
    python3 ./src/Video_LED_Sync_using_ICA_sachi.py -i "$IP" -o "$OP" -f "$FREQ" > /dev/null 2>&1

    # === (Optional) Stitch step ===
    # python3 ./src/join_views.py "$IP"

    # ==== Tracking ====
    if [[ -f "$IP/stitched.mp4" ]]; then
        echo "   [Tracker] Running YOLO..."
        python3 ./src/TrackerYolov_2025.py --input_folder "$IP/stitched.mp4" --output_folder "$OP" > /dev/null 2>&1 --onnx_weight "$ONNX_WEIGHTS_PATH"
    else
        echo "   [Warning] $IP/stitched.mp4 not found. Skipping tracking."
    fi

    # ==== Plotting ====
    echo "   [Plotting] Running plot_trials..."
    python3 plot_trials_20251127.py -o "$OP" > /dev/null 2>&1

    # ==== Compression ====
    VIDEO_FILE=$(ls "$OP"/*.mp4 2>/dev/null | head -n 1)

    if [[ -n "$VIDEO_FILE" ]]; then
        echo "   [Compress] Compressing $VIDEO_FILE..."
        TEMP_FILE="$OP/__temp_compressed.mp4"
        
        # Using nvenc flags non-interactively
        ffmpeg -y -hide_banner -loglevel error -i "$VIDEO_FILE" -vcodec h264_nvenc -qp 30 "$TEMP_FILE"

        if [[ -f "$TEMP_FILE" ]]; then
            mv -f "$TEMP_FILE" "$VIDEO_FILE"
        fi
    fi

    echo ">>> FINISHED JOB: $IP"
}

# =================MAIN MANAGER LOOP=================

echo "Scanning $ROOT_DIR for ip/op pairs..."

# Find directory pairs
find "$ROOT_DIR" -maxdepth 1 -type d -name "ip*" | sort | while read -r IP_PATH; do
    
    # Extract the folder name (e.g., ip1)
    DIR_NAME=$(basename "$IP_PATH")
    
    # Extract the number (e.g., 1 from ip1)
    NUM="${DIR_NAME#ip}"
    
    # Construct expected Output Path
    OP_PATH="$ROOT_DIR/op$NUM"

    if [[ -d "$OP_PATH" ]]; then
        echo "--------------------------------------------------------"
        echo "Found Next Candidate: $DIR_NAME -> op$NUM"
        
        # === RESOURCE CHECK LOOP ===
        while true; do
            CURRENT_CPU=$(get_cpu_usage)
            CURRENT_GPU=$(get_gpu_usage)
            
            # --- NEW: TRACKING LOGIC ---
            # Count running background jobs
            JOBS_RUNNING=$(jobs -r | wc -l)

            echo "   [SYSTEM STATUS]"
            echo "   Resources: CPU: $CURRENT_CPU% | GPU: $CURRENT_GPU% (Limit: $MAX_CPU_LOAD%)"
            echo "   Active Jobs: $JOBS_RUNNING"
            
            if [[ "$JOBS_RUNNING" -gt 0 ]]; then
                echo "   Currently Running Programs:"
                # This grabs the active jobs, strips the function name, and cleans up quotes/ampersands to show just folders
                jobs -r | sed 's/.*run_pipeline//' | tr -d '&' | awk '{print "     >> IP: " $1 " | OP: " $2}'
            fi
            echo "   ---------------------------"
            # ---------------------------

            if (( CURRENT_CPU < MAX_CPU_LOAD )) && (( CURRENT_GPU < MAX_GPU_LOAD )); then
                # Resources are OK, break loop and start job
                break
            else
                # Resources too high. Check if any background jobs are running.
                
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
        # run_pipeline is launched with specific arguments, which allows 'jobs' command to track them
        run_pipeline "$IP_PATH" "$OP_PATH" &
        
        # Sleep briefly to let the new process spike the CPU/GPU load 
        # so the next check sees the true usage.
        sleep "$RAMP_UP_DELAY"
        
    else
        echo "Skipping $DIR_NAME: Corresponding $OP_PATH not found."
    fi
done

# Wait for all remaining background jobs to finish before exiting script
echo "All pairs queued. Waiting for remaining jobs to finish..."
wait
echo "All Done."