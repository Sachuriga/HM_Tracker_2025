#!/bin/bash

# =================CONFIGURATION=================
ROOT_DIR="$1"
MAX_CPU_LOAD=95 
FREQ=20000
PYTHON_EXEC="/home/sachuriga/yolov12x/bin/python3"
ONNX_WEIGHTS_PATH="/home//sachuriga/data/yolov_models/rat_yolov12_drive_run7/weights/best.pt"

# --- NEW SETTING: GPU IDLE THRESHOLD ---
# If a GPU is using more than this amount of VRAM (in MiB), consider it BUSY.
# Set to ~1000-2000 to allow for basic desktop display usage, but catch training jobs.
GPU_MEM_THRESHOLD=1500 

# Initial cooldown to allow a process to ramp up resources
RAMP_UP_DELAY=10
# ===============================================

# 1. Check if Root Directory is provided
if [[ -z "$ROOT_DIR" ]]; then
    echo "Usage: ./batch_runner.sh <path_to_data_folder>"
    exit 1
fi

# === DETECT GPUS ===
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "Detected $NUM_GPUS NVIDIA GPU(s)."
else
    NUM_GPUS=0
    echo "No NVIDIA GPUs detected. Script will likely fail."
    exit 1
fi

# Initialize array to track OUR script's processes
declare -a GPU_PIDS
for ((i=0; i<NUM_GPUS; i++)); do
    GPU_PIDS[$i]=""
done

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

get_gpu_memory_usage() {
    local gpu_id=$1
    # Returns memory used in MiB as an integer
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id"
}

# =================THE PIPELINE WORKER=================
run_pipeline() {

    source ~/yolov12x/bin/activate

    local IP="$1"
    local OP="$2"
    local ASSIGNED_GPU_ID="$3"
    local JOB_NAME=$(basename "$IP")
    local LOG_FILE="$OP/pipeline_log.txt"
    
    mkdir -p "$OP"
    
    # Initialize the log file
    echo ">>> PREPARING JOB: Pair $IP -> $OP (On GPU: $ASSIGNED_GPU_ID)" > "$LOG_FILE"

    # --- POP UP WINDOW ---
    if command -v gnome-terminal &> /dev/null; then
        env -u LD_LIBRARY_PATH -u PYTHONPATH gnome-terminal --title="GPU $ASSIGNED_GPU_ID : $JOB_NAME" --geometry=100x24 -- bash -c "tail -f \"$LOG_FILE\"" &
    elif command -v xterm &> /dev/null; then
        xterm -T "GPU $ASSIGNED_GPU_ID : $JOB_NAME" -e "tail -f \"$LOG_FILE\"" &
    fi

    {
        echo "================= STARTING PIPELINE $(date) =================" 
        echo "=== ASSIGNED GPU ID: $ASSIGNED_GPU_ID ==="

        # === Extract DIO ===
        for i in $(find "$IP" -name "*.rec" -type f); do
            echo "--- RUNNING TRODES: $i ---" 
            #/home/genzel/Desktop/Documents/Param/Trodes_2-3-2_Ubuntu2004/trodesexport -dio -rec "$i" 
        done

        # === Run Sync Script ===
        echo "--- RUNNING LED SYNC ---" 
        #"$PYTHON_EXEC" -u src/Video_LED_Sync_using_ICA.py -i "$IP" -o "$OP" -f "$FREQ" 
        
        if [ $? -ne 0 ]; then
            echo "!!! ERROR: LED SYNC FAILED ($OP) !!!" 
            return 
        fi

        # === Stitch step ===
        echo "--- RUNNING STITCHING ---"
        #"$PYTHON_EXEC" -u ./src/join_views.py "$IP" 

        # ==== Tracking ====
        if [[ -f "$IP/stitched.mp4" ]]; then
            echo "--- RUNNING YOLO TRACKER (GPU $ASSIGNED_GPU_ID) ---" 

            # IMPORTANT: Explicitly export CUDA_VISIBLE_DEVICES to be safe
            export CUDA_VISIBLE_DEVICES=$ASSIGNED_GPU_ID
            
            "$PYTHON_EXEC" -u src/TrackerYolov11_cluster.py \
                --input_folder "$IP" \
                --output_folder "$OP" \
                --onnx_weight "$ONNX_WEIGHTS_PATH" \
                --gpu_id 0 \
                --batch_size 32

            unset CUDA_VISIBLE_DEVICES

            if [ $? -ne 0 ]; then
                echo "!!! ERROR: TRACKER FAILED ($OP) !!!" 
                return
            fi
        else
            echo "   [Warning] $IP/stitched.mp4 not found. Skipping tracking." 
        fi

        # ==== Plotting ====
        echo "--- RUNNING PLOTTING ---" 
        "$PYTHON_EXEC" -u src/plot_trials.py -o "$OP" 
        
        # ==== Compression ====
        VIDEO_FILE=$(ls "$OP"/*.mp4 2>/dev/null | head -n 1)
        if [[ -n "$VIDEO_FILE" ]]; then
            echo "--- RUNNING COMPRESSION ---" 
            TEMP_FILE="$OP/__temp_compressed.mp4"
            # Attempt to use specific GPU for FFmpeg
            ffmpeg -y -hide_banner -loglevel warning -gpu "$ASSIGNED_GPU_ID" -i "$VIDEO_FILE" -vcodec h264_nvenc -qp 30 "$TEMP_FILE" 
            if [[ -f "$TEMP_FILE" ]]; then
                mv -f "$TEMP_FILE" "$VIDEO_FILE"
            fi
        fi

        echo "================= FINISHED $(date) =================" 

    } >> "$LOG_FILE" 2>&1
}

# =================MAIN MANAGER LOOP=================

echo "Scanning $ROOT_DIR..."
echo "Detected $NUM_GPUS GPUs."
echo "Only using GPUs with VRAM usage < ${GPU_MEM_THRESHOLD}MiB."

# Find directory pairs
find "$ROOT_DIR" -maxdepth 1 -type d -name "ip*" | sort | while read -r IP_PATH; do
    
    DIR_NAME=$(basename "$IP_PATH")
    NUM="${DIR_NAME#ip}"
    OP_PATH="$ROOT_DIR/op$NUM"

    if [[ -d "$OP_PATH" ]]; then
        
        # === RESOURCE SCHEDULER LOOP ===
        SELECTED_GPU=-1
        
        while true; do
            # 1. Check CPU Safety
            CURRENT_CPU=$(get_cpu_usage)
            if (( CURRENT_CPU >= MAX_CPU_LOAD )); then
                echo -ne "System CPU Busy ($CURRENT_CPU%). Waiting...\r"
                sleep 5
                continue
            fi

            # 2. Check for AVAILABLE GPU
            for ((i=0; i<NUM_GPUS; i++)); do
                # A. Check if WE (this script) are running a job there
                PID_CHECK="${GPU_PIDS[$i]}"
                if [[ -n "$PID_CHECK" ]] && kill -0 "$PID_CHECK" 2>/dev/null; then
                    # We are running a job here, so skip this GPU
                    continue
                fi

                # B. Check REAL Hardware usage (Are others using it?)
                CURRENT_VRAM=$(get_gpu_memory_usage "$i")
                
                if [[ "$CURRENT_VRAM" -lt "$GPU_MEM_THRESHOLD" ]]; then
                    SELECTED_GPU=$i
                    # Found a free one! Break the inner for-loop
                    break 
                fi
            done

            # 3. Decision
            if [[ "$SELECTED_GPU" -ne -1 ]]; then
                # Found a valid GPU, break the while-waiting loop
                break
            else
                # All GPUs are busy (either by us or someone else)
                echo -ne "Waiting for a spare GPU (Scanning $NUM_GPUS devices)...   \r"
                sleep 5
            fi
        done

        echo "" # Clear line

        # === LAUNCH JOB ===
        echo "Launching Job: $DIR_NAME on GPU $SELECTED_GPU..."
        
        run_pipeline "$IP_PATH" "$OP_PATH" "$SELECTED_GPU" &
        NEW_PID=$!
        GPU_PIDS[$SELECTED_GPU]=$NEW_PID
        
        # Ramp up delay to allow VRAM usage to spike so the next loop detects it
        sleep "$RAMP_UP_DELAY"
        
    else
        echo "Skipping $DIR_NAME: Corresponding $OP_PATH not found." >&2
    fi
done

echo "All pairs queued. Waiting for remaining jobs to finish..."
wait
echo "All Done."