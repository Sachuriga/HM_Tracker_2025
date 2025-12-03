#!/bin/bash

# =================CONFIGURATION=================
ROOT_DIR="$1"
# 注意：现在脚本主要由 GPU 数量限制并发，这 CPU 阈值作为二级保险
MAX_CPU_LOAD=95 
FREQ=20000
PYTHON_EXEC="/home/sachuriga/yolov12x/bin/python3"
ONNX_WEIGHTS_PATH="/home/genzel/Desktop/Documents/Param/yolov3_training_best.onnx"

# Initial cooldown to allow a process to ramp up resources
RAMP_UP_DELAY=5
# ===============================================

# 1. Check if Root Directory is provided
if [[ -z "$ROOT_DIR" ]]; then
    echo "Usage: ./batch_runner.sh <path_to_data_folder>"
    exit 1
fi

# === NEW: DETECT GPUS ===
# 获取系统中的 GPU 数量
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "Detected $NUM_GPUS NVIDIA GPU(s)."
else
    NUM_GPUS=0
    echo "No NVIDIA GPUs detected. Logic will default to CPU or fail if GPU is required."
fi

# 初始化 GPU 状态数组，用来存储当前占用该 GPU 的进程 PID
# GPU_PIDS[0] 存储占用 GPU 0 的任务 PID，以此类推
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

# =================THE PIPELINE WORKER=================
run_pipeline() {

    source ~/yolov12x/bin/activate

    local IP="$1"
    local OP="$2"
    local ASSIGNED_GPU_ID="$3"  # 接收分配的 GPU ID
    local JOB_NAME=$(basename "$IP")
    local LOG_FILE="$OP/pipeline_log.txt"
    
    mkdir -p "$OP"
    
    # Initialize the log file
    echo ">>> PREPARING JOB: Pair $IP -> $OP (On GPU: $ASSIGNED_GPU_ID)" > "$LOG_FILE"

    # --- NEW: POP UP WINDOW ---
    if command -v gnome-terminal &> /dev/null; then
        # 标题显示使用的 GPU ID
        env -u LD_LIBRARY_PATH -u PYTHONPATH gnome-terminal --title="GPU $ASSIGNED_GPU_ID : $JOB_NAME" --geometry=100x24 -- bash -c "tail -f \"$LOG_FILE\"" &
    elif command -v xterm &> /dev/null; then
        xterm -T "GPU $ASSIGNED_GPU_ID : $JOB_NAME" -e "tail -f \"$LOG_FILE\"" &
    fi
    # --------------------------

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

            # IMPORTANT: 这里传入了 --gpu_id 参数
            # 请确保你的 TrackerYolov11.py 已经按照上一个回答修改以接受此参数
            "$PYTHON_EXEC" -u src/TrackerYolov11.py \
                --input_folder "$IP/stitched.mp4" \
                --output_folder "$OP" \
                --onnx_weight "$ONNX_WEIGHTS_PATH" \
                --gpu_id "$ASSIGNED_GPU_ID"

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
        # 注意：ffmpeg 也可以指定 GPU，如果需要硬件加速编码，
        # 可以添加 -gpu $ASSIGNED_GPU_ID (取决于 ffmpeg 版本)
        VIDEO_FILE=$(ls "$OP"/*.mp4 2>/dev/null | head -n 1)
        if [[ -n "$VIDEO_FILE" ]]; then
            echo "--- RUNNING COMPRESSION ---" 
            TEMP_FILE="$OP/__temp_compressed.mp4"
            # h264_nvenc 默认通常使用 GPU 0，如果有多个显卡，
            # 可以尝试添加 -gpu $ASSIGNED_GPU_ID 选项 (如果不报错的话)
            ffmpeg -y -hide_banner -loglevel warning -i "$VIDEO_FILE" -vcodec h264_nvenc -qp 30 "$TEMP_FILE" 
            if [[ -f "$TEMP_FILE" ]]; then
                mv -f "$TEMP_FILE" "$VIDEO_FILE"
            fi
        fi

        echo "================= FINISHED $(date) =================" 

    } >> "$LOG_FILE" 2>&1
}

# =================MAIN MANAGER LOOP=================

echo "Scanning $ROOT_DIR..."
echo "Detected $NUM_GPUS GPUs. Jobs will be distributed across them."

# Find directory pairs
find "$ROOT_DIR" -maxdepth 1 -type d -name "ip*" | sort | while read -r IP_PATH; do
    
    DIR_NAME=$(basename "$IP_PATH")
    NUM="${DIR_NAME#ip}"
    OP_PATH="$ROOT_DIR/op$NUM"

    if [[ -d "$OP_PATH" ]]; then
        
        # === RESOURCE SCHEDULER LOOP ===
        # 这个循环会一直运行，直到找到一个空闲的 GPU
        SELECTED_GPU=-1
        
        while true; do
            # 1. Check CPU Safety (全局检查)
            CURRENT_CPU=$(get_cpu_usage)
            if (( CURRENT_CPU >= MAX_CPU_LOAD )); then
                echo -ne "System CPU Busy ($CURRENT_CPU%). Waiting...\r"
                sleep 5
                continue
            fi

            # 2. Check for free GPU Slot
            # 我们遍历 GPU_PIDS 数组，检查里面的 PID 是否还活着
            for ((i=0; i<NUM_GPUS; i++)); do
                PID_CHECK="${GPU_PIDS[$i]}"
                
                # 如果 PID 为空，或者 PID 对应的进程已经不存在了(任务跑完了)
                if [[ -z "$PID_CHECK" ]] || ! kill -0 "$PID_CHECK" 2>/dev/null; then
                    SELECTED_GPU=$i
                    break # 找到了空闲显卡，跳出 for 循环
                fi
            done

            # 3. Decision
            if [[ "$SELECTED_GPU" -ne -1 ]]; then
                # 找到了可用显卡，跳出 while 等待循环
                break
            else
                # 所有显卡都在忙
                echo -ne "All $NUM_GPUS GPUs are busy. Waiting for a slot...\r"
                sleep 5
            fi
        done

        echo "" # Clear line

        # === LAUNCH JOB ===
        echo "Launching Job: $DIR_NAME on GPU $SELECTED_GPU..."
        
        # 后台运行，并传入选中的 GPU ID
        run_pipeline "$IP_PATH" "$OP_PATH" "$SELECTED_GPU" &
        
        # 获取刚刚启动的后台进程 PID
        NEW_PID=$!
        
        # 将 PID 填入对应的 GPU 槽位，标记该显卡为"忙碌"
        GPU_PIDS[$SELECTED_GPU]=$NEW_PID
        
        sleep "$RAMP_UP_DELAY"
        
    else
        echo "Skipping $DIR_NAME: Corresponding $OP_PATH not found." >&2
    fi
done

echo "All pairs queued. Waiting for remaining jobs to finish..."
wait
echo "All Done."