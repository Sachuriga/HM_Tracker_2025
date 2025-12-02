#!/bin/bash

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

echo "Starting Resource Monitor (Press Ctrl+C to stop)"
echo "-------------------------------------"
printf "%-10s | %-8s | %-8s\n" "TIME" "CPU %" "GPU %"
echo "-------------------------------------"

while true; do
    # Capture values
    CURRENT_CPU=$(get_cpu_usage)
    CURRENT_GPU=$(get_gpu_usage)
    TIMESTAMP=$(date "+%H:%M:%S")

    # Print nicely formatted row
    printf "%-10s | %-8s | %-8s\n" "$TIMESTAMP" "$CURRENT_CPU" "$CURRENT_GPU"

    # Wait 10 seconds before next check
    sleep 10
done