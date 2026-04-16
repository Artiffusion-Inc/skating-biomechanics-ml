#!/bin/bash
# Monitor YOLO26-Pose HP search experiments
# Runs every 5 minutes via cron

LOG_DIR="/root/logs"
STATUS_LOG="/root/logs/hp_status.log"
GPU_LOG="/root/logs/gpu_status.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Create logs if not exist
touch "$STATUS_LOG" "$GPU_LOG"

# Function to log with timestamp
log() {
    echo "[$TIMESTAMP] $1" >> "$STATUS_LOG"
}

# Check GPU status
log "=== GPU Status ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader >> "$GPU_LOG" 2>&1
log "GPU status logged to $GPU_LOG"

# Check running processes
RUNNING=$(ps aux | grep 'train_yolo26_pose' | grep -v grep | wc -l)
log "Running experiments: $RUNNING/10"

if [ "$RUNNING" -lt 10 ]; then
    log "WARNING: Only $RUNNING experiments running!"
fi

# Check each log for progress
log "=== Training Progress ==="
for log_file in $LOG_DIR/hp_*.log; do
    cfg=$(basename "$log_file" .log)

    # Get last epoch number
    last_epoch=$(grep -oE 'Epoch[[:space:]]+[0-9]+' "$log_file" | tail -1 | grep -oE '[0-9]+' || echo "0")

    # Check for errors
    errors=$(grep -iE 'error|nan|inf' "$log_file" | tail -3)

    if [ -n "$errors" ]; then
        log "ERROR in $cfg: $errors"
    else
        log "$cfg: Epoch $last_epoch"
    fi
done

# Check disk space
DISK_USAGE=$(df -h / | awk 'NR==2 {print $5}')
log "Disk usage: $DISK_USAGE"

# Check if any experiment finished
log "=== Completed Experiments ==="
for log_file in $LOG_DIR/hp_*.log; do
    if grep -q "training complete" "$log_file" 2>/dev/null; then
        cfg=$(basename "$log_file" .log)
        log "COMPLETED: $cfg"
    fi
done

log "=================================="
echo "" >> "$STATUS_LOG"
