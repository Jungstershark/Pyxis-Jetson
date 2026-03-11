#!/bin/bash
set -e

PYTHON_BIN="/home/pixysjetson1/.pyenv/versions/myvenv/bin/python"
SCRIPT_DIR="/home/pixysjetson1/pyxis/main_script"

echo "Starting Jetson CV stack..."

cleanup() {
    echo "Stopping all processes..."
    kill "$PID_INF" "$PID_CAM2" 2>/dev/null || true
    pkill -f "gst-launch-1.0 v4l2src device=/dev/video2" || true
    exit
}

trap cleanup SIGINT SIGTERM

cd "$SCRIPT_DIR"

echo "Using Python: $PYTHON_BIN"
"$PYTHON_BIN" -c "import sys, numpy, ultralytics, supervision; print(sys.executable); print('python env ok')" 

echo "Starting inference..."
"$PYTHON_BIN" jetson_inference_sender.py &
PID_INF=$!

sleep 2

echo "Starting camera2 stream..."
./stream_cam2.sh &
PID_CAM2=$!

echo "Running:"
echo "Inference PID: $PID_INF"
echo "Cam2 PID: $PID_CAM2"

wait
