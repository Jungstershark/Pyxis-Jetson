#!/usr/bin/env python3

import socket
import json
import time
import subprocess
import numpy as np
from ultralytics import YOLO
import supervision as sv

# =========================
# CONFIG
# =========================

MODEL_PATH = "../weights/y11m_t0_960/best.pt"

CAMERA_DEVICE = "/dev/video0"   # AI camera
TARGET_CLASS = "pilot_ladder"

W = 640
H = 480
FPS = 60

SRC_IP = "10.42.0.1"   # Jetson ethernet
DST_IP = "10.42.0.2"   # Raspberry Pi ethernet
DST_PORT = 5005

# =========================
# UDP SOCKET
# =========================

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((SRC_IP, 0))

# =========================
# LOAD MODEL
# =========================

print("Loading model...")
model = YOLO(MODEL_PATH)
names = model.names

print("Model loaded.")
print("Classes:", names)

# =========================
# GSTREAMER CAMERA
# =========================

def gst_cmd(device):
    return [
        'gst-launch-1.0', '-q',
        'v4l2src', f'device={device}',
        '!', f'image/jpeg,width={W},height={H},framerate={FPS}/1',
        '!', 'jpegdec',
        '!', 'videoconvert',
        '!', f'video/x-raw,format=BGR,width={W},height={H}',
        '!', 'fdsink', 'fd=1'
    ]

print("Starting camera pipeline...")

proc = subprocess.Popen(
    gst_cmd(CAMERA_DEVICE),
    stdout=subprocess.PIPE,
    bufsize=W * H * 3
)

frame_size = W * H * 3

# =========================
# FPS TRACKING
# =========================

frame_count = 0
start_time = time.time()

print("Running inference...")

# =========================
# MAIN LOOP
# =========================

while True:

    raw = proc.stdout.read(frame_size)

    if len(raw) != frame_size:
        print("Frame read failed")
        continue

    frame = np.frombuffer(raw, dtype=np.uint8).reshape((H, W, 3)).copy()

    results = model.predict(
        frame,
        imgsz=640,
        conf=0.25,
        iou=0.5,
        device=0,
        half=True,
        verbose=False
    )

    det = sv.Detections.from_ultralytics(results[0])

    ladder_detections = []

    for i in range(len(det.xyxy)):

        class_id = int(det.class_id[i])
        class_name = names[class_id]

        if class_name != TARGET_CLASS:
            continue

        x1, y1, x2, y2 = det.xyxy[i]
        conf = float(det.confidence[i])

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        ladder_detections.append({
            "confidence": conf,
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "cx": float(cx),
            "cy": float(cy),
            "cx_norm": float(cx / W),
            "cy_norm": float(cy / H)
        })

    packet = {
        "timestamp": time.time(),
        "frame_width": W,
        "frame_height": H,
        "target_class": TARGET_CLASS,
        "target_valid": len(ladder_detections) > 0,
        "num_ladders": len(ladder_detections),
        "detections": ladder_detections
    }

    sock.sendto(json.dumps(packet).encode(), (DST_IP, DST_PORT))

    frame_count += 1

    if frame_count % 30 == 0:
        elapsed = time.time() - start_time
        fps = frame_count / elapsed
        print(f"FPS: {fps:.2f} | ladders: {len(ladder_detections)}")

# =========================
# CLEANUP
# =========================

proc.terminate()

