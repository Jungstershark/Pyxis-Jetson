#!/bin/bash

gst-launch-1.0 v4l2src device=/dev/video0 \
! image/jpeg,width=640,height=480,framerate=60/1 \
! jpegdec \
! videoconvert \
! x264enc bitrate=2048 speed-preset=ultrafast tune=zerolatency \
! rtph264pay \
! udpsink host=10.42.0.2 port=5600
