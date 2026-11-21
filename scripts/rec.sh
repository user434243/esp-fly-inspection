#!/bin/bash
# Record the whole screen to ~/Videos/drone_demo/  -  press q (or Ctrl+C) in this terminal to stop
mkdir -p ~/Videos/drone_demo
OUT=~/Videos/drone_demo/demo_$(date +%Y%m%d_%H%M%S).mp4
RES=$(xdpyinfo 2>/dev/null | awk '/dimensions/{print $2}'); RES=${RES:-1920x1080}
if ffmpeg -hide_banner -loglevel error -f lavfi -i color=black:s=256x256 -frames:v 1 \
          -c:v h264_nvenc -f null - 2>/dev/null; then
    ENC="-c:v h264_nvenc -preset p4 -cq 23"; NAME="GPU (NVENC)"
else
    ENC="-c:v libx264 -preset ultrafast -crf 23"; NAME="CPU (x264)"
fi
echo "Recording $RES with $NAME -> $OUT   (press q here to stop)"
ffmpeg -hide_banner -loglevel error -stats -f x11grab -framerate 30 -video_size $RES -i $DISPLAY \
       $ENC -pix_fmt yuv420p "$OUT"
echo "Saved: $OUT"
