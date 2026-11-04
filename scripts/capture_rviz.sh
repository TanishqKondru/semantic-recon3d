#!/usr/bin/env bash
# Screenshot RViz after the pipeline has played frames [0, STOP) in lockstep.
#   source ~/anaconda3/bin/activate sr3d-ros && bash scripts/capture_rviz.sh 185 docs/figures/ros_rviz.png
# Needs a display with OpenGL (WSLg works; conda's Xvfb has no GLX). The RViz window is grabbed directly,
# because WSLg's rootless X server cannot read the root window.
set -euo pipefail
STOP=${1:-185}
OUT=${2:-docs/figures/ros_rviz.png}
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
QT_QPA_PLATFORM=xcb rviz2 -d launch/proximity.rviz >outputs/ros/rviz.log 2>&1 &
RVIZ=$!
trap 'kill $RVIZ 2>/dev/null || true' EXIT
sleep 8
ros2 launch launch/proximity.launch.py lockstep:=true stop:="$STOP" log:=outputs/ros/capture_timing.csv \
  >outputs/ros/capture.log 2>&1
sleep 4
python - "$OUT" <<'EOF'
import sys
from PIL import Image
from Xlib import X, display

def find(w):
    for c in w.query_tree().children:
        if "RViz" in str(c.get_wm_name() or ""):
            return c
        if (hit := find(c)) is not None:
            return hit

win = find(display.Display().screen().root)
g = win.get_geometry()
raw = win.get_image(0, 0, g.width, g.height, X.ZPixmap, 0xFFFFFFFF)
Image.frombytes("RGB", (g.width, g.height), raw.data, "raw", "BGRX").save(sys.argv[1])
print("saved", sys.argv[1])
EOF
