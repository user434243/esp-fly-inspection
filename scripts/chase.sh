#!/bin/bash
# 3rd-person chase camera in Gazebo.   chase [behind_m] [above_m]   e.g.  chase 0.6 0.25
B=${1:-0.6}; U=${2:-0.25}
NAME=$(gz model --list 2>/dev/null | grep -oE '(cf_[0-9]+|crazyflie[_0-9]*)' | head -1)
if [ -z "$NAME" ]; then
    echo "Drone not found. Models in the world:"; gz model --list; exit 1
fi
echo "Following $NAME: ${B} m behind, ${U} m above"
gz service -s /gui/follow --reqtype gz.msgs.StringMsg --reptype gz.msgs.Boolean \
   --timeout 3000 --req "data: \"$NAME\""
sleep 0.5
gz service -s /gui/follow/offset --reqtype gz.msgs.Vector3d --reptype gz.msgs.Boolean \
   --timeout 3000 --req "x: -$B, y: 0, z: $U"
