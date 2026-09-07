#!/usr/bin/env bash
# 대차 도킹 폐루프(cart_node 합성 대차 + dock_node + agv_sim_node + rviz2)를 헤드리스로 돌려 rviz 캡처와 로그를 남긴다 (WSL2).
#   wsl -d Ubuntu-22.04 -- bash -c "bash /mnt/e/Robot_Sim/scripts/wsl_dock_demo.sh"
# 합성 장면만 쓰므로 산출물은 공개 가능.
set -eo pipefail
OUT_RVIZ="${1:-/mnt/e/Robot_Sim/assets/ros2_dock_rviz.png}"
LOG="${2:-/mnt/e/Robot_Sim/explore/ros2/dock_demo.log}"
WS=/mnt/e/Robot_Sim/ros2_ws
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source "$WS/install/setup.bash"
export LIBGL_ALWAYS_SOFTWARE=1 QT_QPA_PLATFORM=xcb ROS_DOMAIN_ID=78
export FASTRTPS_DEFAULT_PROFILES_FILE="$WS/fastdds_no_shm.xml" RCUTILS_LOGGING_BUFFERED_STREAM=0 PYTHONUNBUFFERED=1
mkdir -p "$(dirname "$LOG")" "$(dirname "$OUT_RVIZ")"
Xvfb :97 -screen 0 1600x1000x24 >/dev/null 2>&1 &
XV=$!
LAUNCH=""
cleanup() {
    [ -n "$LAUNCH" ] && kill -- -$LAUNCH 2>/dev/null || true
    sleep 2
    [ -n "$LAUNCH" ] && kill -9 -- -$LAUNCH 2>/dev/null || true
    kill $XV 2>/dev/null || true
}
trap cleanup EXIT
sleep 2
: > "$LOG"
ros2 daemon stop >/dev/null 2>&1 || true
# rviz 가 뜬 뒤 도킹이 시작되게 startup 을 늦춘다 (캡처용). 시작 자세: 측방 120 mm, 거리 480 mm, 요 −6°
DISPLAY=:97 setsid ros2 launch robotsim_perception_ros dock.launch.py init_hook_u_mm:=120.0 init_rim_v_mm:=-480.0 init_yaw_deg:=-6.0 >> "$LOG" 2>&1 &
LAUNCH=$!
# 도킹 중간(6번째 프레임 즈음)에 캡처
for i in $(seq 1 60); do
    n=$(grep -c 'dock#' "$LOG" 2>/dev/null || true)
    if [ "${n:-0}" -ge 14 ] || grep -q 'DOCKED' "$LOG"; then break; fi
    sleep 1
done
sleep 2
import -display :97 -window root "$OUT_RVIZ"
echo "saved $OUT_RVIZ (frames so far: $(grep -c 'dock#' "$LOG"))"
for i in $(seq 1 90); do
    if grep -q 'DOCKED' "$LOG"; then break; fi
    if ! kill -0 "$LAUNCH" 2>/dev/null; then break; fi
    sleep 1
done
echo "--- cart/dock log:"
(grep -E 'dock#|DOCKED|dock target|AGV start' "$LOG" || true) | cut -c1-200 | tail -30
