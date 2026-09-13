#!/usr/bin/env bash
# MoveIt2(OMPL)로 같은 계획 문제를 풀어 우리 구현과 비교한다 (WSL2, 헤드리스).
#   (Windows) .venv\Scripts\python.exe tools\twin_path_plan.py --seeds 4 --dump-problems explore\twin\plan_problems.json
#   (WSL2)    bash /mnt/e/Robot_Sim/scripts/wsl_moveit_compare.sh
# 산출물: explore/twin/moveit_plan_result.json (+ 로그). 합성 장면만 쓰므로 공개 가능한 수치다.
set -eo pipefail
PROBLEMS="${1:-/mnt/e/Robot_Sim/explore/twin/plan_problems.json}"
OUT="${2:-/mnt/e/Robot_Sim/explore/twin/moveit_plan_result.json}"
LOG="${3:-/mnt/e/Robot_Sim/explore/ros2/moveit_compare.log}"
WS=/mnt/e/Robot_Sim/ros2_ws
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source "$WS/install/setup.bash"
export FASTRTPS_DEFAULT_PROFILES_FILE="$WS/fastdds_no_shm.xml" ROS_DOMAIN_ID=80 RCUTILS_LOGGING_BUFFERED_STREAM=0 PYTHONUNBUFFERED=1
mkdir -p "$(dirname "$LOG")" "$(dirname "$OUT")"
: > "$LOG"
ros2 daemon stop >/dev/null 2>&1 || true

MG=""
cleanup() {
    [ -n "$MG" ] && kill -- -$MG 2>/dev/null || true
    sleep 2
    [ -n "$MG" ] && kill -9 -- -$MG 2>/dev/null || true
}
trap cleanup EXIT

setsid ros2 launch robotsim_moveit_config move_group.launch.py >> "$LOG" 2>&1 &
MG=$!
for i in $(seq 1 60); do
    grep -q "MoveGroup context initialization complete\|Ready to take commands" "$LOG" && break
    sleep 1
done
sleep 2
ros2 run robotsim_moveit_config plan_compare.py --ros-args \
    -p problems:="$PROBLEMS" -p out:="$OUT" -p planning_time:=5.0 2>&1 | tee -a "$LOG" | tail -40
