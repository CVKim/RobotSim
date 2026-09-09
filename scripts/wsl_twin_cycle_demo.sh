#!/usr/bin/env bash
# 트윈 연결 사이클(인식 노드 + pick_executor + twin_bridge, 카메라·로봇 = MuJoCo 트윈)을 헤드리스로 끝까지 돌리고
# rviz2 캡처 + 로그 + 결과 요약을 남긴다 (WSL2). 트윈 서버는 Windows 쪽에서 먼저 띄워져 있어야 한다:
#   .venv\Scripts\python.exe tools/twin_server.py --arm track --boxes 12 --seed 500
#   wsl -d Ubuntu-22.04 -e bash /mnt/e/Robot_Sim/scripts/wsl_twin_cycle_demo.sh [host] [rviz.png] [log] [max_wait_s]
# 트윈 프레임은 합성이므로 산출물은 공개 가능.
set -eo pipefail
HOST="${1:-}"; [ "$HOST" = "-" ] && HOST=""      # "-" 또는 비면 launch 가 기본 게이트웨이(Windows 호스트)를 쓴다 (PowerShell 은 빈 인자를 떨어뜨린다)
OUT_RVIZ="${2:-/mnt/e/Robot_Sim/assets/ros2_twin_cycle_rviz.png}"
LOG="${3:-/mnt/e/Robot_Sim/explore/ros2/twin_cycle.log}"
MAX_WAIT="${4:-1500}"
USE_ACTION="${5:-false}"      # true 면 픽 명령을 액션(/robot/execute_pick)으로 주고받는다
WS=/mnt/e/Robot_Sim/ros2_ws
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source "$WS/install/setup.bash"
export LIBGL_ALWAYS_SOFTWARE=1 QT_QPA_PLATFORM=xcb ROS_DOMAIN_ID=77
export FASTRTPS_DEFAULT_PROFILES_FILE="$WS/fastdds_no_shm.xml" RCUTILS_LOGGING_BUFFERED_STREAM=0 PYTHONUNBUFFERED=1
mkdir -p "$(dirname "$LOG")" "$(dirname "$OUT_RVIZ")"

Xvfb :99 -screen 0 1600x1000x24 >/dev/null 2>&1 &
XV=$!
LAUNCH=""
cleanup() {                   # 프로세스 그룹째 정리 — 잔류 노드는 다음 실행의 rviz 를 오염시킨다 (docs/42)
    [ -n "$LAUNCH" ] && kill -- -$LAUNCH 2>/dev/null || true
    sleep 2
    [ -n "$LAUNCH" ] && kill -9 -- -$LAUNCH 2>/dev/null || true
    kill $XV 2>/dev/null || true
}
trap cleanup EXIT
sleep 2

: > "$LOG"
ros2 daemon stop >/dev/null 2>&1 || true
HOST_ARG=""; [ -n "$HOST" ] && HOST_ARG="host:=$HOST"
DISPLAY=:99 setsid ros2 launch twin_bridge twin_cycle.launch.py $HOST_ARG use_action:="$USE_ACTION" >> "$LOG" 2>&1 &
LAUNCH=$!

# 세 번째 명령이 실행(실시간 재생)되는 중간에 캡처: 'cmd #3:' 로그를 기다린 뒤 4 초. 그 전에 DONE 이 나거나 launch 가 죽으면 바로 캡처
for i in $(seq 1 120); do
    if grep -qE 'cmd #3:|goal #3 |DONE|process has died' "$LOG" 2>/dev/null; then break; fi   # 액션 경로는 'goal #3' 로 찍힌다
    if ! kill -0 "$LAUNCH" 2>/dev/null; then echo "launch exited early"; break; fi
    sleep 2
done
sleep 4
import -display :99 -window root "$OUT_RVIZ"
echo "saved $OUT_RVIZ ($(grep -c 'cmd #[0-9]*:' "$LOG") commands issued so far)"

# DONE 까지 기다린다 (최대 MAX_WAIT 초)
t0=$(date +%s)
while ! grep -q 'DONE' "$LOG"; do
    if [ $(( $(date +%s) - t0 )) -gt "$MAX_WAIT" ]; then echo "timeout after $MAX_WAIT s"; break; fi
    if ! kill -0 "$LAUNCH" 2>/dev/null; then echo "launch exited before DONE"; break; fi
    sleep 5
done
sleep 3
echo "--- execution results (bridge log):"
grep -oE 'cmd #[0-9]+ -> [a-z_]+' "$LOG" | awk '{print $4}' | sort | uniq -c || true
echo "--- node log (pick_executor / bridge / perception):"
(grep -E 'pick #|cmd #|robot:|DONE|no command|exhausted|Traceback|Error|twin at' "$LOG" || true) | cut -c1-170 | tail -60
