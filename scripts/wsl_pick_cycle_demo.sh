#!/usr/bin/env bash
# 인식 -> 제어 -> 재촬영 사이클 데모를 헤드리스로 돌려 rviz2 캡처 + rqt_graph 캡처 + 로그를 남긴다 (WSL2).
#   wsl -d Ubuntu-22.04 -e bash /mnt/e/Robot_Sim/scripts/wsl_pick_cycle_demo.sh [rviz.png] [graph.png] [log]
# 합성 소스만 쓰므로 산출물은 공개 가능.
set -eo pipefail
OUT_RVIZ="${1:-/mnt/e/Robot_Sim/assets/ros2_pick_cycle_rviz.png}"
OUT_GRAPH="${2:-/mnt/e/Robot_Sim/assets/ros2_pick_cycle_rqt_graph.png}"
LOG="${3:-/mnt/e/Robot_Sim/explore/ros2/pick_cycle.log}"
WS=/mnt/e/Robot_Sim/ros2_ws
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source "$WS/install/setup.bash"
export LIBGL_ALWAYS_SOFTWARE=1 QT_QPA_PLATFORM=xcb ROS_DOMAIN_ID=77
export FASTRTPS_DEFAULT_PROFILES_FILE="$WS/fastdds_no_shm.xml" RCUTILS_LOGGING_BUFFERED_STREAM=0 PYTHONUNBUFFERED=1
mkdir -p "$(dirname "$LOG")"

Xvfb :99 -screen 0 1600x1000x24 >/dev/null 2>&1 &
XV1=$!
Xvfb :98 -screen 0 1400x800x24 >/dev/null 2>&1 &
XV2=$!
LAUNCH=""; RQ=""
cleanup() {                   # 어떤 이유로 끝나도(set -e 포함) 프로세스 그룹째 정리 — 잔류 노드가 다음 실행의 rviz 를 오염시킨다
    [ -n "$LAUNCH" ] && kill -- -$LAUNCH 2>/dev/null || true
    [ -n "$RQ" ] && kill -- -$RQ 2>/dev/null || true
    sleep 2
    [ -n "$LAUNCH" ] && kill -9 -- -$LAUNCH 2>/dev/null || true
    [ -n "$RQ" ] && kill -9 -- -$RQ 2>/dev/null || true
    kill $XV1 $XV2 2>/dev/null || true
}
trap cleanup EXIT
sleep 2

# launch 는 setsid 로 자기 세션에 (비대화형 셸 백그라운드 잡의 타이머 정지 문제 회피). rviz 는 :99 에.
# 로그는 append 모드로 연다: 아래 tee -a 와 같은 파일을 쓰므로 둘 다 O_APPEND 여야 서로 덮어쓰지 않는다.
: > "$LOG"
DISPLAY=:99 setsid ros2 launch pick_executor pick_cycle.launch.py execute_time_s:=2.0 >> "$LOG" 2>&1 &
LAUNCH=$!
sleep 11                      # 첫 촬영(3 s 뒤) + 사이클 3~4개 진행된 시점: 남은 박스 8개 안팎 + 궤적이 같이 보인다
import -display :99 -window root "$OUT_RVIZ"
echo "saved $OUT_RVIZ (t=11 s)"
# ros2 CLI 데몬이 '!rclpy.ok()' 상태로 남아 있으면 echo 가 xmlrpc Fault 로 죽는다 (겪음) -> 데몬 없이 직접 디스커버리
ros2 daemon stop >/dev/null 2>&1 || true
echo "target_poses (once):"; (timeout 20 ros2 topic echo --no-daemon /robot/target_poses --once || true) | head -40 | tee -a "$LOG"
echo "state (once):";        (timeout 20 ros2 topic echo --no-daemon /pick_executor/state --once || true) | tee -a "$LOG"

# rqt_graph 는 다른 디스플레이에서 (노드들이 살아 있는 동안 찍어야 그래프가 나온다).
# rqt 는 -geometry 를 안 받고 그래프 종류도 CLI 옵션이 없다. 설정 ini 를 손으로 고쳐 봤더니 위젯 상태가 뒤엉켜 빈 그래프가
# 나왔다(겪음) -> ini 를 지워 기본 상태로 띄우고, xdotool 로 창 크기·그래프 종류(Nodes/Topics (active))·tf 숨김을 조작한다.
rm -f ~/.config/ros.org/rqt_gui.ini
# 두 노드가 다 뜬 뒤에 rqt_graph 를 띄운다 (안 그러면 빈 그래프가 찍힌다)
for _ in 1 2 3 4 5 6; do
    if ros2 node list --no-daemon 2>/dev/null | grep -q pick_executor; then break; fi
    sleep 2
done
DISPLAY=:98 setsid rqt_graph > /tmp/rqt_graph.log 2>&1 &
RQ=$!
sleep 12
if command -v xdotool >/dev/null; then
    export DISPLAY=:98
    xdotool search --onlyvisible --name '.' windowsize %@ 1380 780 windowmove %@ 0 0 2>/dev/null || true
    sleep 3
    xdotool mousemove 190 42 click 1 2>/dev/null; sleep 1; xdotool key Down Return 2>/dev/null; sleep 2   # 그래프 종류 -> 2번째 항목
    xdotool mousemove 335 104 click 1 2>/dev/null; sleep 2                                              # Hide: tf (rviz 의 TF 리스너 숨김)
    xdotool mousemove 23 42 click 1 2>/dev/null; sleep 4                                                # 새로고침: 커진 창에 맞춰 다시 그림
    export DISPLAY=:99
fi
import -display :98 -window root "$OUT_GRAPH"
echo "saved $OUT_GRAPH"

sleep 20                      # 나머지 사이클이 끝나 DONE 이 찍힐 때까지
echo "--- node log (pick_executor / perception):"
(grep -E 'pick #|DONE|skip|no command|exhausted|requesting first|Traceback|Error' "$LOG" || true) | cut -c1-170 | tail -24
