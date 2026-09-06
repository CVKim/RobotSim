#!/usr/bin/env bash
# 헤드리스 rviz2 스크린샷 (WSL2). Xvfb 가상 디스플레이에서 노드 + rviz2 를 띄우고 ImageMagick 으로 캡처.
#   wsl -d Ubuntu-22.04 -e bash /mnt/e/Robot_Sim/scripts/wsl_rviz_screenshot.sh [source] [out.png]
# source 기본 synthetic (공개 가능). 실측 세션 경로를 주면 로컬 전용 산출물로만 쓸 것.
# 종료는 PID 로만 한다 — `pkill -f perception_node` 는 이 스크립트를 부른 셸(명령줄에 같은 문자열)까지 죽인다.
set -eo pipefail          # -u 는 ROS setup.bash 가 미정의 변수를 참조해 실패하므로 쓰지 않는다
SRC="${1:-synthetic}"
OUT="${2:-/mnt/e/Robot_Sim/assets/ros2_rviz_synthetic.png}"
WS=/mnt/e/Robot_Sim/ros2_ws
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source "$WS/install/setup.bash"

export DISPLAY=:99 LIBGL_ALWAYS_SOFTWARE=1 QT_QPA_PLATFORM=xcb ROS_DOMAIN_ID=77
# WSL2: Fast DDS 공유메모리 전송 비활성 (ros2_ws/fastdds_no_shm.xml), 로그는 리다이렉트 시에도 즉시 출력
export FASTRTPS_DEFAULT_PROFILES_FILE="$WS/fastdds_no_shm.xml" RCUTILS_LOGGING_BUFFERED_STREAM=0 PYTHONUNBUFFERED=1

Xvfb :99 -screen 0 1600x1000x24 >/dev/null 2>&1 &
XV=$!
sleep 2
# setsid 필수: 비대화형 셸의 백그라운드 잡으로 `ros2 run` 을 띄우면 타이머가 2프레임 뒤 멈춘다
# (WSL2 에서 실측: bg 0~2 프레임 vs setsid 14 프레임 / 9초). 자기 세션으로 분리하면 정상.
setsid ros2 run robotsim_perception_ros perception_node --ros-args -p "source:=$SRC" -p rate_hz:=1.0 \
    -p synthetic_pick_every:=10 > /tmp/perception_node.log 2>&1 &
NODE=$!
sleep 8
echo "node log lines: $(wc -l < /tmp/perception_node.log)"
echo "boxes hz: $(timeout 6 ros2 topic hz /perception/boxes 2>&1 | grep -m1 average || echo none)"

setsid rviz2 -d "$WS/src/robotsim_perception_ros/config/perception.rviz" > /tmp/rviz2.log 2>&1 &
RV=$!
sleep 22
echo "rviz subscribed to /perception/boxes: $(ros2 topic info -v /perception/boxes 2>/dev/null | grep -c 'Node name: rviz')"
echo "rviz subscribed to /tof/points:       $(ros2 topic info -v /tof/points 2>/dev/null | grep -c 'Node name: rviz')"
import -display :99 -window root "$OUT"
kill $RV $NODE $XV 2>/dev/null || true
echo "saved $OUT"
echo "--- rviz2 log (non-GL):"; grep -viE "ogre|GLX|mesa|libGL|^$|X11 connection" /tmp/rviz2.log | head -6 || true
echo "--- node log tail:"; tail -2 /tmp/perception_node.log | cut -c1-110
