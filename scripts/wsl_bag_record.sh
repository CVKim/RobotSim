#!/usr/bin/env bash
# 인식 -> 제어 사이클(합성 12박스)을 돌리면서 ros2 bag 으로 기록한다 (WSL2). 두 개를 남긴다:
#   1) 전체 bag  explore/ros2/bags/pick_cycle_full   — /tof/points 포함, rviz 재생용 (로컬 전용, 수십 MB)
#   2) 작은 bag  ros2_ws/src/pick_executor/test/data/pick_cycle_synth — /perception/pick_poses + /perception/status 만 (수십 KB, 커밋 가능:
#      합성 프레임에서 나온 포즈·상태 JSON 뿐이라 회사 데이터가 없다). launch_testing 재생 테스트(test_bag_replay_launch.py)의 입력.
#   wsl -d Ubuntu-22.04 -e bash /mnt/e/Robot_Sim/scripts/wsl_bag_record.sh
set -eo pipefail
WS=/mnt/e/Robot_Sim/ros2_ws
FULL=/mnt/e/Robot_Sim/explore/ros2/bags/pick_cycle_full
SMALL=$WS/src/pick_executor/test/data/pick_cycle_synth
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source "$WS/install/setup.bash"
export ROS_DOMAIN_ID=76 FASTRTPS_DEFAULT_PROFILES_FILE="$WS/fastdds_no_shm.xml" RCUTILS_LOGGING_BUFFERED_STREAM=0 PYTHONUNBUFFERED=1
mkdir -p "$(dirname "$FULL")" "$(dirname "$SMALL")"
rm -rf "$FULL" "$SMALL"
LOG=/mnt/e/Robot_Sim/explore/ros2/bag_record.log; : > "$LOG"

LAUNCH=""; REC1=""; REC2=""
cleanup() {
    for p in $REC1 $REC2 $LAUNCH; do kill -INT -- -$p 2>/dev/null || true; done
    sleep 3
    for p in $REC1 $REC2 $LAUNCH; do kill -9 -- -$p 2>/dev/null || true; done
}
trap cleanup EXIT
ros2 daemon stop >/dev/null 2>&1 || true

# 기록기를 먼저 띄운다 (토픽이 생기면 자동으로 붙는다: --include-hidden-topics 불필요, /tf_static 은 transient_local 이라 QoS 프로필을 준다)
setsid ros2 bag record -o "$FULL" --qos-profile-overrides-path "$WS/src/pick_executor/test/data/tf_static_qos.yaml" \
    /tof/points /perception/boxes /perception/pick_poses /perception/next_pick /perception/status /diagnostics \
    /robot/target_poses /robot/trajectory /pick_executor/state /tf /tf_static >> "$LOG" 2>&1 &
REC1=$!
setsid ros2 bag record -o "$SMALL" /perception/pick_poses /perception/status >> "$LOG" 2>&1 &
REC2=$!
sleep 4
setsid ros2 launch pick_executor pick_cycle.launch.py rviz:=false execute_time_s:=1.0 >> "$LOG" 2>&1 &
LAUNCH=$!
for i in $(seq 1 60); do
    if grep -q 'DONE' "$LOG"; then break; fi
    sleep 2
done
sleep 2
kill -INT -- -$LAUNCH 2>/dev/null || true; sleep 2
kill -INT -- -$REC1 -$REC2 2>/dev/null || true; sleep 4
echo "--- bags:"; ros2 bag info "$FULL" 2>/dev/null | grep -E 'Duration|Messages|Topic:' | cut -c1-120
ros2 bag info "$SMALL" 2>/dev/null | grep -E 'Duration|Messages|Topic:' | cut -c1-120
du -sh "$FULL" "$SMALL"
