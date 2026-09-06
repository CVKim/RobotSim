#!/usr/bin/env bash
# ROS2 Humble 설치 (WSL2 Ubuntu 22.04) + 이 레포의 인식 패키지를 WSL python 에 연결.
#   wsl -d Ubuntu-22.04 -e bash /mnt/e/Robot_Sim/scripts/wsl_install_ros2.sh
# desktop 메타패키지 = rviz2 포함. xvfb/imagemagick 은 헤드리스 rviz 스크린샷용.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
APT="sudo DEBIAN_FRONTEND=noninteractive apt-get -y -o Dpkg::Options::=--force-confnew"

echo "[1/6] base packages"
$APT update
$APT install locales curl gnupg lsb-release software-properties-common
sudo locale-gen en_US en_US.UTF-8 >/dev/null
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
sudo add-apt-repository -y universe >/dev/null

echo "[2/6] ROS2 apt source"
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo "$UBUNTU_CODENAME") main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list >/dev/null
$APT update

echo "[3/6] ros-humble-desktop (rviz2 포함) + colcon"
$APT install ros-humble-desktop ros-dev-tools python3-colcon-common-extensions python3-pip \
    xvfb imagemagick

echo "[4/6] python deps for robotsim_perception"
python3 -m pip install --quiet --upgrade pip
# 22.04 의 시스템 setuptools(59)는 PEP 660 편집 설치를 못 한다 → 64+ 로 올린다
python3 -m pip install --quiet --upgrade "setuptools>=64,<80" wheel   # <80: colcon-core 0.21 호환
python3 -m pip install --quiet numpy "opencv-python-headless>=4.8" tifffile pytest

echo "[5/6] editable install of the repo package (deps already satisfied)"
python3 -m pip install --quiet -e /mnt/e/Robot_Sim --no-deps --no-build-isolation

echo "[5b] WSL2 clock: systemd-timesyncd 와 호스트 동기화가 번갈아 시각을 밀어 ±66 s 점프가 나던 문제 예방"
sudo systemctl disable --now systemd-timesyncd >/dev/null 2>&1 || true
sudo hwclock -w >/dev/null 2>&1 || true

echo "[6/6] smoke"
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
ros2 --help | head -1
python3 -c "import robotsim_perception, rclpy, sensor_msgs_py.point_cloud2; print('imports ok', robotsim_perception.__version__)"
echo INSTALL_OK
