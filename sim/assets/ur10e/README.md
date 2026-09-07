# UR10e MJCF (MuJoCo Menagerie)

출처: https://github.com/google-deepmind/mujoco_menagerie/tree/main/universal_robots_ur10e (commit 8161bba, 2026-09 기준)
라이선스: BSD-3-Clause (LICENSE 파일, ROS-Industrial universal_robot 에서 파생)

ur10e.xml 과 LICENSE 만 레포에 둔다. 메시(assets/*.obj, 34 MB)는 레포에 넣지 않고
`python scripts/fetch_ur10e_meshes.py` 로 받는다. 메시가 없으면 sim/arm.py 가 충돌용 캡슐을 보이게 해서 그대로 돈다.
