#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""같은 계획 문제를 MoveIt2(OMPL)로 풀어 우리 구현(sim/plan_rrt.py)과 비교한다.

    ros2 launch robotsim_moveit_config move_group.launch.py
    ros2 run robotsim_moveit_config plan_compare.py --ros-args \\
        -p problems:=/mnt/e/Robot_Sim/explore/twin/plan_problems.json \\
        -p out:=/mnt/e/Robot_Sim/explore/twin/moveit_plan_result.json

문제 파일은 Windows 쪽에서 `tools/twin_path_plan.py --dump-problems` 가 만든다: 트윈의 충돌 세계(원시 도형)와
시작·목표 관절각이 들어 있다. 여기서는 그 도형들을 MoveIt 계획 장면에 올리고, 관절 공간 목표로 계획을 요청해
성공 여부·계획 시간·관절 이동량을 기록한다. **같은 세계, 같은 문제**로 두 계획기를 비교하는 것이 목적이다.

실행(컨트롤러)은 붙이지 않는다. 궤적을 실제로 도는 것은 MuJoCo 트윈이다.
"""
import json
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import (CollisionObject, Constraints, JointConstraint, MotionPlanRequest,
                             PlanningScene, RobotState, WorkspaceParameters)
from moveit_msgs.srv import ApplyPlanningScene, GetMotionPlan
from rclpy.node import Node
from sensor_msgs.msg import JointState
from shape_msgs.msg import Plane, SolidPrimitive


def quat_from_mat(mat):
    """행 우선 3x3 (MuJoCo geom_xmat) -> 쿼터니언 (x, y, z, w)."""
    m = np.asarray(mat, float).reshape(3, 3)
    t = float(np.trace(m))
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
        elif i == 1:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return x / n, y / n, z / n, w / n


def collision_objects(geoms, frame="base_link"):
    """트윈 도형 목록 -> MoveIt CollisionObject 목록 (박스·원기둥·바닥 평면)."""
    out = []
    for g in geoms:
        co = CollisionObject()
        co.header.frame_id = frame
        co.id = g["name"]
        co.operation = CollisionObject.ADD
        p = Pose()
        p.position.x, p.position.y, p.position.z = (float(v) for v in g["pos"])
        qx, qy, qz, qw = quat_from_mat(g["mat"])
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = qx, qy, qz, qw
        if g["kind"] == "box":
            s = SolidPrimitive()
            s.type = SolidPrimitive.BOX
            s.dimensions = [2 * float(g["size"][0]), 2 * float(g["size"][1]), 2 * float(g["size"][2])]
            co.primitives.append(s)
            co.primitive_poses.append(p)
        elif g["kind"] == "cylinder":
            s = SolidPrimitive()
            s.type = SolidPrimitive.CYLINDER
            s.dimensions = [2 * float(g["size"][1]), float(g["size"][0])]      # [높이, 반지름]
            co.primitives.append(s)
            co.primitive_poses.append(p)
        elif g["kind"] == "plane":
            pl = Plane()
            pl.coef = [0.0, 0.0, 1.0, -float(g["pos"][2])]                     # z = pos_z 바닥
            co.planes.append(pl)
            co.plane_poses.append(Pose(orientation=p.orientation))
        else:
            continue
        out.append(co)
    return out


class PlanCompare(Node):
    def __init__(self):
        super().__init__("plan_compare")
        self.problems_path = self.declare_parameter("problems", "").value
        self.out_path = self.declare_parameter("out", "").value
        self.group = self.declare_parameter("group", "arm").value
        self.planning_time = float(self.declare_parameter("planning_time", 5.0).value)
        self.attempts = int(self.declare_parameter("attempts", 1).value)
        self.scene_cli = self.create_client(ApplyPlanningScene, "/apply_planning_scene")
        self.plan_cli = self.create_client(GetMotionPlan, "/plan_kinematic_path")

    def wait(self):
        for cli, name in ((self.scene_cli, "/apply_planning_scene"), (self.plan_cli, "/plan_kinematic_path")):
            while not cli.wait_for_service(timeout_sec=2.0):
                self.get_logger().info(f"{name} 기다리는 중 (move_group 이 떠 있나?)")

    def apply_scene(self, geoms):
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = collision_objects(geoms)
        req = ApplyPlanningScene.Request(scene=scene)
        fut = self.scene_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=20.0)
        ok = bool(fut.result() and fut.result().success)
        self.get_logger().info(f"계획 장면에 도형 {len(scene.world.collision_objects)}개 올림: {ok}")
        return ok

    def plan(self, joint_names, start, goal):
        req = MotionPlanRequest()
        req.group_name = self.group
        req.allowed_planning_time = self.planning_time
        req.num_planning_attempts = self.attempts
        req.workspace_parameters = WorkspaceParameters()
        req.workspace_parameters.header.frame_id = "base_link"
        req.workspace_parameters.min_corner.x = req.workspace_parameters.min_corner.y = -3.0
        req.workspace_parameters.min_corner.z = -1.0
        req.workspace_parameters.max_corner.x = req.workspace_parameters.max_corner.y = 3.0
        req.workspace_parameters.max_corner.z = 3.0
        rs = RobotState()
        rs.joint_state = JointState(name=list(joint_names), position=[float(v) for v in start])
        req.start_state = rs
        c = Constraints()
        for n, v in zip(joint_names, goal):
            jc = JointConstraint()
            jc.joint_name = n
            jc.position = float(v)
            jc.tolerance_above = jc.tolerance_below = 1e-3
            jc.weight = 1.0
            c.joint_constraints.append(jc)
        req.goal_constraints.append(c)

        t0 = time.perf_counter()
        fut = self.plan_cli.call_async(GetMotionPlan.Request(motion_plan_request=req))
        rclpy.spin_until_future_complete(self, fut, timeout_sec=self.planning_time + 15.0)
        wall = time.perf_counter() - t0
        res = fut.result()
        if res is None:
            return {"ok": False, "code": "no_response", "wall_s": round(wall, 3)}
        r = res.motion_plan_response
        pts = r.trajectory.joint_trajectory.points
        cost = float(sum(float(np.abs(np.asarray(b.positions) - np.asarray(a.positions)).sum())
                         for a, b in zip(pts[:-1], pts[1:]))) if len(pts) > 1 else 0.0
        return {"ok": int(r.error_code.val) == 1, "code": int(r.error_code.val),
                "plan_s": round(float(r.planning_time), 3), "wall_s": round(wall, 3),
                "points": len(pts), "cost_rad": round(cost, 3)}


def main():
    rclpy.init()
    node = PlanCompare()
    node.wait()
    data = json.loads(open(node.problems_path, encoding="utf-8").read())
    joint_names = data["joint_names"]
    rows = []
    for scene in data["scenes"]:
        node.apply_scene(scene["geoms"])
        time.sleep(1.0)                       # 장면이 반영될 시간
        for p in scene["problems"]:
            r = node.plan(joint_names, p["start"], p["goal"])
            r.update({"pair": p["pair"], "seed": scene["seed"], "straight_ok": p["straight_ok"]})
            rows.append(r)
            node.get_logger().info(f"시드 {scene['seed']} {p['pair']}: "
                                   f"{'성공' if r['ok'] else '실패(' + str(r['code']) + ')'} "
                                   f"계획 {r.get('plan_s', '-')} s 비용 {r.get('cost_rad', '-')}")
    ok = [r for r in rows if r["ok"]]
    summary = {"what": "MoveIt2(OMPL RRTConnect)로 같은 문제를 계획한 기록",
               "n": len(rows), "ok": len(ok),
               "plan_s_median": round(float(np.median([r["plan_s"] for r in ok])), 3) if ok else None,
               "cost_rad_median": round(float(np.median([r["cost_rad"] for r in ok])), 3) if ok else None,
               "runs": rows}
    if node.out_path:
        open(node.out_path, "w", encoding="utf-8", newline="\n").write(json.dumps(summary, ensure_ascii=False, indent=1))
        node.get_logger().info(f"saved {node.out_path}")
    node.get_logger().info(f"MoveIt 계획 {len(ok)}/{len(rows)} 성공")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
