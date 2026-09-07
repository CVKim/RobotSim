# -*- coding: utf-8 -*-
"""agv_sim_node — 차동 구동 AGV 의 운동학 시뮬레이터. /cmd_vel 을 적분해 대차와의 상대 포즈를 낸다.

    dock_node ──/cmd_vel (Twist)──▶ agv_sim_node ──/agv/rel_pose (String JSON: hook_u_mm, rim_v_mm, yaw_deg)──▶ cart_node(source=synthetic_dock)

  상대 포즈 정의·운동학은 robotsim_perception.dock (RelPose, step). cart_node 는 이 포즈로 합성 대차 장면을 렌더하므로,
  AGV 가 움직이면 다음 프레임의 고리 위치가 바뀐다 — 인식 → 제어 → 플랜트 폐루프.
  stop-and-go: 명령 하나를 cmd_hold_s(0.5 s) 동안 적용하고 멈춘다. dock_node 는 AGV 가 멈춘 것을 보고(/agv/rel_pose 의 moving=false)
  다음 촬영을 요청하므로, 측정은 항상 정지한 포즈의 것이다 — 연속 주행 모드에서는 인식 지연(~0.45 s)+명령 유지(~0.5 s)의 묵은 명령으로
  요가 발산했다(실측). 실제 도킹도 저속 접근에서는 이 방식이 안전하다.
  파라미터: init_hook_u_mm, init_rim_v_mm, init_yaw_deg, rate_hz(20), cmd_hold_s(0.5; 0 이면 연속 모드 = cmd_timeout_s 동안 유지), cmd_timeout_s(1.0)
"""
from __future__ import annotations

import json

import rclpy
from geometry_msgs.msg import Twist
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from std_msgs.msg import String

from robotsim_perception.dock import RelPose, step


class AgvSimNode(Node):
    def __init__(self):
        super().__init__("agv_sim")
        P = self.declare_parameter
        self.rp = RelPose(float(P("init_hook_u_mm", -80.0).value), float(P("init_rim_v_mm", -450.0).value),
                          float(P("init_yaw_deg", 5.0).value))
        rate = float(P("rate_hz", 20.0).value)
        self.cmd_timeout = float(P("cmd_timeout_s", 1.0).value)
        self.cmd_hold = float(P("cmd_hold_s", 0.5).value)
        self.dt = 1.0 / rate
        self.v_mm_s, self.w_rad_s = 0.0, 0.0
        self.steady = Clock(clock_type=ClockType.STEADY_TIME)
        self.last_cmd = None
        self.cmd_seq = 0
        self.sub = self.create_subscription(Twist, "/cmd_vel", self.on_cmd, 10)
        self.pub = self.create_publisher(String, "/agv/rel_pose", 10)
        self.create_timer(self.dt, self.on_tick, clock=self.steady)
        self.t = 0.0
        self.get_logger().info(f"AGV start: hook_u {self.rp.hook_u_mm:.0f} mm, rim_v {self.rp.rim_v_mm:.0f} mm, yaw {self.rp.yaw_deg:.1f} deg")

    def on_cmd(self, msg: Twist):
        self.v_mm_s = float(msg.linear.x) * 1000.0
        self.w_rad_s = float(msg.angular.z)
        self.last_cmd = self.steady.now()
        self.cmd_seq += 1

    def on_tick(self):
        age = None if self.last_cmd is None else (self.steady.now() - self.last_cmd).nanoseconds * 1e-9
        hold = self.cmd_hold if self.cmd_hold > 0 else self.cmd_timeout
        if age is None or age > hold:
            v, w = 0.0, 0.0
        else:
            v, w = self.v_mm_s, self.w_rad_s
        self.rp = step(self.rp, v, w, self.dt, substeps=2)
        self.t += self.dt
        moving = bool(v != 0.0 or w != 0.0)
        self.pub.publish(String(data=json.dumps(dict(self.rp.to_dict(), t=round(self.t, 3), v_mm_s=round(v, 1), w_rad_s=round(w, 3),
                                                     moving=moving, cmd_seq=self.cmd_seq))))


def main(args=None):
    rclpy.init(args=args)
    node = AgvSimNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
