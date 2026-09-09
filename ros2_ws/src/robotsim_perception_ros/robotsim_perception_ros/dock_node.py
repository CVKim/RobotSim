# -*- coding: utf-8 -*-
"""dock_node — cart_node 가 낸 고리 포즈로 AGV 를 결합 위치까지 몰고 가는 도킹 제어 노드.

    cart_node ──/cart/status (String JSON: hook_plane_mm, cart.rim_yaw_deg, status)──▶ dock_node ──▶ /cmd_vel (geometry_msgs/Twist)
                                                                                          ├──▶ /dock/state (robotsim_interfaces/DockState)
                                                                                          └──▶ /dock/state_json (String JSON, 같은 내용)

  판단은 robotsim_perception.dock.DockController (ROS 없이 pytest). 측정은 평면 좌표의 고리 위치와 대차 요 — 데크 플레이트가 바닥과
  평행하므로 평면 좌표 = AGV 프레임이다(실제 시스템에서는 카메라→base_link 외참 TF 가 이 변환을 한다; /cart/hook_pose 를 TF 로 base_link 에
  옮기는 경로가 그것이다).
  좌표 변환 사슬 검사(tf_check): 같은 고리를 두 경로로 구해 맞는지 본다 — (1) cart_node 가 낸 `hook_cart_mm`,
  (2) `/cart/hook_pose`(tof_optical 좌표)를 tf2 의 `lookup_transform(cart <- tof_optical)` 로 옮긴 값. 두 값의 최대 차이를
  DockState.tf_check_mm 으로 낸다. TF 를 내기만 하고 아무도 쓰지 않으면 규약이 틀려도 조용히 넘어가므로, 쓰는 쪽을 하나 둔다.
  인식이 실패한 프레임(status != OK)에는 정지 명령을 낸다. 일정 시간 측정이 없으면(watchdog) 정지.
  stop-and-go (trigger_capture=true): 명령을 낸 뒤 /agv/rel_pose 가 moving=false 로 바뀌면 cart_node 의 ~/capture 를 불러 다음 프레임을
  받는다 — 측정이 항상 정지 포즈의 것이라 인식 지연이 제어에 들어가지 않는다 (연속 모드에서는 요가 발산했다, agv_sim_node 주석).

  파라미터: dock_v_mm(-200) lookahead_mm(250) v_max_mm_s(150) w_max_rad_s(0.5) tol_u_mm/tol_v_mm(10) tol_yaw_deg(2) watchdog_s(3.0)
            trigger_capture(true) capture_service('/robotsim_cart/capture') startup_delay_s(2.0)
            tf_check(true) cart_frame('cart')
"""
from __future__ import annotations

import json

import rclpy
import tf2_geometry_msgs                      # PoseStamped <-> tf2 변환 등록 (import 만으로 do_transform_* 이 붙는다)
from geometry_msgs.msg import PoseStamped, Twist
from std_srvs.srv import Trigger
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from robotsim_interfaces.msg import DockState
from robotsim_perception.dock import DockController


def transform_pose(pose_stamped: PoseStamped, tf):
    """tf2_geometry_msgs.do_transform_pose 의 시그니처가 배포판마다 다르다(Galactic: PoseStamped, Humble: Pose).
    둘 다 받아 Pose 를 돌려준다."""
    try:
        return tf2_geometry_msgs.do_transform_pose(pose_stamped.pose, tf)
    except (TypeError, AttributeError):
        return tf2_geometry_msgs.do_transform_pose(pose_stamped, tf).pose


class DockNode(Node):
    def __init__(self):
        super().__init__("dock_node")
        P = self.declare_parameter
        self.ctrl = DockController(dock_v_mm=float(P("dock_v_mm", -200.0).value), lookahead_mm=float(P("lookahead_mm", 250.0).value),
                                   v_max_mm_s=float(P("v_max_mm_s", 150.0).value), w_max_rad_s=float(P("w_max_rad_s", 0.5).value),
                                   tol_u_mm=float(P("tol_u_mm", 10.0).value), tol_v_mm=float(P("tol_v_mm", 10.0).value),
                                   tol_yaw_deg=float(P("tol_yaw_deg", 2.0).value))
        self.watchdog_s = float(P("watchdog_s", 3.0).value)
        self.trigger_capture = bool(P("trigger_capture", True).value)
        startup_delay = float(P("startup_delay_s", 2.0).value)
        self.cli = self.create_client(Trigger, str(P("capture_service", "/robotsim_cart/capture").value)) if self.trigger_capture else None
        self.awaiting_stop = False           # 명령을 냈고 AGV 가 멈추기를 기다리는 중
        self.cmd_seq_sent = 0
        self.retry_timer = None
        self.sub_agv = self.create_subscription(String, "/agv/rel_pose", self.on_agv, 10) if self.trigger_capture else None
        self.sub = self.create_subscription(String, "/cart/status", self.on_status, 10)
        self.pub_cmd = self.create_publisher(Twist, "/cmd_vel", 10)
        self.pub_state = self.create_publisher(DockState, "/dock/state", 10)
        self.pub_state_json = self.create_publisher(String, "/dock/state_json", 10)
        self.tf_check = bool(P("tf_check", True).value)
        self.cart_frame = str(P("cart_frame", "cart").value)
        self.tf_buffer = Buffer() if self.tf_check else None
        self.tf_listener = TransformListener(self.tf_buffer, self) if self.tf_check else None
        self.sub_hook = self.create_subscription(PoseStamped, "/cart/hook_pose", self.on_hook, 10) if self.tf_check else None
        self.last_hook = None
        self.tf_check_mm = -1.0
        self.steady = Clock(clock_type=ClockType.STEADY_TIME)
        self.last_rx = None
        self.n, self.misses, self.docked = 0, 0, False
        self.create_timer(0.5, self.on_watchdog, clock=self.steady)
        if self.trigger_capture:
            self.start_timer = self.create_timer(startup_delay, self.on_startup, clock=self.steady)
        self.get_logger().info(f"dock target: hook at v={self.ctrl.dock_v_mm:.0f} mm, tol u/v {self.ctrl.tol_u_mm}/{self.ctrl.tol_v_mm} mm, yaw {self.ctrl.tol_yaw_deg} deg")

    def on_hook(self, msg: PoseStamped):
        self.last_hook = msg

    def check_tf(self, st: dict):
        """같은 고리를 (1) cart_node 의 hook_cart_mm 과 (2) tf2 로 옮긴 /cart/hook_pose 로 각각 구해 최대 차이를 mm 로 남긴다."""
        if not self.tf_check or self.last_hook is None or not st.get("hook_cart_mm"):
            return
        src = self.last_hook.header.frame_id
        try:
            tf = self.tf_buffer.lookup_transform(self.cart_frame, src, Time())      # 최신 것 (측정과 같은 프레임에서 갱신된다)
        except TransformException as e:
            self.get_logger().debug(f"tf {self.cart_frame} <- {src} not available yet: {e}")
            return
        p = transform_pose(self.last_hook, tf).position
        got = (p.x * 1000.0, p.y * 1000.0, p.z * 1000.0)
        want = [float(v) for v in st["hook_cart_mm"]]
        self.tf_check_mm = max(abs(a - b) for a, b in zip(got, want))
        if self.tf_check_mm > 1.0:      # 1 mm 를 넘으면 TF 규약과 계산이 어긋난 것이다
            self.get_logger().warn(f"tf check: hook via TF {got[0]:.1f},{got[1]:.1f},{got[2]:.1f} mm vs "
                                   f"status {want[0]:.1f},{want[1]:.1f},{want[2]:.1f} mm (max {self.tf_check_mm:.2f} mm)")

    def on_status(self, msg: String):
        try:
            st = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self.last_rx = self.steady.now()
        self.check_tf(st)
        self.n += 1
        if self.docked:
            self._publish(0.0, 0.0, "docked", st)
            return
        if st.get("status") != "OK" or not st.get("hook_plane_mm"):
            self.misses += 1
            v_cmd, w_cmd = self.ctrl.on_miss()           # 잃었으면 잠시 기어가며 다시 찾고, 계속 없으면 정지
            self._publish(v_cmd, w_cmd, "no_measurement", st)
            return
        u, v, _ = st["hook_plane_mm"]
        yaw = float((st.get("cart") or {}).get("rim_yaw_deg", 0.0))
        v_cmd, w_cmd, done = self.ctrl.command(float(u), float(v), yaw)
        if done:
            self.docked = True
            e = self.ctrl.errors(float(u), float(v), yaw)
            self.get_logger().info(f"DOCKED after {self.n} frames: e_u {e['e_u']:.1f} mm e_v {e['e_v']:.1f} mm yaw {e['e_yaw']:.2f} deg (misses {self.misses})")
        self._publish(v_cmd, w_cmd, self.ctrl.mode, st, (float(u), float(v), yaw))

    def on_startup(self):
        self.start_timer.cancel()
        self.destroy_timer(self.start_timer)
        self.request_capture()

    def _arm_retry(self, delay_s: float):
        if self.retry_timer is not None:
            self.retry_timer.cancel()
            self.destroy_timer(self.retry_timer)
        self.retry_timer = self.create_timer(delay_s, self._retry_capture, clock=self.steady)

    def _retry_capture(self):
        if self.retry_timer is not None:
            self.retry_timer.cancel()
            self.destroy_timer(self.retry_timer)
            self.retry_timer = None
        self.request_capture()

    def request_capture(self):
        if self.cli is None or self.docked:
            return
        if not self.cli.service_is_ready():
            self.get_logger().warn("capture service not ready, retrying in 1 s")
            self._arm_retry(1.0)
            return
        self.cli.call_async(Trigger.Request())          # 응답은 /cart/status 토픽으로 온다 (on_status)

    def on_agv(self, msg: String):
        """AGV 가 명령 뒤에 멈추면 다음 촬영을 요청한다 (stop-and-go)."""
        if not self.awaiting_stop or self.docked:
            return
        try:
            d = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if not d.get("moving", False) and int(d.get("cmd_seq", 0)) >= self.cmd_seq_sent:
            self.awaiting_stop = False
            self.request_capture()

    def _publish(self, v_mm_s, w_rad_s, mode, st, meas=None):
        tw = Twist()
        tw.linear.x = float(v_mm_s) / 1000.0
        tw.angular.z = float(w_rad_s)
        self.pub_cmd.publish(tw)
        self.cmd_seq_sent += 1
        if self.trigger_capture and not self.docked:
            if v_mm_s == 0.0 and w_rad_s == 0.0:
                self._arm_retry(0.5)                   # 정지 명령(측정 실패 등): 잠시 뒤 다시 촬영
            else:
                self.awaiting_stop = True
        out = {"mode": mode, "docked": self.docked, "v_mm_s": round(float(v_mm_s), 1), "w_rad_s": round(float(w_rad_s), 3),
               "frames": self.n, "misses": self.misses, "tf_check_mm": round(float(self.tf_check_mm), 2)}
        err = self.ctrl.errors(*meas) if meas is not None else None
        if err is not None:
            out["errors"] = {k: round(val, 1) for k, val in err.items()}
        self.pub_state_json.publish(String(data=json.dumps(out, ensure_ascii=False)))

        ds = DockState()
        ds.header.stamp = self.get_clock().now().to_msg()
        ds.header.frame_id = self.cart_frame
        ds.mode, ds.docked = str(mode), bool(self.docked)
        if meas is not None:
            ds.hook_u_mm, ds.rim_v_mm, ds.yaw_deg = float(meas[0]), float(meas[1]), float(meas[2])
        if err is not None:
            ds.e_u_mm, ds.e_v_mm, ds.e_yaw_deg = float(err["e_u"]), float(err["e_v"]), float(err["e_yaw"])
        ds.v_mm_s, ds.w_rad_s = float(v_mm_s), float(w_rad_s)
        ds.frames, ds.misses = int(self.n), int(self.misses)
        ds.tf_check_mm = float(self.tf_check_mm)
        self.pub_state.publish(ds)

    def on_watchdog(self):
        if self.last_rx is not None and (self.steady.now() - self.last_rx).nanoseconds * 1e-9 > self.watchdog_s and not self.docked:
            self.pub_cmd.publish(Twist())          # 측정이 끊기면 정지
            if self.trigger_capture:
                self.awaiting_stop = False
                self.last_rx = self.steady.now()
                self.request_capture()             # 응답이 유실됐으면 다시 요청


def main(args=None):
    rclpy.init(args=args)
    node = DockNode()
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
