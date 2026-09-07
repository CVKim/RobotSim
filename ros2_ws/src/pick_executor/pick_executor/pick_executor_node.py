# -*- coding: utf-8 -*-
"""pick_executor — 인식 노드의 픽 포즈를 받아 로봇 명령(3점 궤적)을 내고, 재촬영을 요청해 사이클을 닫는 노드.

    /perception/pick_poses (PoseArray, base_link) ─┐
    /perception/status     (String JSON)          ─┴─▶ pick_executor ──▶ /robot/target_poses (PoseArray: pre-pick, pick, lift)
                                                        │               ──▶ /robot/trajectory   (MarkerArray, rviz 표시용)
                                                        │               ──▶ /pick_executor/state (String JSON)
                                                        └── (trigger_capture) ──▶ /robotsim_perception/capture (std_srvs/Trigger 클라이언트)

  이 노드가 보여 주는 ROS 개념: 구독 2개를 한 프레임 단위로 짝지어 판단(스탬프로 짝 맞춤), 퍼블리시, **서비스 클라이언트**
  (비동기 future), 단조 시계 타이머로 '실행 시간' 흉내(one-shot), 감시 타이머, 상태 기계, 파라미터.
  판단 로직은 logic.py 에 있어 ROS 없이 테스트된다.

  파라미터
    min_dist_m      0.05   직전 명령의 pick 위치와 이 거리 안이면 같은 장면으로 보고 보내지 않음
    clearance_m     0.15   pre-pick 거리 (접근 방향 반대)
    lift_m          0.25   들어올리는 높이
    execute_time_s  2.0    로봇 실행 흉내 (이 시간 뒤 재촬영 요청). 건너뛴 프레임 뒤 재촬영 간격으로도 쓴다
    trigger_capture false  true 면 인식 노드의 ~/capture 를 호출해 사이클을 닫는다 (인식 노드는 rate_hz:=0 트리거 모드로)
    capture_service '/robotsim_perception/capture'
    stop_on_empty   true   LAYER_EMPTY 가 연속 empty_retries(2)+1 번이거나 소스 소진이면 DONE (한 번의 층 선택 실패로 멈추지 않게)
    empty_retries   2
    watchdog_s      10.0   촬영 요청 뒤 이 시간 안에 판단이 없으면 오류 로그 + 재요청 (메시지 유실 대비)
    startup_delay_s 3.0    기동 후 첫 촬영 요청까지 대기 (디스커버리)
    done_topic      ''     비어 있지 않으면 이 토픽(String JSON {"ok","result",...})의 로봇 완료 보고로 실행을 끝낸다
                           (execute_time_s 타이머 대신). 트윈 브리지(twin_bridge)가 /robot/execution_result 로 낸다.
    execute_timeout_s 120  done_topic 모드에서 이 시간 안에 보고가 없으면 오류 로그 후 재촬영

  실제 셀에서는 /robot/target_poses 를 로봇 드라이버(또는 MoveIt) 가 받고 완료 신호를 준다 — done_topic 모드가 그 형태다.
  execute_time_s 타이머 모드는 로봇이 없을 때의 흉내다.
"""
from __future__ import annotations

import json

import rclpy
from geometry_msgs.msg import Point, Pose, PoseArray, Quaternion, Vector3
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from std_msgs.msg import ColorRGBA, Header, String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from .logic import Executor, Trajectory


def _header(stamp, frame_id: str) -> Header:
    h = Header()
    h.stamp = stamp
    h.frame_id = frame_id
    return h


def trajectory_to_posearray(traj: Trajectory, stamp, frame_id: str) -> PoseArray:
    pa = PoseArray()
    pa.header = _header(stamp, frame_id)
    qx, qy, qz, qw = traj.orientation
    for p in traj.points():
        pose = Pose()
        pose.position = Point(x=p[0], y=p[1], z=p[2])
        pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        pa.poses.append(pose)
    return pa


def trajectory_markers(traj, stamp, frame_id: str, ns: str = "traj") -> MarkerArray:
    """pre-pick -> pick -> lift 를 잇는 선 + 점 3개. 첫 요소는 DELETEALL (고유 ns).

    traj 가 None 이면 고정 id 4개를 명시적으로 DELETE 한다 — rviz2(Humble) 가 MarkerArray 안의 DELETEALL 을 확실히
    처리하지 않아 이전 마커가 남는 것을 실측했기 때문(robotsim_perception_ros/msgs.py 주석)."""
    ma = MarkerArray()
    clear = Marker()
    clear.header = _header(stamp, frame_id)
    clear.ns, clear.id, clear.action, clear.type = ns + "_clear", 0, Marker.DELETEALL, Marker.SPHERE
    clear.scale = Vector3(x=0.1, y=0.1, z=0.1)
    clear.pose.orientation = Quaternion(w=1.0)
    clear.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
    ma.markers.append(clear)
    if traj is None:
        for mid in range(4):
            d = Marker()
            d.header = _header(stamp, frame_id)
            d.ns, d.id, d.action = ns, mid, Marker.DELETE
            ma.markers.append(d)
        return ma
    line = Marker()
    line.header = _header(stamp, frame_id)
    line.ns, line.id, line.type, line.action = ns, 0, Marker.LINE_STRIP, Marker.ADD
    line.pose.orientation = Quaternion(w=1.0)
    line.scale = Vector3(x=0.012, y=0.0, z=0.0)
    line.color = ColorRGBA(r=1.0, g=0.85, b=0.2, a=1.0)
    line.points = [Point(x=p[0], y=p[1], z=p[2]) for p in traj.points()]
    ma.markers.append(line)
    for i, (p, rgb) in enumerate(zip(traj.points(), [(0.2, 0.7, 1.0), (1.0, 0.3, 0.3), (0.3, 1.0, 0.4)])):
        s = Marker()
        s.header = _header(stamp, frame_id)
        s.ns, s.id, s.type, s.action = ns, i + 1, Marker.SPHERE, Marker.ADD
        s.pose.position = Point(x=p[0], y=p[1], z=p[2])
        s.pose.orientation = Quaternion(w=1.0)
        s.scale = Vector3(x=0.04, y=0.04, z=0.04)
        s.color = ColorRGBA(r=rgb[0], g=rgb[1], b=rgb[2], a=1.0)
        ma.markers.append(s)
    return ma


class PickExecutorNode(Node):
    def __init__(self):
        super().__init__("pick_executor")
        P = self.declare_parameter
        self.ex = Executor(min_dist_m=float(P("min_dist_m", 0.05).value),
                           clearance_m=float(P("clearance_m", 0.15).value),
                           lift_m=float(P("lift_m", 0.25).value),
                           stop_on_empty=bool(P("stop_on_empty", True).value),
                           empty_retries=int(P("empty_retries", 2).value),
                           max_no_progress=int(P("max_no_progress", 12).value))
        self.last_cmd_stamp = None
        self.execute_time_s = float(P("execute_time_s", 2.0).value)
        self.done_topic = str(P("done_topic", "").value)
        self.execute_timeout_s = float(P("execute_timeout_s", 120.0).value)
        self.trigger_capture = bool(P("trigger_capture", False).value)
        capture_service = str(P("capture_service", "/robotsim_perception/capture").value)
        startup_delay = float(P("startup_delay_s", 3.0).value)
        self.watchdog_s = float(P("watchdog_s", 10.0).value)
        self.frame_id = "base_link"

        self.sub_poses = self.create_subscription(PoseArray, "/perception/pick_poses", self.on_poses, 10)
        self.sub_status = self.create_subscription(String, "/perception/status", self.on_status, 10)
        self.pub_targets = self.create_publisher(PoseArray, "/robot/target_poses", 10)
        self.pub_traj = self.create_publisher(MarkerArray, "/robot/trajectory", 10)
        self.pub_state = self.create_publisher(String, "/pick_executor/state", 10)
        self.cli = self.create_client(Trigger, capture_service) if self.trigger_capture else None
        self.sub_done = (self.create_subscription(String, self.done_topic, self.on_robot_done, 10)
                         if self.done_topic else None)

        # 타이머는 단조 시계로 (WSL2 벽시계 점프 대책 — 인식 노드와 같은 이유). rclpy 타이머는 주기적이므로
        # one-shot 은 콜백에서 cancel 하고, 새로 만들 때는 반드시 이전 것을 cancel+destroy 한다 (안 하면 참조를 잃은
        # 타이머가 영원히 주기적으로 울린다 — 검토에서 지적된 결함).
        self.steady = Clock(clock_type=ClockType.STEADY_TIME)
        self.exec_timer = None
        self.watchdog = None
        self.start_timer = self.create_timer(startup_delay, self.on_startup, clock=self.steady) if self.trigger_capture else None
        self.get_logger().info(
            f"min_dist={self.ex.min_dist_m} m clearance={self.ex.clearance_m} lift={self.ex.lift_m} "
            f"execute_time={self.execute_time_s}s trigger_capture={self.trigger_capture} -> {capture_service}"
            + (f" done_topic={self.done_topic} (timeout {self.execute_timeout_s:.0f}s)" if self.done_topic else ""))

    # ---- 타이머 유틸 -------------------------------------------------------
    def _arm(self, attr: str, period_s: float, callback):
        """attr('exec_timer' | 'watchdog') 에 one-shot 타이머를 (재)장전. 기존 것은 cancel + destroy."""
        self._disarm(attr)
        setattr(self, attr, self.create_timer(period_s, callback, clock=self.steady))

    def _disarm(self, attr: str):
        t = getattr(self, attr)
        if t is not None:
            t.cancel()
            self.destroy_timer(t)
            setattr(self, attr, None)

    # ---- 구독 콜백 --------------------------------------------------------
    def on_poses(self, msg: PoseArray):
        if msg.header.frame_id:
            self.frame_id = msg.header.frame_id
        poses = [((p.position.x, p.position.y, p.position.z),
                  (p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)) for p in msg.poses]
        key = (int(msg.header.stamp.sec), int(msg.header.stamp.nanosec))
        self.apply_action(self.ex.on_poses(poses, key=key))

    def on_status(self, msg: String):
        try:
            st = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn(f"status is not JSON: {msg.data[:80]}")
            return
        key = tuple(int(v) for v in st["stamp"]) if isinstance(st.get("stamp"), list) and len(st["stamp"]) == 2 else None
        self.apply_action(self.ex.on_status(st, key=key))

    # ---- 판단 결과 처리 --------------------------------------------------
    # 이름 주의: rclpy.Node 에는 `handle` 속성(C 노드 핸들)이 있어서 메서드 이름을 handle 로 두면
    # Node.__init__ 의 `with self.handle:` 이 우리 메서드를 잡아 AttributeError: __enter__ 로 죽는다 (실제로 겪음).
    def apply_action(self, action):
        if action is None:
            return
        stamp = self.get_clock().now().to_msg()
        if action.kind in ("send", "skip_duplicate", "skip_status", "empty", "done", "retry_later"):
            self._disarm("watchdog")                    # 이 프레임에 대한 판단이 났다
        if action.kind == "send":
            traj = action.trajectory
            self.last_cmd_stamp = [int(stamp.sec), int(stamp.nanosec)]   # 결과 보고의 cmd_stamp 와 대조한다
            self.pub_targets.publish(trajectory_to_posearray(traj, stamp, self.frame_id))
            self.pub_traj.publish(trajectory_markers(traj, stamp, self.frame_id))
            pk = traj.pick
            self.get_logger().info(
                f"{action.reason}: pick=({pk[0]:.3f},{pk[1]:.3f},{pk[2]:.3f}) m pre_z={traj.pre_pick[2]:.3f} "
                f"lift_z={traj.lift[2]:.3f} -> /robot/target_poses (3 poses)")
            # 로봇 완료 보고를 기다리는 모드면 타이머는 '보고가 안 올 때' 의 안전장치(timeout)다
            self._arm("exec_timer", self.execute_timeout_s if self.done_topic else self.execute_time_s, self.on_execute_done)
        elif action.kind == "busy":
            self.get_logger().debug(action.reason)
        elif action.kind in ("skip_duplicate", "skip_status", "empty"):
            if action.kind == "skip_status":        # rclpy 로거는 호출 지점마다 심각도 고정 -> 분기해서 부른다
                self.get_logger().warn(f"no command: {action.reason}")
            else:
                self.get_logger().info(f"no command: {action.reason}")
            if self.trigger_capture and self.exec_timer is None and self.ex.state != "DONE":
                self._arm("exec_timer", self.execute_time_s, self.on_execute_done)   # 잠시 뒤 재촬영 (백오프)
        elif action.kind == "done":
            self._disarm("exec_timer")
            self.pub_traj.publish(trajectory_markers(None, stamp, self.frame_id))
            self.get_logger().info(f"DONE: {action.reason} — {self.ex.counters}")
        elif action.kind == "recapture":
            self.request_capture()
        elif action.kind == "retry_later":
            self.get_logger().warn(f"{action.reason} — retrying in {self.execute_time_s:.0f} s")
            self._arm("exec_timer", self.execute_time_s, self.on_execute_done)
        elif action.kind == "noop":
            return
        self.pub_state.publish(String(data=json.dumps(dict(self.ex.snapshot(), last_action=action.kind), ensure_ascii=False)))

    # ---- 타이머 / 서비스 클라이언트 ------------------------------------------
    def on_startup(self):
        self.start_timer.cancel()
        self.destroy_timer(self.start_timer)
        self.start_timer = None
        self.get_logger().info("requesting first capture")
        self.request_capture()

    def on_robot_done(self, msg: String):
        """로봇(트윈 브리지)의 실행 결과 보고. EXECUTING 중일 때만 뜻이 있다 (그 외는 늦게 온 보고 -> 무시)."""
        try:
            res = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn(f"robot result is not JSON: {msg.data[:80]}")
            return
        if self.ex.state != "EXECUTING":
            self.get_logger().debug(f"robot result while {self.ex.state}: ignored")
            return
        cs = res.get("cmd_stamp")
        if cs is not None and self.last_cmd_stamp is not None and [int(v) for v in cs] != self.last_cmd_stamp:
            self.get_logger().warn(f"robot result for another command (stamp {cs} != {self.last_cmd_stamp}) — ignored")
            return
        self._disarm("exec_timer")
        line = (f"robot: {res.get('result')} (ok={res.get('ok')}) sim {res.get('cycle_s', '?')} s, "
                f"remaining {res.get('remaining', '?')}, placed {res.get('placed', '?')}")
        if res.get("ok"):                       # rclpy 로거는 호출 지점마다 심각도 고정 -> 분기해서 부른다
            self.get_logger().info(line)
        else:
            self.get_logger().warn(line)
        if self.trigger_capture:
            self.apply_action(self.ex.on_robot_result(res))
        else:
            self.ex.on_robot_result(res)

    def on_execute_done(self):
        self._disarm("exec_timer")
        if self.done_topic and self.ex.state == "EXECUTING":
            self.get_logger().error(f"robot did not report completion within {self.execute_timeout_s:.0f} s — requesting a new frame")
        if self.trigger_capture:
            self.apply_action(self.ex.on_execute_done())
        elif self.ex.state == "EXECUTING":
            self.ex.state = "IDLE"

    def on_watchdog(self):
        self._disarm("watchdog")
        if self.ex.state == "DONE":
            return
        self.get_logger().error(f"no decision within {self.watchdog_s:.0f} s of the capture request — requesting again")
        self.request_capture()

    def request_capture(self):
        if self.cli is None or self.ex.state == "DONE":
            return
        if not self.cli.service_is_ready():
            self.get_logger().warn("capture service not available yet, retrying in 1 s")
            self._arm("exec_timer", 1.0, self.on_execute_done)
            return
        fut = self.cli.call_async(Trigger.Request())
        fut.add_done_callback(self.on_capture_response)
        self._arm("watchdog", self.watchdog_s, self.on_watchdog)

    def on_capture_response(self, fut):
        try:
            resp = fut.result()
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"capture call failed: {e}")
            return
        if not resp.success:
            # 상태가 OK 가 아닌 프레임도 success=False 로 온다 — 그 프레임은 토픽으로 이미 나갔으므로 logic 이 noop 을 돌려준다.
            self.apply_action(self.ex.on_capture_failed(resp.message))
        # success 면 인식 노드가 곧 pick_poses + status 를 낸다 -> on_poses/on_status 가 이어 받는다


def main(args=None):
    rclpy.init(args=args)
    node = PickExecutorNode()
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
