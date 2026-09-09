# -*- coding: utf-8 -*-
"""twin_bridge — MuJoCo 셀 트윈(Windows 의 tools/twin_server.py)을 ROS 그래프에서 '로봇 드라이버' 자리에 놓는 노드.

    pick_executor ──/robot/target_poses (PoseArray 3점)──▶ twin_bridge ──TCP──▶ 트윈 서버 (UR10e 가 IK·보간으로 실행)
    pick_executor ◀──/robot/execution_result (String JSON)── twin_bridge ◀──────  결과 {"ok","result","cycle_s",...}

  같은 일을 **액션**으로도 받는다: `/robot/execute_pick` (robotsim_interfaces/action/ExecutePick).
  픽 한 번은 몇 초에서 수십 초가 걸리는 장시간 작업이라 원래 액션이 맞는 그릇이다 — 목표 하나에 결과 하나가 대응하고,
  진행 중에 피드백(재생 진행률·현재 TCP)이 오며, 실행 중 새 목표는 서버가 거절한다(토픽 경로에서는 busy 를 따로 보고해야 했다).
  취소는 받지 않는다(REJECT): 트윈의 한 실행은 TCP 요청 하나로 끝까지 돌아 중간에 끊을 수단이 없다 — 실제 드라이버라면
  여기서 정지 명령을 보낸다.
                                                              ├─▶ /robot/tcp_path (MarkerArray: 이번 실행의 TCP 경로 + 현재 위치)
                                                              ├─▶ /twin/state    (String JSON: 남은 박스·놓은 박스·TCP)
                                                              ├─▶ /joint_states  (sensor_msgs/JointState: 트윈의 관절각)
                                                                    └─ robot_state_publisher + URDF 로 rviz 에 팔이 그려진다
                                                              ├─▶ TF base_link → tcp (동적: 실행 뒤 TCP 위치)
                                                              └── ~/reset (std_srvs/Trigger): 장면 초기화

  실행은 수십 초 걸리므로(시뮬 8 s 를 실시간보다 느리게 계산) 작업 스레드에서 트윈을 호출하고, 그동안 노드는 계속 spin 한다.
  실행 중 새 명령이 오면 거부한다(pick_executor 도 busy 로 안 보내지만 방어). 트윈 소켓은 락으로 한 번에 하나만 쓴다.

  트윈은 시뮬 8 s 를 1 s 안에 계산한다. realtime_factor > 0 이면 돌아온 TCP 경로를 시뮬 시각대로 실시간 재생하면서
  (TF base_link→tcp, 마커) 재생이 끝난 뒤에 완료를 보고한다 — rviz 에서 팔 끝이 움직이는 것을 보고, 실제 로봇처럼 사이클 시간이
  흐르게 하기 위해서다. 0 이면 계산이 끝나는 즉시 보고한다(최대 속도).

  파라미터: host ('' = WSL2 기본 게이트웨이 = Windows 호스트), port 5555, frame_id base_link, tcp_frame tcp,
            timeout_s 600 (한 실행의 소켓 타임아웃), state_period_s 1.0, realtime_factor 1.0,
            action_name '/robot/execute_pick' (빈 문자열이면 액션 서버를 열지 않는다), publish_joint_states true
"""
from __future__ import annotations

import json
import threading
import time

import rclpy
from geometry_msgs.msg import Point, PoseArray, Quaternion, TransformStamped, Vector3
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.clock import Clock, ClockType
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from robotsim_interfaces.action import ExecutePick
from sensor_msgs.msg import JointState
from std_msgs.msg import ColorRGBA, Header, String
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from robotsim_perception.twin_link import DEFAULT_PORT, TwinClient

from .logic import execute_request, path_points, result_message


def _header(stamp, frame_id: str) -> Header:
    h = Header()
    h.stamp = stamp
    h.frame_id = frame_id
    return h


def path_markers(points, tcp_now, stamp, frame_id: str) -> MarkerArray:
    ma = MarkerArray()
    # 같은 ns/id 로 ADD 하면 rviz 가 그 자리에서 교체한다. 존재하지 않는 마커에 DELETE 를 보내면 rviz2 표시가 'Status: Error' 가 된다(캡처에서 확인)
    if len(points) >= 2:
        line = Marker()
        line.header = _header(stamp, frame_id)
        line.ns, line.id, line.type, line.action = "tcp_path", 0, Marker.LINE_STRIP, Marker.ADD
        line.pose.orientation = Quaternion(w=1.0)
        line.scale = Vector3(x=0.008, y=0.0, z=0.0)
        line.color = ColorRGBA(r=0.3, g=0.9, b=1.0, a=0.9)
        line.points = [Point(x=p[0], y=p[1], z=p[2]) for p in points]
        ma.markers.append(line)
    if tcp_now is not None:
        s = Marker()
        s.header = _header(stamp, frame_id)
        s.ns, s.id, s.type, s.action = "tcp_now", 0, Marker.SPHERE, Marker.ADD
        s.pose.position = Point(x=float(tcp_now[0]), y=float(tcp_now[1]), z=float(tcp_now[2]))
        s.pose.orientation = Quaternion(w=1.0)
        s.scale = Vector3(x=0.06, y=0.06, z=0.06)
        s.color = ColorRGBA(r=1.0, g=0.5, b=0.1, a=1.0)
        ma.markers.append(s)
    return ma


class TwinBridgeNode(Node):
    def __init__(self):
        super().__init__("twin_bridge")
        P = self.declare_parameter
        host = str(P("host", "").value)
        port = int(P("port", DEFAULT_PORT).value)
        self.frame_id = str(P("frame_id", "base_link").value)
        self.tcp_frame = str(P("tcp_frame", "tcp").value)
        timeout_s = float(P("timeout_s", 600.0).value)
        state_period = float(P("state_period_s", 1.0).value)
        self.realtime_factor = float(P("realtime_factor", 1.0).value)

        self.client = TwinClient(host or None, port, timeout_s=timeout_s)
        self.lock = threading.Lock()          # 소켓은 한 스레드씩
        self.busy = False
        self.n_cmd = 0
        self.last_tcp = None

        self.sub = self.create_subscription(PoseArray, "/robot/target_poses", self.on_targets, 10)
        self.pub_result = self.create_publisher(String, "/robot/execution_result", 10)
        self.pub_path = self.create_publisher(MarkerArray, "/robot/tcp_path", 10)
        self.pub_state = self.create_publisher(String, "/twin/state", 10)
        self.publish_joints = bool(P("publish_joint_states", True).value)
        self.pub_joints = self.create_publisher(JointState, "/joint_states", 10) if self.publish_joints else None
        self.joint_names = []
        self.tf = TransformBroadcaster(self)
        self.srv_reset = self.create_service(Trigger, "~/reset", self.on_reset)
        steady = Clock(clock_type=ClockType.STEADY_TIME)
        self.state_timer = self.create_timer(state_period, self.on_state_timer, clock=steady)

        # 액션 서버: 실행이 오래 걸리는 콜백이라 다른 콜백(상태 타이머·reset)이 막히지 않도록
        # ReentrantCallbackGroup + MultiThreadedExecutor 로 돈다 (main 참고).
        action_name = str(P("action_name", "/robot/execute_pick").value)
        self.action_srv = ActionServer(
            self, ExecutePick, action_name,
            execute_callback=self.on_execute_goal,
            goal_callback=self.on_goal_request,
            cancel_callback=lambda gh: CancelResponse.REJECT,
            callback_group=ReentrantCallbackGroup()) if action_name else None

        try:
            info = self._call(self.client.hello)
            self.joint_names = list(info.get("joint_names") or [])
            self.get_logger().info(f"twin at {self.client.host}:{self.client.port}: arm={info.get('arm')} "
                                   f"boxes={info.get('remaining')} seed={info.get('seed')} gt_selfcheck={info.get('gt_selfcheck_mm')} mm")
        except OSError as e:
            self.get_logger().error(f"cannot reach twin server at {self.client.host}:{self.client.port}: {e} "
                                    f"— start tools/twin_server.py on Windows; retrying on demand")

    def _call(self, fn, *a, **kw):
        with self.lock:
            return fn(*a, **kw)

    # ---- 명령 ------------------------------------------------------------------
    def _nack(self, result: str, stamp, reason: str):
        """받은 명령을 실행하지 않을 때도 결과를 낸다 — 안 내면 pick_executor 가 execute_timeout_s 까지 기다린다 (리뷰 지적)."""
        out = result_message({"ok": False, "result": result, "error": reason}, self.n_cmd, stamp)
        self.pub_result.publish(String(data=json.dumps(out, ensure_ascii=False)))

    def on_targets(self, msg: PoseArray):
        stamp = (int(msg.header.stamp.sec), int(msg.header.stamp.nanosec))
        if self.busy:
            self.get_logger().warn("execute request while the twin is still executing — rejected (busy)")
            self._nack("busy", stamp, "twin still executing the previous command")
            return
        poses = [((p.position.x, p.position.y, p.position.z),
                  (p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)) for p in msg.poses]
        try:
            req = execute_request(poses, msg.header.frame_id or self.frame_id)
        except ValueError as e:
            self.get_logger().error(f"bad target_poses: {e}")
            self._nack("bad_request", stamp, str(e))
            return
        self.busy = True
        self.n_cmd += 1
        pk = req["pick"]
        self.get_logger().info(f"cmd #{self.n_cmd}: pick=({pk[0]:.3f},{pk[1]:.3f},{pk[2]:.3f}) m -> twin execute")
        threading.Thread(target=self._run, args=(req, self.n_cmd, stamp), daemon=True).start()

    def _run(self, req: dict, seq: int, stamp):
        self._execute(req, seq, stamp)

    def _execute(self, req: dict, seq: int, stamp, feedback=None) -> dict:
        """트윈에 한 번 실행시키고 결과를 /robot/execution_result 로 보고한 뒤 그 dict 를 돌려준다.
        토픽 경로(_run)와 액션 경로(on_execute_goal)가 같은 몸통을 쓴다."""
        try:
            res = self._call(self.client.execute, req["pre_pick"], req["pick"], req["lift"], req["quat"], req["frame_id"])
        except (OSError, RuntimeError) as e:
            res = {"ok": False, "result": "bridge_error", "error": repr(e)}
            self.get_logger().error(f"twin execute failed: {e!r}")
        now = self.get_clock().now().to_msg()
        out = result_message(res, seq, stamp)
        line = (f"cmd #{seq} -> {out['result']} sim {out.get('cycle_s', '?')} s wall {out.get('wall_s', '?')} s "
                f"pick_err {out.get('pick_err_mm', '-')} mm remaining {out.get('remaining', '?')} placed {out.get('placed', '?')}")
        # rclpy 로거는 호출 지점(파일:줄)마다 심각도가 고정된다 — 같은 줄에서 info/warn 을 바꿔 부르면
        # 'Logger severity cannot be changed between calls' 로 스레드가 죽는다 (첫 실행 cmd #4 에서 실제로 죽어 결과 보고가 유실됐다)
        if out["ok"]:
            self.get_logger().info(line)
        else:
            self.get_logger().warn(line)
        if res.get("joint_names"):
            self.joint_names = list(res["joint_names"])
        path = res.get("tcp_path") or []
        if self.realtime_factor > 0 and len(path) >= 2:
            self._replay(path, feedback, res.get("q_path") or [])   # 시뮬 시각대로 TCP·관절을 움직여 보여 준 뒤 보고
        else:
            pts = path_points(res)
            if pts:
                self.last_tcp = pts[-1]
            self.pub_path.publish(path_markers(pts, self.last_tcp, now, self.frame_id))
            self._broadcast_tcp(now)
        self.busy = False
        self.pub_result.publish(String(data=json.dumps(out, ensure_ascii=False)))
        return out

    def _publish_joints(self, q, stamp):
        """트윈의 관절각 -> /joint_states. robot_state_publisher 가 이것으로 URDF 팔의 TF 를 낸다."""
        if self.pub_joints is None or not q or not self.joint_names:
            return
        js = JointState()
        js.header = _header(stamp, "")
        js.name = list(self.joint_names[:len(q)])
        js.position = [float(v) for v in q[:len(js.name)]]
        self.pub_joints.publish(js)

    def _replay(self, path, feedback=None, q_path=None):
        """tcp_path [[t,x,y,z],...] 를 실시간(÷realtime_factor)으로 재생: 10 Hz 로 지나온 경로 + 현재 TCP 마커 + TF."""
        t0 = time.monotonic()
        pts, last_pub = [], -1.0
        n = len(path)
        for i, (t, x, y, z) in enumerate(path):
            wait = float(t) / self.realtime_factor - (time.monotonic() - t0)
            if wait > 0:
                time.sleep(wait)
            pts.append((float(x), float(y), float(z)))
            self.last_tcp = pts[-1]
            if time.monotonic() - last_pub >= 0.1:
                now = self.get_clock().now().to_msg()
                self.pub_path.publish(path_markers(pts, self.last_tcp, now, self.frame_id))
                self._broadcast_tcp(now)
                if q_path and i < len(q_path):        # 같은 스텝 간격으로 기록되므로 인덱스가 짝이다
                    self._publish_joints(q_path[i][1:], now)
                last_pub = time.monotonic()
                if feedback is not None:
                    feedback("replaying", (i + 1) / float(n), self.last_tcp)
        now = self.get_clock().now().to_msg()
        self.pub_path.publish(path_markers(pts, self.last_tcp, now, self.frame_id))
        self._broadcast_tcp(now)

    # ---- 액션 (/robot/execute_pick) ------------------------------------------------
    def on_goal_request(self, goal_request):
        """실행 중이면 새 목표를 거절한다 — 트윈은 한 번에 하나만 실행할 수 있다."""
        if self.busy:
            self.get_logger().warn("execute_pick goal while the twin is busy — rejected")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def on_execute_goal(self, goal_handle):
        g = goal_handle.request
        result = ExecutePick.Result()
        poses = [((p.pose.position.x, p.pose.position.y, p.pose.position.z),
                  (p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z, p.pose.orientation.w))
                 for p in (g.pre_pick, g.pick, g.lift)]
        try:
            req = execute_request(poses, g.pick.header.frame_id or self.frame_id)
        except ValueError as e:
            self.get_logger().error(f"bad execute_pick goal: {e}")
            result.ok, result.result = False, "bad_request"
            goal_handle.abort()
            return result
        self.busy = True
        self.n_cmd += 1
        seq = self.n_cmd
        stamp = (int(g.pick.header.stamp.sec), int(g.pick.header.stamp.nanosec))
        pk = req["pick"]
        self.get_logger().info(f"goal #{seq} ({g.label or 'pick'}): pick=({pk[0]:.3f},{pk[1]:.3f},{pk[2]:.3f}) m -> twin execute")

        def feedback(phase: str, progress: float, tcp):
            fb = ExecutePick.Feedback()
            fb.phase = phase
            fb.progress = float(progress)
            if tcp is not None:
                fb.tcp = Point(x=float(tcp[0]), y=float(tcp[1]), z=float(tcp[2]))
            goal_handle.publish_feedback(fb)

        feedback("executing", 0.0, self.last_tcp)
        out = self._execute(req, seq, stamp, feedback=feedback)
        result.ok = bool(out.get("ok", False))
        result.result = str(out.get("result", "unknown"))
        for key, field in (("cycle_s", "cycle_s"), ("pick_err_mm", "pick_err_mm"), ("place_err_mm", "place_err_mm")):
            v = out.get(key)
            if v is not None:
                setattr(result, field, float(v))
        result.remaining = int(out.get("remaining", 0) or 0)
        result.placed = int(out.get("placed", 0) or 0)
        if result.ok:
            goal_handle.succeed()
        else:
            goal_handle.abort()               # 실패한 픽은 abort — 클라이언트가 결과를 그대로 읽고 다음 후보로 간다
        return result

    # ---- 상태 / TF ---------------------------------------------------------------
    def on_state_timer(self):
        if self.busy:
            return                            # 실행 중에는 트윈이 응답할 수 없다 (직렬 처리) — 스킵
        try:
            st = self._call(self.client.state)
        except (OSError, RuntimeError):
            return
        if st.get("tcp"):
            self.last_tcp = tuple(st["tcp"])
        now = self.get_clock().now().to_msg()
        self._publish_joints(st.get("q") or [], now)          # 실행 중이 아닐 때도 팔이 rviz 에 서 있게
        self.pub_state.publish(String(data=json.dumps(dict(st, busy=self.busy), ensure_ascii=False)))
        self._broadcast_tcp(now)

    def _broadcast_tcp(self, stamp):
        if self.last_tcp is None:
            return
        ts = TransformStamped()
        ts.header = _header(stamp, self.frame_id)
        ts.child_frame_id = self.tcp_frame
        ts.transform.translation = Vector3(x=float(self.last_tcp[0]), y=float(self.last_tcp[1]), z=float(self.last_tcp[2]))
        ts.transform.rotation = Quaternion(x=1.0, y=0.0, z=0.0, w=0.0)     # 툴 +Z 아래 (탑다운)
        self.tf.sendTransform(ts)

    def on_reset(self, request, response):
        if self.busy:
            response.success, response.message = False, "twin is executing"
            return response
        try:
            info = self._call(self.client.reset)
            response.success, response.message = True, json.dumps(info)
        except (OSError, RuntimeError) as e:
            response.success, response.message = False, repr(e)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = TwinBridgeNode()
    # 액션 실행 콜백이 수 초~수십 초 블로킹하므로 단일 스레드 spin 이면 상태 타이머·reset 이 그동안 멈춘다.
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.client.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
