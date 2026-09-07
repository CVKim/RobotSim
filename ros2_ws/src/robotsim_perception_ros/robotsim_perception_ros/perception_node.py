# -*- coding: utf-8 -*-
"""robotsim_perception ROS2 노드.

    ToF 프레임(합성 | 실측 .mim 재생)  ->  runtime.decide()  ->  토픽/TF/서비스

  퍼블리시
    /tof/points               sensor_msgs/PointCloud2     tof_optical (센서 QoS: best-effort, depth 1)
    /perception/boxes         visualization_msgs/MarkerArray   base_link
    /perception/pick_poses    geometry_msgs/PoseArray     base_link, 픽 순서대로
    /perception/next_pick     geometry_msgs/PoseStamped   base_link
    /perception/status        std_msgs/String             한 줄 JSON (Decision + 소스 이름)
    /diagnostics              diagnostic_msgs/DiagnosticArray   판정 상태 + 카메라 드리프트
    TF static                 base_link -> tof_optical    (T_base_cam)
  서비스
    ~/capture                 std_srvs/Trigger            프레임 1장 즉시 처리 (rate_hz=0 이면 트리거 모드)

  파라미터 (ros2 param list 로 확인)
    source            'synthetic' | 세션 폴더 경로 (실측, 로컬 전용)
    rate_hz           주기 처리 Hz. 0 이면 서비스 트리거만
    lattice           v2 격자 보완 사용 (True)
    min_confidence    픽 후보 신뢰도 하한 (0.55)
    min_valid_frac    프레임 유효 픽셀 하한 (0.25) — 미만이면 RETAKE
    source_roi_mm     소스 팔레트 반경 (620) — 목적지 스택 제외
    cam_height_mm     탑다운 설치 기본 외참 높이 (4183)
    extrinsics_json   T_base_cam JSON 경로 (있으면 cam_height_mm 무시)
    dest_x_mm/dest_y_mm  목적지 스택 중심 (카메라 좌표)
    sku               [L, W, H] mm
    cloud_stride      포인트클라우드 다운샘플 (2)

  실제 셀에서는 source 자리에 센서 드라이버가 오고, /perception/pick_poses 를 로봇 제어 노드가 구독한다.
"""
from __future__ import annotations

import json
import time

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import PoseArray, PoseStamped
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import StaticTransformBroadcaster
from visualization_msgs.msg import MarkerArray

from robotsim_perception.pose import load_transform, topdown_camera_transform
from robotsim_perception.runtime import HealthMonitor, Thresholds, decide

from . import msgs
from .sources import make_source


class SourceError(RuntimeError):
    """프레임 소스(센서·트윈 서버)가 프레임을 못 줬다. 노드는 죽지 않고 SOURCE_ERROR 를 보고한다."""


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("robotsim_perception")
        P = self.declare_parameter
        self.source_spec = P("source", "synthetic").value
        self.rate_hz = float(P("rate_hz", 1.0).value)
        loop = bool(P("loop", True).value)
        seed = int(P("seed", 0).value)
        self.th = Thresholds(
            lattice=bool(P("lattice", True).value),
            min_confidence=float(P("min_confidence", 0.55).value),
            min_valid_frac=float(P("min_valid_frac", 0.25).value),
            source_roi_mm=float(P("source_roi_mm", 620.0).value) or None,
        )
        self.stride = int(P("cloud_stride", 2).value)
        self.publish_cloud = bool(P("publish_cloud", True).value)
        cam_h = float(P("cam_height_mm", 4183.0).value)
        ext = str(P("extrinsics_json", "").value)
        self.dest = (float(P("dest_x_mm", -1191.4).value), float(P("dest_y_mm", -38.3).value))
        self.sku = tuple(float(v) for v in P("sku", [293.0, 219.0, 283.0]).value)
        self.base_frame = str(P("base_frame", "base_link").value)
        self.cam_frame = str(P("camera_frame", "tof_optical").value)

        pick_every = int(P("synthetic_pick_every", 1).value)   # 합성 소스: N 프레임마다 박스 1개 제거
        # True 면 임의 박스 대신 '계획 1번 픽' 박스를 다음 프레임에서 제거 — 인식→제어→재촬영 사이클 데모용 (pick_executor)
        self.remove_picked = bool(P("synthetic_remove_picked", False).value)
        self.T = load_transform(ext) if ext else topdown_camera_transform(cam_h)
        self.source = make_source(self.source_spec, seed=seed, loop=loop, pick_every=pick_every)
        if self.remove_picked and hasattr(self.source, "auto_pop"):
            self.source.auto_pop = False
        self.health = HealthMonitor()

        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub_cloud = self.create_publisher(PointCloud2, "/tof/points", sensor_qos)
        self.pub_markers = self.create_publisher(MarkerArray, "/perception/boxes", 10)
        self.pub_poses = self.create_publisher(PoseArray, "/perception/pick_poses", 10)
        self.pub_next = self.create_publisher(PoseStamped, "/perception/next_pick", 10)
        self.pub_status = self.create_publisher(String, "/perception/status", 10)
        self.pub_diag = self.create_publisher(DiagnosticArray, "/diagnostics", 10)
        self.srv = self.create_service(Trigger, "~/capture", self.on_capture)

        self.tf_static = StaticTransformBroadcaster(self)
        self.tf_static.sendTransform(
            msgs.static_transform(self.T, self.get_clock().now().to_msg(), self.base_frame, self.cam_frame))

        self.timer = None
        if self.rate_hz > 0:
            # 단조 시계(STEADY_TIME) 사용. 기본 ROS 시계는 시스템 시간이라 벽시계가 뒤로 점프하면
            # (WSL2 에서 실측: 프레임 간 −66 s 점프, NTP 보정 시에도 발생 가능) 타이머가 멈춘다.
            self.timer = self.create_timer(1.0 / self.rate_hz, self._tick,
                                           clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.n = 0
        self.get_logger().info(
            f"source={self.source_spec} rate={self.rate_hz}Hz lattice={self.th.lattice} "
            f"min_conf={self.th.min_confidence} frames {self.base_frame}->{self.cam_frame}")

    # ------------------------------------------------------------ core cycle
    def _tick(self):
        try:
            self.process()
        except SourceError as e:
            self.get_logger().error(f"frame source failed, skipping this tick: {e}")

    def process(self):
        try:
            item = self.source.next()
        except (OSError, RuntimeError, ValueError) as e:
            # 소켓 끊김·서버 오류·손상 프레임. rclpy 는 콜백 예외를 spin 밖으로 던져 노드를 죽이므로 여기서 잡는다 (리뷰 지적)
            raise SourceError(repr(e)) from e
        if item is None:
            self.get_logger().info("source exhausted")
            if self.timer is not None:
                self.timer.cancel()
            return None
        frame, name = item
        t0 = time.perf_counter()
        dec = decide(frame, self.th, sku=self.sku, dest_xy_mm=self.dest)
        if self.health.ref_D is None:
            self.health.set_reference(frame)
        health = self.health.check(frame)
        stamp = self.get_clock().now().to_msg()

        plan_pps = msgs.pick_poses(dec.boxes, dec.plan, self.T)
        if self.remove_picked and dec.plan and hasattr(self.source, "remove_nearest"):
            first = next((b for b in dec.boxes if b.id == dec.plan[0].box_id), None)
            if first is not None:
                self.source.remove_nearest(float(first.center_mm[0]), float(first.center_mm[1]))
        if self.publish_cloud:
            self.pub_cloud.publish(msgs.frame_to_pointcloud2(frame, stamp, self.cam_frame, self.stride))
        self.pub_markers.publish(msgs.boxes_to_markers(dec.boxes, self.T, plan_pps, stamp, self.base_frame))
        self.pub_poses.publish(msgs.poses_to_posearray(plan_pps, stamp, self.base_frame))
        if plan_pps:
            self.pub_next.publish(msgs.pose_stamped(plan_pps[0], stamp, self.base_frame))
        # stamp 를 status JSON 에도 넣는다: 구독자(pick_executor)가 pick_poses 헤더 스탬프와 맞춰 같은 프레임인지 확인한다
        self.pub_status.publish(msgs.decision_to_string(
            dec, {"source": name, "health": health.get("status"), "seq": self.n,
                  "stamp": [int(stamp.sec), int(stamp.nanosec)]}))
        self.pub_diag.publish(msgs.diagnostics(dec, health, stamp))
        self.n += 1
        self.get_logger().info(
            f"[{self.n}] {name}: {dec.status} boxes={dec.n_boxes} pickable={dec.n_pickable} "
            f"valid={dec.valid_frac:.0%} health={health.get('status')} "
            f"{(time.perf_counter() - t0) * 1e3:.0f} ms")
        return dec

    def on_capture(self, request, response):
        try:
            dec = self.process()
        except SourceError as e:
            self.get_logger().error(f"capture failed: {e}")
            response.success = False
            response.message = json.dumps({"status": "SOURCE_ERROR", "reason": str(e)[:200]}, ensure_ascii=False)
            return response
        if dec is None:
            response.success, response.message = False, "source exhausted"
        else:
            response.success = dec.status == "OK"
            response.message = dec.log_line()
        return response


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass                                   # Ctrl-C / SIGTERM(launch, timeout) 은 정상 종료
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
