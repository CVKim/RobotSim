# -*- coding: utf-8 -*-
"""대차(카트) 견인 고리 노드 — 비스듬한 ToF 카메라 -> 대차 고정 좌표계 -> 고리 포즈.

    ToF 프레임(합성 | 실측 .mim 재생)  ->  cart.analyze_cart()  ->  토픽/TF/서비스

  퍼블리시
    /tof/points        sensor_msgs/PointCloud2       tof_optical (best-effort)
    /cart/hook_pose    geometry_msgs/PoseStamped     tof_optical — 위치 = 고리 wall_top, 자세 = 대차 축 (도킹 목표)
    /cart/markers      visualization_msgs/MarkerArray  cart 프레임 — 데크 판 · 림 · 레일 · 고리
    /cart/status       std_msgs/String               CartResult 한 줄 JSON
    /diagnostics       diagnostic_msgs/DiagnosticArray  상태 + 피팅 품질(림·레일 MAD, 카메라 높이)
    TF (동적)          tof_optical -> cart           프레임마다 재추정 (R_cart, t_cart)
  서비스
    ~/capture          std_srvs/Trigger              프레임 1장 즉시 처리

  파라미터
    source        'synthetic' | 세션 폴더(상위 폴더면 순서대로 재생, 로컬 전용)
    rate_hz       주기 (0 이면 트리거만)   loop   seed   cloud_stride   publish_cloud
    camera_frame  'tof_optical'   cart_frame 'cart'
    extrinsics_json  있으면 base_link -> tof_optical 정적 TF 도 낸다 (실측값은 핸드아이 캘리브레이션 필요)

  빈피킹 노드와의 차이: 카메라가 대차를 약 40° 비스듬히 보므로 '로봇 베이스 기준 탑다운' 가정을 쓸 수 없고,
  좌표계를 데크 구조물(플레이트 평면·림·레일)에서 매 프레임 추정한다. 그래서 TF 가 정적이 아니라 동적이다.
"""
from __future__ import annotations

import time

import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import PoseStamped
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from visualization_msgs.msg import MarkerArray

from robotsim_perception.cart import analyze_cart
from robotsim_perception.pose import load_transform

from . import msgs
from .sources import make_cart_source


class CartNode(Node):
    def __init__(self):
        super().__init__("robotsim_cart")
        P = self.declare_parameter
        self.source_spec = P("source", "synthetic").value
        self.rate_hz = float(P("rate_hz", 1.0).value)
        loop = bool(P("loop", True).value)
        seed = int(P("seed", 0).value)
        self.stride = int(P("cloud_stride", 2).value)
        self.publish_cloud = bool(P("publish_cloud", True).value)
        self.cam_frame = str(P("camera_frame", "tof_optical").value)
        self.cart_frame = str(P("cart_frame", "cart").value)
        self.base_frame = str(P("base_frame", "base_link").value)
        ext = str(P("extrinsics_json", "").value)

        self.source = make_cart_source(self.source_spec, seed=seed, loop=loop)

        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub_cloud = self.create_publisher(PointCloud2, "/tof/points", sensor_qos)
        self.pub_hook = self.create_publisher(PoseStamped, "/cart/hook_pose", 10)
        self.pub_markers = self.create_publisher(MarkerArray, "/cart/markers", 10)
        self.pub_status = self.create_publisher(String, "/cart/status", 10)
        self.pub_diag = self.create_publisher(DiagnosticArray, "/diagnostics", 10)
        self.srv = self.create_service(Trigger, "~/capture", self.on_capture)
        self.tf = TransformBroadcaster(self)
        if ext:
            self.tf_static = StaticTransformBroadcaster(self)
            self.tf_static.sendTransform(msgs.static_transform(
                load_transform(ext), self.get_clock().now().to_msg(), self.base_frame, self.cam_frame))

        self.timer = None
        if self.rate_hz > 0:
            self.timer = self.create_timer(1.0 / self.rate_hz, self.process,
                                           clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.n = 0
        self.get_logger().info(f"source={self.source_spec} rate={self.rate_hz}Hz frames {self.cam_frame}->{self.cart_frame}")

    def process(self):
        item = self.source.next()
        if item is None:
            self.get_logger().info("source exhausted")
            if self.timer is not None:
                self.timer.cancel()
            return None
        frame, name = item
        t0 = time.perf_counter()
        res = analyze_cart(frame)
        stamp = self.get_clock().now().to_msg()
        if self.publish_cloud:
            self.pub_cloud.publish(msgs.frame_to_pointcloud2(frame, stamp, self.cam_frame, self.stride))
        if res.status == "OK":
            self.tf.sendTransform(msgs.cart_transform(res, stamp, self.cam_frame, self.cart_frame))
            self.pub_hook.publish(msgs.hook_pose_stamped(res, stamp, self.cam_frame))
        self.pub_markers.publish(msgs.cart_markers(res, stamp, self.cart_frame))
        self.pub_status.publish(msgs.cart_status_string(res, {"source": name, "seq": self.n}))
        self.pub_diag.publish(msgs.cart_diagnostics(res, stamp))
        self.n += 1
        hc = res.hook_cart_mm
        self.get_logger().info(
            f"[{self.n}] {name}: {res.status} "
            + (f"hook_cart=({hc[0]:.1f},{hc[1]:.1f},{hc[2]:.1f}) mm camH={res.plane['camera_height_mm']:.0f} "
               f"yaw={res.cart.get('rim_yaw_deg', 0):+.2f} " if hc else f"({res.reason}) ")
            + f"valid={res.valid_frac:.0%} {(time.perf_counter() - t0) * 1e3:.0f} ms")
        return res

    def on_capture(self, request, response):
        res = self.process()
        if res is None:
            response.success, response.message = False, "source exhausted"
        else:
            response.success = res.status == "OK"
            response.message = res.log_line()
        return response


def main(args=None):
    rclpy.init(args=args)
    node = CartNode()
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
