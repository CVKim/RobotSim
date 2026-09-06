# -*- coding: utf-8 -*-
"""robotsim_perception 결과 -> ROS2 메시지 변환 (순수 함수, 노드·그래프 없이 테스트 가능).

좌표계 약속
  tof_optical : 카메라 광학 프레임. x 오른쪽, y 아래, z 전방(깊이). 실측 .mim 의 X/Y/D 와 같다.
  base_link   : 로봇 베이스. T_base_cam(4x4, mm) 으로 변환. 기본값은 탑다운 설치 예시
                (robotsim_perception.pose.topdown_camera_transform) — 실제 값은 핸드아이 캘리브레이션 필요.

메시지
  PointCloud2  /tof/points            tof_optical, XYZI (m, 강도)
  MarkerArray  /perception/boxes      base_link, CUBE(상면) + TEXT(id·신뢰도) + ARROW(다음 픽 접근)
  PoseArray    /perception/pick_poses base_link, 픽 순서대로. 툴 +Z 가 접근 방향(아래), yaw = 상면 장축
  PoseStamped  /perception/next_pick  위 첫 번째
  String       /perception/status     runtime.Decision 한 줄 JSON
  DiagnosticArray /diagnostics        판정 상태 + 드리프트 감시
"""
from __future__ import annotations

import json
import math
from typing import Sequence

import numpy as np
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import Point, Pose, PoseArray, PoseStamped, Quaternion, TransformStamped, Vector3
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import ColorRGBA, Header, String
from visualization_msgs.msg import Marker, MarkerArray

from robotsim_perception.pose import PickPose, box_to_pick_pose

MM = 1e-3


# ------------------------------------------------------------------ math helpers

def rot_z(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def quat_from_matrix(R: np.ndarray) -> Quaternion:
    """3x3 회전행렬 -> geometry_msgs Quaternion (x,y,z,w)."""
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    return Quaternion(x=x / n, y=y / n, z=z / n, w=w / n)


def tool_orientation(approach: Sequence[float], yaw_deg: float) -> Quaternion:
    """툴 프레임: +Z = 접근 벡터(박스 쪽), X = 상면 장축(yaw). 탑다운이면 Z 가 -base_Z."""
    a = np.asarray(approach, float)
    a = a / (np.linalg.norm(a) or 1.0)
    x0 = np.array([math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg)), 0.0])
    x = x0 - a * float(np.dot(x0, a))              # 접근축에 직교화
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0])
    x = x / np.linalg.norm(x)
    y = np.cross(a, x)
    R = np.stack([x, y, a], axis=1)                # 열 = 툴 x, y, z 축 (base 좌표)
    return quat_from_matrix(R)


def header(stamp, frame_id: str) -> Header:
    h = Header()
    h.stamp = stamp
    h.frame_id = frame_id
    return h


# ------------------------------------------------------------------ conversions

def frame_to_pointcloud2(frame, stamp, frame_id: str, stride: int = 2) -> PointCloud2:
    """Frame(X/Y/D mm, valid) -> PointCloud2 XYZI in tof_optical (m). stride 로 다운샘플."""
    s = max(int(stride), 1)
    X, Y, D, I, V = (frame.X[::s, ::s], frame.Y[::s, ::s], frame.D[::s, ::s],
                     frame.I[::s, ::s], frame.valid[::s, ::s])
    pts = np.stack([X[V] * MM, Y[V] * MM, D[V] * MM, I[V]], axis=1).astype(np.float32)
    fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
              for i, n in enumerate(("x", "y", "z", "intensity"))]
    return pc2.create_cloud(header(stamp, frame_id), fields, pts.tolist())


def static_transform(T_base_cam: np.ndarray, stamp, base_frame: str, cam_frame: str) -> TransformStamped:
    """T_base_cam (mm) -> TF base_frame -> cam_frame (m)."""
    T = np.asarray(T_base_cam, float)
    ts = TransformStamped()
    ts.header = header(stamp, base_frame)
    ts.child_frame_id = cam_frame
    ts.transform.translation = Vector3(x=float(T[0, 3]) * MM, y=float(T[1, 3]) * MM, z=float(T[2, 3]) * MM)
    ts.transform.rotation = quat_from_matrix(T[:3, :3])
    return ts


def pick_poses(boxes, plan, T_base_cam: np.ndarray, clearance_mm: float = 150.0) -> list:
    """Box + PickStep 순서 -> PickPose 목록 (base 좌표 mm, 픽 순서대로)."""
    by_id = {b.id: b for b in boxes}
    out = []
    for step in plan:
        b = by_id.get(step.box_id)
        if b is None:
            continue
        out.append(box_to_pick_pose(b, T_base_cam, clearance_mm=clearance_mm))
    return out


def poses_to_posearray(pps: Sequence[PickPose], stamp, frame_id: str) -> PoseArray:
    pa = PoseArray()
    pa.header = header(stamp, frame_id)
    for p in pps:
        pose = Pose()
        pose.position = Point(x=p.position_mm[0] * MM, y=p.position_mm[1] * MM, z=p.position_mm[2] * MM)
        pose.orientation = tool_orientation(p.approach, p.yaw_deg)
        pa.poses.append(pose)
    return pa


def pose_stamped(p: PickPose, stamp, frame_id: str) -> PoseStamped:
    ps = PoseStamped()
    ps.header = header(stamp, frame_id)
    ps.pose.position = Point(x=p.position_mm[0] * MM, y=p.position_mm[1] * MM, z=p.position_mm[2] * MM)
    ps.pose.orientation = tool_orientation(p.approach, p.yaw_deg)
    return ps


def boxes_to_markers(boxes, T_base_cam: np.ndarray, plan_pps: Sequence[PickPose], stamp, frame_id: str,
                     ns: str = "boxes") -> MarkerArray:
    """상면 CUBE(신뢰도=투명도, inferred=주황, 계획 밖=회색) + TEXT 라벨 + 다음 픽 ARROW. 첫 요소는 DELETEALL.

    모든 박스를 base 좌표로 옮겨 그린다(계획에 못 든 저신뢰 박스도 위치는 보여야 운영자가 판단할 수 있다).
    """
    ma = MarkerArray()
    clear = Marker()
    clear.header = header(stamp, frame_id)
    # ns/id 를 다른 마커와 겹치지 않게 — rviz2 는 한 MarkerArray 안에 같은 (ns,id) 가 있으면
    # Error 상태를 띄운다. DELETEALL 은 ns/id 와 무관하게 디스플레이의 마커를 전부 지운다.
    clear.ns = ns + "_clear"
    clear.id = 0
    clear.action = Marker.DELETEALL
    clear.type = Marker.CUBE
    clear.scale = Vector3(x=0.1, y=0.1, z=0.1)
    clear.pose.orientation = Quaternion(w=1.0)
    clear.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
    ma.markers.append(clear)

    planned = {p.box_id for p in plan_pps}
    for b in boxes:
        p = box_to_pick_pose(b, T_base_cam)
        m = Marker()
        m.header = header(stamp, frame_id)
        m.ns, m.id, m.type, m.action = ns, int(b.id), Marker.CUBE, Marker.ADD
        # 상면 위 2 cm 에 3 cm 두께 판으로 그린다 — 같은 높이에 얇게 두면 포인트클라우드에 묻힌다
        m.pose.position = Point(x=p.position_mm[0] * MM, y=p.position_mm[1] * MM, z=p.position_mm[2] * MM + 0.02)
        m.pose.orientation = quat_from_matrix(rot_z(p.yaw_deg))
        L, W = float(b.dims_mm[0]) * MM, float(b.dims_mm[1]) * MM
        m.scale = Vector3(x=max(L, 0.01), y=max(W, 0.01), z=0.03)
        inferred = getattr(b, "source", "detected") == "inferred"
        if inferred:
            rgb = (1.0, 0.6, 0.1)
        elif b.id in planned:
            rgb = (0.1, 0.85, 0.3)
        else:
            rgb = (0.55, 0.55, 0.55)
        # 불투명으로 그린다. 반투명(alpha<1) 큐브는 소프트웨어 GL(Xvfb/llvmpipe)에서 조명에 씻겨 회백색으로
        # 보였다. 신뢰도는 라벨에 표기한다.
        m.color = ColorRGBA(r=rgb[0], g=rgb[1], b=rgb[2], a=1.0)
        ma.markers.append(m)

        t = Marker()
        t.header = header(stamp, frame_id)
        t.ns, t.id, t.type, t.action = ns + "_label", int(b.id), Marker.TEXT_VIEW_FACING, Marker.ADD
        t.pose.position = Point(x=m.pose.position.x, y=m.pose.position.y, z=m.pose.position.z + 0.05)
        t.pose.orientation = Quaternion(w=1.0)
        t.scale = Vector3(x=0.035, y=0.035, z=0.035)
        t.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.95)
        t.text = f"#{b.id} {b.confidence:.2f}"       # 치수는 /perception/status·JSON 에 있으므로 라벨은 짧게
        ma.markers.append(t)

    if plan_pps:                            # 다음 픽: pre-pick -> 상면 중심 화살표
        p = plan_pps[0]
        a = Marker()
        a.header = header(stamp, frame_id)
        a.ns, a.id, a.type, a.action = ns + "_next", 0, Marker.ARROW, Marker.ADD
        a.pose.orientation = Quaternion(w=1.0)
        a.points = [Point(x=p.pre_pick_mm[0] * MM, y=p.pre_pick_mm[1] * MM, z=p.pre_pick_mm[2] * MM),
                    Point(x=p.position_mm[0] * MM, y=p.position_mm[1] * MM, z=p.position_mm[2] * MM)]
        a.scale = Vector3(x=0.02, y=0.05, z=0.06)
        a.color = ColorRGBA(r=0.15, g=0.55, b=1.0, a=1.0)
        ma.markers.append(a)
    return ma


def decision_to_string(dec, extra: dict | None = None) -> String:
    d = json.loads(dec.log_line())
    if extra:
        d.update(extra)
    return String(data=json.dumps(d, ensure_ascii=False))


_LEVEL = {"OK": DiagnosticStatus.OK, "LAYER_EMPTY": DiagnosticStatus.OK,
          "LOW_CONFIDENCE": DiagnosticStatus.WARN, "RETAKE": DiagnosticStatus.WARN,
          "NO_SURFACE": DiagnosticStatus.ERROR}
_HEALTH = {"OK": DiagnosticStatus.OK, "WARN": DiagnosticStatus.WARN, "FAIL": DiagnosticStatus.ERROR,
           "NO_REFERENCE": DiagnosticStatus.WARN, "INSUFFICIENT": DiagnosticStatus.WARN}


def diagnostics(dec, health: dict, stamp, hw_id: str = "tof") -> DiagnosticArray:
    da = DiagnosticArray()
    da.header = header(stamp, "")
    s1 = DiagnosticStatus(level=_LEVEL.get(dec.status, DiagnosticStatus.WARN),
                          name="robotsim_perception/decision", message=f"{dec.status}: {dec.reason}",
                          hardware_id=hw_id)
    s1.values = [KeyValue(key="n_boxes", value=str(dec.n_boxes)),
                 KeyValue(key="n_pickable", value=str(dec.n_pickable)),
                 KeyValue(key="valid_frac", value=f"{dec.valid_frac:.3f}"),
                 KeyValue(key="latency_ms", value=f"{dec.latency_ms:.1f}")]
    s2 = DiagnosticStatus(level=_HEALTH.get(health.get("status", "WARN"), DiagnosticStatus.WARN),
                          name="robotsim_perception/camera_drift",
                          message=str(health.get("status")), hardware_id=hw_id)
    s2.values = [KeyValue(key=k, value=str(v)) for k, v in health.items() if k != "status"]
    da.status = [s1, s2]
    return da


# ------------------------------------------------------------------ 대차(카트) 모드
# cart.CartResult -> TF(tof_optical -> cart), PoseStamped(고리, tof_optical), MarkerArray(cart 프레임), 진단.
# 대차 프레임은 프레임마다 다시 추정되므로 정적이 아니라 동적 TF 로 낸다.

def cart_transform(res, stamp, cam_frame: str, cart_frame: str) -> TransformStamped:
    """p_cart = R @ p_cam + t (mm)  ->  TF parent=cam_frame, child=cart_frame.

    TF 는 '자식 프레임의 원점·축을 부모 좌표로' 표현하므로 회전 = R^T, 이동 = -R^T t (m)."""
    R = np.asarray(res.R_cart, float)
    t = np.asarray(res.t_cart, float)
    ts = TransformStamped()
    ts.header = header(stamp, cam_frame)
    ts.child_frame_id = cart_frame
    o = -R.T @ t
    ts.transform.translation = Vector3(x=float(o[0]) * MM, y=float(o[1]) * MM, z=float(o[2]) * MM)
    ts.transform.rotation = quat_from_matrix(R.T)
    return ts


def hook_pose_stamped(res, stamp, cam_frame: str) -> PoseStamped:
    """고리 wall_top 을 카메라 프레임 PoseStamped 로. 자세 = 대차 축(도킹 목표 자세)."""
    ps = PoseStamped()
    ps.header = header(stamp, cam_frame)
    p = res.hook_cam_mm
    ps.pose.position = Point(x=p[0] * MM, y=p[1] * MM, z=p[2] * MM)
    ps.pose.orientation = quat_from_matrix(np.asarray(res.R_cart, float).T)
    return ps


def cart_markers(res, stamp, cart_frame: str, ns: str = "cart") -> MarkerArray:
    """대차 프레임 마커: 데크 판, 림 라인(v'=0), 레일 안쪽 벽(u'=0, u'=gap), 고리 기둥 + 라벨. 첫 요소 DELETEALL."""
    ma = MarkerArray()
    clear = Marker()
    clear.header = header(stamp, cart_frame)
    clear.ns, clear.id, clear.action, clear.type = ns + "_clear", 0, Marker.DELETEALL, Marker.CUBE
    clear.scale = Vector3(x=0.1, y=0.1, z=0.1)
    clear.pose.orientation = Quaternion(w=1.0)
    clear.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
    ma.markers.append(clear)
    if res.status != "OK":
        return ma

    def mk(mid, mtype, rgb, scale, ns_suffix=""):
        m = Marker()
        m.header = header(stamp, cart_frame)
        m.ns, m.id, m.type, m.action = ns + ns_suffix, mid, mtype, Marker.ADD
        m.pose.orientation = Quaternion(w=1.0)
        m.scale = Vector3(x=scale[0], y=scale[1], z=scale[2])
        m.color = ColorRGBA(r=rgb[0], g=rgb[1], b=rgb[2], a=1.0)
        return m

    u0, u1, v0, v1 = (float(x) * MM for x in res.deck_extent_cart_mm)
    deck = mk(0, Marker.CUBE, (0.30, 0.38, 0.50), (max(u1 - u0, 0.05), max(v1 - v0, 0.05), 0.02))
    # 포인트클라우드(h≈0)가 판 위에 보이도록 판은 h -3..-1 cm 에 둔다
    deck.pose.position = Point(x=(u0 + u1) / 2, y=(v0 + v1) / 2, z=-0.02)
    ma.markers.append(deck)

    rim = mk(1, Marker.LINE_STRIP, (0.95, 0.35, 0.95), (0.006, 0.0, 0.0))
    rim.points = [Point(x=u0, y=0.0, z=0.003), Point(x=u1, y=0.0, z=0.003)]
    ma.markers.append(rim)

    rails = mk(2, Marker.LINE_LIST, (1.0, 0.85, 0.2), (0.006, 0.0, 0.0))
    rails.points = [Point(x=0.0, y=v0, z=0.03), Point(x=0.0, y=-0.05, z=0.03)]
    gap = res.cart.get("rail_gap_mm")
    if gap:
        g = float(gap) * MM
        rails.points += [Point(x=g, y=v0, z=0.03), Point(x=g, y=-0.05, z=0.03)]
    ma.markers.append(rails)

    u, v, h = (float(x) * MM for x in res.hook_cart_mm)
    hook = mk(3, Marker.CUBE, (0.1, 0.85, 0.3), (0.03, 0.04, max(h, 0.02)))
    hook.pose.position = Point(x=u, y=v - 0.02, z=h / 2)        # 벽면이 v'=0(림)에 있고 몸통은 뒤(-v')로
    ma.markers.append(hook)

    label = mk(4, Marker.TEXT_VIEW_FACING, (1.0, 1.0, 1.0), (0.025, 0.025, 0.025), "_label")
    label.pose.position = Point(x=u, y=v + 0.05, z=h + 0.08)   # 카메라 쪽(+v')·위로 띄워 TF 이름과 겹치지 않게
    hu, hv, hh = res.hook_cart_mm
    label.text = f"hook u={hu:.0f} v={hv:.0f} h={hh:.0f} mm"
    ma.markers.append(label)
    return ma


_CART_LEVEL = {"OK": DiagnosticStatus.OK, "NO_HOOK": DiagnosticStatus.WARN,
               "NO_CART_FRAME": DiagnosticStatus.WARN, "NO_PLANE": DiagnosticStatus.ERROR}


def cart_status_string(res, extra: dict | None = None) -> String:
    d = json.loads(res.log_line())
    if extra:
        d.update(extra)
    return String(data=json.dumps(d, ensure_ascii=False))


def cart_diagnostics(res, stamp, hw_id: str = "tof") -> DiagnosticArray:
    da = DiagnosticArray()
    da.header = header(stamp, "")
    st = DiagnosticStatus(level=_CART_LEVEL.get(res.status, DiagnosticStatus.WARN),
                          name="robotsim_cart/hook", message=f"{res.status}: {res.reason}", hardware_id=hw_id)
    vals = {"valid_frac": f"{res.valid_frac:.3f}", "latency_ms": f"{res.latency_ms:.1f}"}
    if res.plane:
        vals["plate_mad_mm"] = str(res.plane["plate_mad_mm"])
        vals["camera_height_mm"] = str(res.plane["camera_height_mm"])
    for k in ("rim_fit_mad_mm", "rail_edge_mad_mm", "rim_yaw_deg", "rail_gap_mm"):
        if k in res.cart:
            vals[k] = str(res.cart[k])
    if res.hook_cart_mm:
        vals["hook_cart_mm"] = ",".join(f"{x:.1f}" for x in res.hook_cart_mm)
    st.values = [KeyValue(key=k, value=v) for k, v in vals.items()]
    da.status = [st]
    return da
