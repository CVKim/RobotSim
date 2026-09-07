# -*- coding: utf-8 -*-
"""픽 포즈 요(yaw) 규약: rect_px 에서 뽑을 때는 항상 '긴 변' 의 각도여야 한다.

cv2.minAreaRect 의 angle 은 첫 변(width) 기준이라 짧은 변이 width 로 잡히면 장축과 90도 어긋난다. 트윈 연결 실행에서
같은 자세의 박스가 프레임마다 요 0 또는 90 으로 나와, 팔이 90 인 박스를 그대로 90도 돌려 놓아 이웃과 겹쳐 튕겨 나갔다(12박스 중 6)."""
import numpy as np

from robotsim_perception.pose import box_to_pick_pose, topdown_camera_transform

T = topdown_camera_transform(4183.0)


def _box(rect_px, ang_deg=None):
    b = {"center_mm": (100.0, -50.0, 3257.0), "normal": (0.0, 0.0, -1.0), "dims_mm": (293.0, 219.0),
         "confidence": 0.9, "id": 3, "rect_px": rect_px}
    if ang_deg is not None:
        b["ang_deg"] = ang_deg
    return b


def test_rect_width_is_long_side_keeps_angle():
    p = box_to_pick_pose(_box(((320.0, 240.0), (98.0, 73.0), 2.0)), T)
    assert abs(p.yaw_deg - (-2.0 % 180.0)) < 1e-6          # 카메라 +2도 -> Y 반전 base 에서 178도


def test_rect_width_is_short_side_adds_90():
    a = box_to_pick_pose(_box(((320.0, 240.0), (73.0, 98.0), 2.0)), T)     # width 가 짧은 변
    b = box_to_pick_pose(_box(((320.0, 240.0), (98.0, 73.0), 92.0)), T)    # 같은 장축을 각도 92 로 표현한 것
    assert abs(a.yaw_deg - b.yaw_deg) < 1e-6
    assert abs(((a.yaw_deg - 88.0) + 90) % 180 - 90) < 1e-6              # 장축 = 카메라 92도 -> base 88도


def test_same_box_two_rect_conventions_give_same_yaw():
    """OpenCV 가 같은 사각형을 (w,h,ang) 또는 (h,w,ang+90) 으로 줄 수 있다 — 요는 같아야 한다."""
    for ang in (0.0, 15.0, 45.0, 89.0):
        r1 = ((300.0, 200.0), (98.0, 73.0), ang)
        r2 = ((300.0, 200.0), (73.0, 98.0), (ang - 90.0))
        y1 = box_to_pick_pose(_box(r1), T).yaw_deg
        y2 = box_to_pick_pose(_box(r2), T).yaw_deg
        assert abs(((y1 - y2) + 90) % 180 - 90) < 1e-6, (ang, y1, y2)


def test_explicit_ang_deg_wins_over_rect():
    p = box_to_pick_pose(_box(((0.0, 0.0), (10.0, 90.0), 0.0), ang_deg=30.0), T)
    assert abs(p.yaw_deg - 150.0) < 1e-6                                   # -30 % 180
    assert np.allclose(p.approach, (0, 0, -1))
