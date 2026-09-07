# -*- coding: utf-8 -*-
"""대차 도킹 — AGV(차동 구동, 운동학)가 카메라로 본 고리 포즈를 향해 다가가 결합 위치에 서는 폐루프의 순수 파이썬 부분.

좌표계
  평면 좌표 (u, v, h): cart.plane_frame — 데크 플레이트 평면에서 카메라 발밑이 원점, h 는 위. 카메라(AGV 에 고정)가 보는 방향은 −v.
  이 시뮬에서는 평면 좌표를 AGV 프레임으로 본다: 데크 플레이트가 바닥과 평행하므로 실제 시스템의 base_link 와 축이 같고,
  실제라면 카메라→base_link 외참(TF)이 하는 일을 여기서는 플레이트 평면 추정이 대신한다.
  상대 포즈 RelPose(hook_u, rim_v, yaw): 대차에 정렬된 좌표(카메라 발밑 원점)에서 고리 위치와, 대차가 AGV 에 대해 돌아간 각.
  이것이 synthetic_cart.make_cart_frame 의 매개변수(hook_u_mm, rim_v_mm, rail_u_mm, cart_yaw_deg)이기도 하다.

운동학 (RelPose 는 '카메라에서 본 대차' 이므로 AGV 가 움직이면 반대로 변한다)
  AGV 전진 방향 = 평면 −v 축 = 대차 정렬 좌표에서 (−sin yaw, −cos yaw). 전진 v 는 카메라 발밑을 그 방향으로 옮기므로
  고리의 상대 위치는 (+v·dt·sin yaw, +v·dt·cos yaw) 만큼 변한다(yaw 0 이면 rim_v 가 0 으로 다가온다). 요 각속도 ω(좌회전 +)는
  yaw 를 −ω·dt 만큼 바꾼다(카메라 발밑 기준 회전이라 상대 위치는 그대로).

제어 (pure pursuit 변형)
  측정 = 고리의 평면 좌표 (u_h, v_h) 와 대차 요 ψ (cart.analyze_cart 의 hook_plane_mm, cart['rim_yaw_deg']).
  목표점 T = 고리 + L · Rz(ψ)·(0, 1) — 고리에서 대차 중심선을 따라 AGV 쪽으로 L 만큼 나온 점. 조향 ω = k_w · atan2(T_u, −T_v)
  (전진 방향 −v 에서 T 를 향한 부호 있는 각). 전진 v 는 결합 거리까지 남은 길이에 비례, 조향각이 크면 줄인다.
  완료: |u_h| < tol_u, |v_h − dock_v| < tol_v, |ψ| < tol_yaw.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

RAIL_OFFSET_MM = 36.0     # 고리 u − 왼쪽 레일 안쪽 벽 u (합성 대차 기본값 −80 − (−116); 실측 hook_cart u' 36 mm)


@dataclass
class RelPose:
    hook_u_mm: float
    rim_v_mm: float
    yaw_deg: float

    def render_kwargs(self, rail_offset_mm: float = RAIL_OFFSET_MM) -> dict:
        """synthetic_cart.make_cart_frame 매개변수."""
        return dict(hook_u_mm=float(self.hook_u_mm), rim_v_mm=float(self.rim_v_mm),
                    rail_u_mm=float(self.hook_u_mm - rail_offset_mm), cart_yaw_deg=float(self.yaw_deg))

    def hook_plane(self) -> Tuple[float, float]:
        """정답: 고리의 평면 좌표 (u, v) = Rz(yaw) @ (hook_u, rim_v)."""
        th = math.radians(self.yaw_deg)
        c, s = math.cos(th), math.sin(th)
        return (c * self.hook_u_mm - s * self.rim_v_mm, s * self.hook_u_mm + c * self.rim_v_mm)

    def to_dict(self) -> dict:
        return asdict(self)


def step(rp: RelPose, v_mm_s: float, w_rad_s: float, dt_s: float, substeps: int = 10) -> RelPose:
    """차동 구동 운동학 한 스텝 (상수 v, ω). 작은 구간으로 나눠 적분한다."""
    u, v, yaw = float(rp.hook_u_mm), float(rp.rim_v_mm), math.radians(rp.yaw_deg)
    h = dt_s / substeps
    for _ in range(substeps):
        yaw_mid = yaw - 0.5 * w_rad_s * h
        u += v_mm_s * h * math.sin(yaw_mid)
        v += v_mm_s * h * math.cos(yaw_mid)
        yaw -= w_rad_s * h
    return RelPose(u, v, math.degrees(yaw))


@dataclass
class DockController:
    """도킹 제어기 (상태 기계; 한 도킹에 인스턴스 하나).

    follow  : 대차 중심선(고리를 지나 대차 방향 cdir) 위, AGV 발밑의 투영점보다 lookahead 만큼 앞(결합점 D 를 넘지 않게)의 점을
              pure pursuit 한다 — 직선 경로 pure pursuit 은 선으로 수렴하면서 방향도 맞춘다. v ∝ 결합점까지 선 방향 남은 거리 s.
    finish  : s 가 허용치 안이면 제자리에서 요를 맞추고, 그래도 측방 오차가 남으면 backoff.
    backoff : 선 방향으로 entry_mm 까지 후진(후진 중에는 요만 유지) 뒤 follow 로 재접근 — 라인 추종에 필요한 거리를 확보한다.
    D 를 지나쳤으면(s < −tol) 천천히 후진해 거리를 맞춘다.
    부호 규약: 평면 좌표에서 전진 = −v, 왼쪽 = +u, ω > 0 = 좌회전 (dock.step 과 tests/test_dock.py 가 이를 검사)."""
    dock_v_mm: float = -200.0        # 결합 위치: 고리가 카메라 발밑 앞 200 mm, 정면(u 0), 요 0
    lookahead_mm: float = 150.0      # 중심선 위 추종점까지의 거리 (절제: 150~250 사이 차이 작음)
    entry_mm: float = 200.0          # backoff 뒤 재접근을 시작하는 거리. 결합 200 + 200 = 고리 400 mm — 합성 고리 검출 한계(~500 mm) 안.
                                     # 400 으로 두면 물러난 자리에서 고리가 안 보여(NO_HOOK) 30회 중 12회가 멈춰 있었다(1차 평가)
    k_w: float = 1.5                 # 조향 이득 (rad/s per rad)
    k_psi: float = 1.2               # 제자리 요 정렬 이득
    k_v: float = 0.5                 # 전진 이득 (mm/s per mm) — 0.8 보다 backoff 가 절반(7/60 vs 15/60), 시간은 비슷
    lost_creep_frames: int = 6       # 측정이 없을 때 이만큼은 천천히 움직여 고리를 다시 찾는다, 그 뒤 정지
    lost_frames: int = 0
    last_s_mm: float = 1e9           # 마지막 측정의 결합점까지 거리 — 가까이서 잃었으면(지나침) 뒤로, 멀어서 잃었으면 앞으로
    v_max_mm_s: float = 150.0
    v_min_mm_s: float = 15.0
    w_max_rad_s: float = 0.5
    tol_u_mm: float = 10.0
    tol_v_mm: float = 10.0
    tol_yaw_deg: float = 2.0
    creep_deg: float = 45.0          # 추종점 방향이 이보다 크게 벗어나면 기어가며 돈다
    mode: str = "follow"

    def reset(self):
        self.mode = "follow"
        self.lost_frames = 0
        self.last_s_mm = 1e9

    def on_miss(self) -> Tuple[float, float]:
        """측정 실패 프레임의 명령. 합성 고리는 림 거리 500 mm 밖(멀리)과 ~200 mm 안(가까이, 화면 아래로 벗어남)에서 안 보인다.
        마지막 측정에서 결합점이 가까웠으면(지나침) 뒤로, 멀었으면 앞으로 잠시 기어가며 다시 찾고, lost_creep_frames 를 넘으면 정지."""
        self.lost_frames += 1
        if self.mode != "docked" and self.lost_frames <= self.lost_creep_frames:
            sign = -1.0 if self.last_s_mm < 120.0 else 1.0
            return float(sign * self.v_min_mm_s * 2.0), 0.0
        return 0.0, 0.0

    def errors(self, u_h: float, v_h: float, yaw_deg: float) -> dict:
        return dict(e_u=float(u_h), e_v=float(v_h - self.dock_v_mm), e_yaw=float(yaw_deg))

    def is_docked(self, u_h: float, v_h: float, yaw_deg: float) -> bool:
        e = self.errors(u_h, v_h, yaw_deg)
        return abs(e["e_u"]) < self.tol_u_mm and abs(e["e_v"]) < self.tol_v_mm and abs(e["e_yaw"]) < self.tol_yaw_deg

    def _clip(self, v: float, w: float) -> Tuple[float, float]:
        return (max(-self.v_max_mm_s, min(self.v_max_mm_s, v)), max(-self.w_max_rad_s, min(self.w_max_rad_s, w)))

    def geometry(self, u_h: float, v_h: float, yaw_deg: float) -> dict:
        """선 방향 cdir, 발밑 투영점 Q, 결합점 D, 남은 거리 s(+ 앞), 측방 오차 e_lat(+ = 발밑이 선의 왼쪽)."""
        psi = math.radians(yaw_deg)
        cdir = (math.sin(psi), -math.cos(psi))
        # 발밑 O=(0,0) 을 중심선(H + t·cdir)에 투영: t = (O − H)·cdir
        t = (-u_h) * cdir[0] + (-v_h) * cdir[1]
        Q = (u_h + t * cdir[0], v_h + t * cdir[1])
        t_dock = self.dock_v_mm                  # D = H + t_dock·cdir (dock_v 는 음수: 고리에서 AGV 쪽으로)
        s = t_dock - t
        e_lat = cdir[0] * (-v_h) - cdir[1] * (-u_h)
        return dict(psi=psi, cdir=cdir, Q=Q, s=s, e_lat=e_lat)

    def command(self, u_h: float, v_h: float, yaw_deg: float) -> Tuple[float, float, bool]:
        """측정(고리 평면 좌표 u,v [mm], 대차 요 ψ [deg]) -> (v [mm/s], ω [rad/s], 완료 여부)."""
        self.lost_frames = 0
        if self.is_docked(u_h, v_h, yaw_deg):
            self.mode = "docked"
            return 0.0, 0.0, True
        g = self.geometry(u_h, v_h, yaw_deg)
        psi, cdir, Q, s, e_lat = g["psi"], g["cdir"], g["Q"], g["s"], g["e_lat"]
        self.last_s_mm = float(s)
        if self.mode == "docked":
            self.mode = "follow"
        if self.mode == "backoff":
            if s >= self.entry_mm:
                self.mode = "follow"
            else:
                return (*self._clip(-4.0 * self.v_min_mm_s, self.k_psi * psi), False)
        if s > self.tol_v_mm:
            # 추종점: 투영점에서 선 방향으로 lookahead 앞 (결합점을 넘지 않게)
            ahead = min(self.lookahead_mm, s)
            T = (Q[0] + ahead * cdir[0], Q[1] + ahead * cdir[1])
            err = math.atan2(T[0], -T[1]) if T[1] < -1e-6 else math.copysign(math.pi / 2, T[0] if abs(T[0]) > 1e-9 else 1.0)
            w = self.k_w * err
            v = max(self.k_v * s, self.v_min_mm_s) * max(math.cos(err), 0.0)
            if abs(err) > math.radians(self.creep_deg):
                v = self.v_min_mm_s
            v = max(v, self.v_min_mm_s)
            return (*self._clip(v, w), False)
        if s < -self.tol_v_mm:
            v = -max(min(self.k_v * -s, self.v_max_mm_s), self.v_min_mm_s)   # 지나침 -> 천천히 후진
            return (*self._clip(v, self.k_psi * psi), False)
        # 거리 OK: 요 먼저, 그다음 측방
        if abs(psi) >= math.radians(self.tol_yaw_deg):
            return (*self._clip(0.0, self.k_psi * psi), False)
        if abs(e_lat) >= self.tol_u_mm:
            self.mode = "backoff"
            return (*self._clip(-4.0 * self.v_min_mm_s, 0.0), False)
        return 0.0, 0.0, False


def measurement_from_result(res) -> Optional[Tuple[float, float, float]]:
    """cart.analyze_cart 결과 -> (u_h, v_h, yaw_deg) 또는 None (고리/대차 프레임을 못 찾음)."""
    if res is None or getattr(res, "status", "") != "OK" or not res.hook_plane_mm:
        return None
    u, v, _ = res.hook_plane_mm
    yaw = float(res.cart.get("rim_yaw_deg", 0.0)) if isinstance(res.cart, dict) else 0.0
    return float(u), float(v), yaw


def simulate(rp0: RelPose, ctrl: DockController, measure, dt_s: float = 0.5, max_steps: int = 60,
             miss_policy: str = "creep", latency_steps: int = 0) -> dict:
    """도킹 한 회. measure(rp) -> (u_h, v_h, yaw) | None (인식 실패). miss_policy 'creep' = 제어기의 on_miss (기어가기), 'hold' = 정지.
    latency_steps: 측정이 몇 스텝 전 포즈의 것인가 (ROS 루프에서 인식 지연 0.45 s + 명령 유지 0.5 s ≈ 2 스텝 — 0 으로 조율한 이득은
    실제 루프에서 요가 발산했다).

    반환 dict(success, steps, final RelPose, final errors(정답 기준), misses, path[(hook_u, rim_v, yaw)...])."""
    rp = rp0
    path = [rp.to_dict()]
    misses = 0
    ctrl.reset()
    hist = [rp]
    for k in range(max_steps):
        m = measure(hist[max(0, len(hist) - 1 - int(latency_steps))])
        if m is None:
            misses += 1
            v, w = ctrl.on_miss() if miss_policy == "creep" else (0.0, 0.0)
            done = False
        else:
            v, w, done = ctrl.command(*m)
        if done:
            gu, gv = rp.hook_plane()
            return dict(success=True, steps=k, final=rp.to_dict(), errors=ctrl.errors(gu, gv, rp.yaw_deg),
                        truth_docked=ctrl.is_docked(gu, gv, rp.yaw_deg), misses=misses, path=path)
        rp = step(rp, v, w, dt_s)
        hist.append(rp)
        path.append(rp.to_dict())
    gu, gv = rp.hook_plane()
    return dict(success=False, steps=max_steps, final=rp.to_dict(), errors=ctrl.errors(gu, gv, rp.yaw_deg),
                truth_docked=ctrl.is_docked(gu, gv, rp.yaw_deg), misses=misses, path=path)
