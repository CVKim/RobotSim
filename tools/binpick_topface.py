# -*- coding: utf-8 -*-
"""T1 프로토타입: 빈피킹 현장 ToF에서 박스 상면 검출 + mm 치수 측정.

원리 (딥러닝 없이 기하만):
  1. D 히스토그램에서 최상층(카메라에 가장 가까운) 평면 깊이 피크 탐색
  2. 피크 ±tol 픽셀 → 상면 마스크 → 모폴로지 정리 → 연결요소 = 박스 후보
  3. 각 박스: X/Y 좌표맵(mm)에서 min-area rect → 상면 L×W (mm)

v2 (detect_boxes_v2): 위 v1 을 그대로 감싸고
  - 박스별 신뢰도(유효픽셀 충전율·평면 RMS·직사각형성·SKU 치수 부합) 를 붙이고
  - 신뢰 박스들로 격자(방향·치수·피치)를 추정해 v1 이 놓친 셀(병합·조각 실패, 무효 블롭)을 상면 지지도로 보완('inferred')
  - 상면층 피크가 바닥/티어시트에 밀려 v1 이 0개를 낸 프레임은 다른 히스토그램 피크 층으로 재시도(layer_fallback).
  실측 30프레임 검증(RGB 대조, 로컬): v1 148 → v2 167 박스, 추가 19개 전부 실제 박스(2층 4x3 격자의 중앙 2개가 v1 에서
  병합 실패, 층 말미 1~2개 잔여 박스가 피크 전환으로 누락), 빈 크레이트 3세션·빈 팔레트·티어시트 프레임 오검출 0.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mim_loader import load_session, valid_mask


def _roi_depth_hist(D, valid, bin_mm=10, roi_frac=0.25):
    """중앙 ROI(팔레트 영역) 유효 깊이 히스토그램 (find_top_layer 와 동일 정의)."""
    h, w = D.shape
    roi = np.zeros_like(valid)
    roi[int(h * roi_frac):int(h * (1 - roi_frac)), int(w * roi_frac):int(w * (1 - roi_frac))] = True
    d = D[valid & roi]
    lo, hi = np.percentile(d, [1, 99])
    bins = np.arange(lo, hi + bin_mm, bin_mm)
    hist, edges = np.histogram(d, bins=bins)
    return hist, edges


def _refine_peak(hist, edges, i):
    """피크 주변 ±2bin 가중 평균으로 서브빈 정밀화."""
    j0, j1 = max(0, i - 2), min(len(hist), i + 3)
    centers = 0.5 * (edges[j0:j1] + edges[j0 + 1:j1 + 1])
    return float(np.average(centers, weights=hist[j0:j1] + 1e-9))


def find_top_layer(D, valid, bin_mm=10, roi_frac=0.25):
    """중앙 ROI(팔레트 영역)의 깊이 히스토그램에서 최대 피크 = 박스 상면 깊이."""
    hist, edges = _roi_depth_hist(D, valid, bin_mm, roi_frac)
    return _refine_peak(hist, edges, int(np.argmax(hist)))


def _intensity_edges(I):
    """ToF 강도 → log 정규화 → Canny → 3x3 팽창 (박스 이음새 에지 맵, uint8 0/255)."""
    In = np.log1p(np.clip(I, 0, None))
    In = cv2.normalize(In, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    In = cv2.GaussianBlur(In, (3, 3), 0)
    edges = cv2.Canny(In, 40, 100)
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.dilate(edges, kernel3, iterations=1)


def _detect_boxes_at(sess, top_d, tol_mm=40, min_area_px=700):
    """detect_boxes 본체 — 상면 깊이 top_d 를 외부에서 지정하는 형태 (detect_boxes 와 동일 동작)."""
    D, X, Y = sess["D"], sess["X"], sess["Y"]
    valid = valid_mask(sess)

    mask = (np.abs(D - top_d) < tol_mm) & valid
    # 박스 간 이음새 = 깊이 그래디언트 에지 → 마스크에서 제거해 분리 유도
    Ds = cv2.medianBlur(D.astype(np.float32), 5)
    gx = cv2.Sobel(Ds, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(Ds, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)
    mask &= grad < 70.0  # mm/px (3m ToF 노이즈 감안)

    m8 = (mask * 255).astype(np.uint8)
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, kernel3, iterations=2)  # 점 구멍 메움
    m8 = cv2.morphologyEx(m8, cv2.MORPH_OPEN, kernel3, iterations=1)

    # ToF 강도(I) 채널 — 깊이와 픽셀 정합 완벽 → 박스 이음새를 에지로 절단
    edges = _intensity_edges(sess["I"])

    sep = m8.copy()
    sep[edges > 0] = 0
    n, labels = cv2.connectedComponents(sep)

    boxes = []
    for i in range(1, n):
        comp = (labels == i)
        if comp.sum() < min_area_px:
            continue
        # 에지로 깎인 영역 복원(살짝 팽창) 후 top-layer 마스크로 제한
        comp = cv2.dilate((comp * 255).astype(np.uint8), kernel3, iterations=2) > 0
        comp &= (m8 > 0)
        # m8 은 MORPH_CLOSE 를 거쳐 무효 픽셀 구멍이 메워져 있다. 그 픽셀의 X/Y 는
        # 센티넬(8191.75)이라 minAreaRect 가 13m 짜리 사각형을 만든다(셀 트윈에서 발견).
        # 미터릭 계산에는 반드시 유효 픽셀만 사용한다.
        comp &= valid
        if comp.sum() < min_area_px:
            continue
        ys, xs = np.nonzero(comp)
        # 미터릭 치수: X/Y 좌표맵(mm)에서 min-area rect
        pts_mm = np.stack([X[comp], Y[comp]], axis=-1).astype(np.float32)
        rect_mm = cv2.minAreaRect(pts_mm)
        (w_mm, h_mm) = rect_mm[1]
        # 픽셀 상 표시용 rect
        rect_px = cv2.minAreaRect(np.stack([xs, ys], axis=-1).astype(np.float32))
        # 직사각형성/종횡비 필터 (바닥 줄무늬 등 오검출 제거)
        rect_area = max(rect_px[1][0] * rect_px[1][1], 1)
        fill = comp.sum() / rect_area
        aspect = max(w_mm, h_mm) / max(min(w_mm, h_mm), 1)
        if fill < 0.6 or aspect > 3.5:
            continue
        boxes.append({
            "area_px": int(comp.sum()),
            "dims_mm": (round(max(w_mm, h_mm), 1), round(min(w_mm, h_mm), 1)),
            "depth_mm": round(float(np.median(D[comp])), 1),
            "rect_px": rect_px,
        })
    return top_d, mask, boxes


def detect_boxes(sess, tol_mm=40, min_area_px=700):
    D = sess["D"]
    valid = valid_mask(sess)
    top_d = find_top_layer(D, valid)
    return _detect_boxes_at(sess, top_d, tol_mm=tol_mm, min_area_px=min_area_px)


def render_overlay(sess, top_d, mask, boxes, out_path):
    D = sess["D"]
    valid = valid_mask(sess)
    lo, hi = np.percentile(D[valid], [2, 98])
    norm = np.clip((D - lo) / max(hi - lo, 1), 0, 1)
    img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_BONE)
    img[~valid] = (20, 20, 20)
    img[mask] = (0.55 * img[mask] + 0.45 * np.array([0, 140, 255])).astype(np.uint8)
    for i, b in enumerate(boxes):
        pts = cv2.boxPoints(b["rect_px"]).astype(np.int32)
        cv2.polylines(img, [pts], True, (0, 255, 0), 2)
        cx, cy = pts.mean(axis=0).astype(int)
        cv2.putText(img, f'{i+1}', (cx - 8, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f'top layer @ {top_d:.0f}mm, {len(boxes)} boxes',
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imwrite(str(out_path), img)


# ---------------------------------------------------------------------------
# v2: 격자(lattice) 기반 결손 보완 + 박스별 신뢰도  (detect_boxes 는 그대로 유지, 래핑만)
# ---------------------------------------------------------------------------
SKU_PRIOR_MM = (293.0, 219.0)   # 실측 30세션 148박스 평균 상면 (L, W) mm
SKU_TOL = 0.15                  # 치수 사전 허용 ±15%


def _plane_rms(X, Y, D, m):
    """m 픽셀의 (X,Y,D) 최소자승 평면 잔차 RMS(mm). 점이 부족하면 None."""
    if m.sum() < 30:
        return None
    pts = np.stack([X[m], Y[m], D[m]], -1).astype(np.float64)
    q = pts - pts.mean(0)
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    return float(np.sqrt(np.mean((q @ vt[2]) ** 2)))


def _affine_px_from_mm(X, Y, m, n_max=20000):
    """상면 픽셀에서 [X,Y,1](mm) -> [u,v](px) 아핀 최소자승. 반환 M(3x2) 또는 None."""
    ys, xs = np.nonzero(m)
    if len(xs) < 200:
        return None
    if len(xs) > n_max:
        sel = np.linspace(0, len(xs) - 1, n_max).astype(int)
        xs, ys = xs[sel], ys[sel]
    A = np.stack([X[ys, xs], Y[ys, xs], np.ones(len(xs))], 1).astype(np.float64)
    B = np.stack([xs, ys], 1).astype(np.float64)
    M, *_ = np.linalg.lstsq(A, B, rcond=None)
    return M


def _dims_score(L, W, prior=SKU_PRIOR_MM, tol=SKU_TOL):
    dev = max(abs(L - prior[0]) / (tol * prior[0]), abs(W - prior[1]) / (tol * prior[1]))
    return float(np.clip(1.0 - dev, 0.0, 1.0))


def _plane_score(rms, rms_good=4.0, rms_bad=20.0):
    if rms is None:
        return 0.0
    return float(np.clip(1.0 - (rms - rms_good) / (rms_bad - rms_good), 0.0, 1.0))


def _box_geometry(X, Y, D, valid, top, b):
    """검출 박스 하나의 mm 기하(중심·장축각·치수) + 신뢰도 구성요소.

    confidence = mean(fill, plane, rect, dims)
      fill  : rect 내 유효 픽셀 비율
      plane : rect 내 상면 3D 점의 평면 피팅 RMS (4mm→1, 20mm→0)
      rect  : 직사각형성 = 연결요소 면적 / minAreaRect 면적
      dims  : SKU 사전(293x219 ±15%) 부합도 (사전 중심→1, 경계→0)
    """
    pts = cv2.boxPoints(b["rect_px"]).astype(np.int32)
    poly = np.zeros(D.shape, np.uint8)
    cv2.fillPoly(poly, [pts], 1)
    poly = poly > 0
    n_poly = max(int(poly.sum()), 1)
    m = poly & top
    fill_valid = float((poly & valid).sum() / n_poly)
    rect_ratio = float(min(b["area_px"] / max(b["rect_px"][1][0] * b["rect_px"][1][1], 1.0), 1.0))
    rms = _plane_rms(X, Y, D, m)
    if m.sum() >= 5:
        (cx, cy), (w, h), ang = cv2.minAreaRect(np.stack([X[m], Y[m]], 1).astype(np.float32))
        if w < h:
            w, h, ang = h, w, ang + 90.0
        has_center = True
    else:  # 상면 픽셀이 거의 없으면 중심 없음
        cx = cy = 0.0
        ang = 0.0
        has_center = False
    L, W = float(b["dims_mm"][0]), float(b["dims_mm"][1])
    comp = {"fill": fill_valid, "plane": _plane_score(rms), "rect": rect_ratio,
            "dims": _dims_score(L, W)}
    conf = float(np.mean(list(comp.values())))
    return {"center_mm": (float(cx), float(cy)), "ang_deg": float(ang % 180.0), "has_center": has_center,
            "L": L, "W": W, "plane_rms_mm": rms, "components": comp, "confidence": conf,
            "poly": poly, "top_px": m}


def _orient_class(ang_deg, theta_deg):
    d = (ang_deg - theta_deg) % 180.0
    return 0 if (d < 45.0 or d >= 135.0) else 1


def _estimate_lattice(geoms, default_gap=25.0):
    """신뢰 박스들로부터 격자: 지배 방향(90도 주기), 셀 치수, 방향군별 e1/e2 피치.

    반환 dict: theta_deg, e1, e2 (단위벡터), L, W, cls(박스별 방향군), ext, pitch[(cls, axis)], gap
      cls 0 = 장축이 e1 방향, cls 1 = 장축이 e2 방향;  ext[cls] = (e1 방향 폭, e2 방향 폭)
    피치는 같은 방향군 박스 쌍의 중심 간격(축 정렬된 이웃)의 중앙값, 쌍이 없으면 폭 + gap.
    """
    ang = np.radians([g["ang_deg"] for g in geoms])
    th = np.arctan2(np.sin(4 * ang).mean(), np.cos(4 * ang).mean()) / 4.0   # (-45,45] deg
    e1 = np.array([np.cos(th), np.sin(th)])
    e2 = np.array([-np.sin(th), np.cos(th)])
    L = float(np.median([g["L"] for g in geoms]))
    W = float(np.median([g["W"] for g in geoms]))
    cls = [_orient_class(g["ang_deg"], np.degrees(th)) for g in geoms]
    ext = {0: (L, W), 1: (W, L)}
    C = np.array([g["center_mm"] for g in geoms])
    pitches, gaps = {}, []
    for c in (0, 1):
        idx = [i for i, k in enumerate(cls) if k == c]
        for axis, (ea, eb) in enumerate(((e1, e2), (e2, e1))):
            ea_ext, eb_ext = ext[c][axis], ext[c][1 - axis]
            vals = []
            for a in range(len(idx)):
                for b_ in range(a + 1, len(idx)):
                    dlt = C[idx[b_]] - C[idx[a]]
                    da, db = abs(dlt @ ea), abs(dlt @ eb)
                    if 0.85 * ea_ext <= da <= 1.6 * ea_ext and db < 0.4 * eb_ext:
                        vals.append(da)
            if vals:
                pitches[(c, axis)] = float(np.median(vals))
                gaps.extend([v - ea_ext for v in vals])
    gap = float(np.clip(np.median(gaps), 0.0, 80.0)) if gaps else default_gap
    for c in (0, 1):
        for axis in (0, 1):
            pitches.setdefault((c, axis), ext[c][axis] + gap)
    return {"theta_deg": float(np.degrees(th)), "e1": e1, "e2": e2, "L": L, "W": W,
            "cls": cls, "ext": ext, "pitch": pitches, "gap": gap}


class _LatticeGrid:
    """상면 평면의 격자 좌표계(원점 c0, 축 e1/e2, res mm/셀)로 리샘플한 top/valid/occ/in-image 맵 + 적분영상.

    격자 좌표에서 후보 셀은 축정렬 직사각형이므로 합산이 O(1) → 오프셋 탐색을 벡터화.
    """

    def __init__(self, shape, M, c0, e1, e2, top, valid, occ, inten, res=6.0):
        H, W = shape
        Minv = np.linalg.inv(M[:2])
        mm = (np.array([[0, 0], [W, 0], [0, H], [W, H]], float) - M[2]) @ Minv
        ab = np.stack([(mm - c0) @ e1, (mm - c0) @ e2], 1)
        self.a0, self.b0 = float(ab[:, 0].min()), float(ab[:, 1].min())
        self.na = int(np.ceil((ab[:, 0].max() - self.a0) / res)) + 1
        self.nb = int(np.ceil((ab[:, 1].max() - self.b0) / res)) + 1
        self.res, self.c0, self.e1, self.e2 = res, np.asarray(c0, float), e1, e2
        A = np.zeros((2, 3))                     # dst(ia, ib) -> src px (WARP_INVERSE_MAP)
        A[:, 0] = res * (e1 @ M[:2])
        A[:, 1] = res * (e2 @ M[:2])
        A[:, 2] = (self.c0 + self.a0 * e1 + self.b0 * e2) @ M[:2] + M[2]
        self.A = A

        def warp(img):
            return cv2.warpAffine(img.astype(np.uint8), A, (self.na, self.nb),
                                  flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        self.top, self.valid, self.occ = warp(top), warp(valid), warp(occ)
        self.inimg = warp(np.ones(shape, np.uint8))
        self.I_top, self.I_valid = cv2.integral(self.top), cv2.integral(self.valid)
        self.I_in, self.I_occ = cv2.integral(self.inimg), cv2.integral(self.occ)
        # 강도 이미지의 격자축 방향 그래디언트 (경계 증거: 셀 변에 수직인 강도 변화). 무효 픽셀은 0.
        Ig = cv2.warpAffine(inten.astype(np.float32), A, (self.na, self.nb),
                            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        v = self.valid > 0
        Ga = np.abs(cv2.Sobel(Ig, cv2.CV_32F, 1, 0, ksize=3)) * v     # d/da (e1 방향)
        Gb = np.abs(cv2.Sobel(Ig, cv2.CV_32F, 0, 1, ksize=3)) * v     # d/db (e2 방향)
        tv = v & (self.top > 0)
        self.gscale = float(np.percentile((Ga + Gb)[tv], 95)) if tv.sum() > 100 else 1.0
        self.gscale = max(self.gscale, 1e-6)
        self.I_Ga, self.I_Gb = cv2.integral(Ga), cv2.integral(Gb)

    def set_near(self, pts_ab, r_mm):
        """격자 위 '팔레트 근방' 마스크: 신뢰 박스 중심들로부터 r_mm 이내 (거리변환 1회)."""
        src = np.full((self.nb, self.na), 255, np.uint8)
        for (pa, pb) in pts_ab:
            ia = int(np.clip(round((pa - self.a0) / self.res), 0, self.na - 1))
            ib = int(np.clip(round((pb - self.b0) / self.res), 0, self.nb - 1))
            src[ib, ia] = 0
        self.near = cv2.distanceTransform(src, cv2.DIST_L2, 5) * self.res <= r_mm

    def to_ab(self, P):
        d = np.asarray(P, float) - self.c0
        return float(d @ self.e1), float(d @ self.e2)

    def to_mm(self, a, b):
        return self.c0 + a * self.e1 + b * self.e2

    def dense_scan(self, h1, h2, support_min, contra_max, occ_max, in_image_min,
                   nms_mm=60.0, max_peaks=40):
        """셀(반폭 h1,h2)을 격자 전 위치에 대해 평가(박스필터) → 조건을 만족하는 점수 국소최대 위치(mm) 목록.

        점수 = support - 2*contra - 3*occ (제안 단계; 경계 증거·순위는 eval 에서). 국소최대 후 거리 NMS.
        """
        ka, kb = max(int(round(2 * h1 / self.res)), 1), max(int(round(2 * h2 / self.res)), 1)
        if ka >= self.na or kb >= self.nb:
            return []

        def S(I, ka_, kb_):
            return (I[kb_:, ka_:] - I[:-kb_, ka_:] - I[kb_:, :-ka_] + I[:-kb_, :-ka_]).astype(np.float64)
        n = S(self.I_in, ka, kb)
        nn = np.maximum(n, 1.0)
        support = S(self.I_top, ka, kb) / nn
        contra = (S(self.I_valid, ka, kb) - S(self.I_top, ka, kb)) / nn
        occ = S(self.I_occ, ka, kb) / nn
        in_image = np.minimum(n / float(ka * kb), 1.0)
        score = support - 2.0 * contra - 3.0 * occ
        H0, W0 = support.shape
        near = self.near[kb // 2:kb // 2 + H0, ka // 2:ka // 2 + W0] if hasattr(self, "near") else True
        ok = ((n > 0) & (in_image >= in_image_min) & (support >= support_min) & (contra <= contra_max)
              & (occ <= occ_max) & near)
        if not ok.any():
            return []
        sm = np.where(ok, score, -1e9).astype(np.float32)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * (ka // 2) + 1, 2 * (kb // 2) + 1))
        mx = cv2.dilate(sm, k)
        peaks = ok & (sm >= mx - 1e-6)
        pb, pa = np.nonzero(peaks)
        order = np.argsort(-sm[pb, pa])
        kept = []
        for i in order:
            pa_ = self.a0 + (pa[i] + 0.5 * ka) * self.res
            pb_ = self.b0 + (pb[i] + 0.5 * kb) * self.res
            if all(np.hypot(pa_ - qa, pb_ - qb) >= nms_mm for qa, qb in kept):
                kept.append((pa_, pb_))
                if len(kept) >= max_peaks:
                    break
        return [self.to_mm(pa_, pb_) for pa_, pb_ in kept]

    def _idx(self, a, b, h1, h2):
        ia0 = np.clip(np.round((a - h1 - self.a0) / self.res).astype(int), 0, self.na)
        ia1 = np.clip(np.round((a + h1 - self.a0) / self.res).astype(int), 0, self.na)
        ib0 = np.clip(np.round((b - h2 - self.b0) / self.res).astype(int), 0, self.nb)
        ib1 = np.clip(np.round((b + h2 - self.b0) / self.res).astype(int), 0, self.nb)
        return ia0, ia1, ib0, ib1

    @staticmethod
    def _sum(I, idx):
        ia0, ia1, ib0, ib1 = idx
        return (I[ib1, ia1] - I[ib0, ia1] - I[ib1, ia0] + I[ib0, ia0]).astype(np.float64)

    def eval(self, center, h1, h2, offsets, in_image_min=0.0, ring_mm=12.0, bnd_w=1.0, band_mm=6.0):
        """center(mm) 주변 offsets(mm, e1/e2 방향)에서 셀(반폭 h1,h2) 지지도 평가 — 벡터화.

        support   = 셀 내 상면 픽셀 비율 (무효 픽셀 = 구멍으로 허용)
        contra    = 셀 내 '유효하지만 상면 깊이가 아닌' 픽셀 비율 (아래층/바닥이 보임 = 박스 없음)
        occ_frac  = 기존 박스와 겹침 비율,  in_image = 셀의 화면 내 비율
        boundary  = 셀 4변의 경계 일관성 (-1~1; _boundary 참조: 틈/이음새 +, 박스 가로지름 0, 지나침 -)
        ring_top  = 셀 경계 바깥 ring_mm 띠의 상면 픽셀 비율 (참고용)
        score = support - 2*contra - 3*occ_frac + bnd_w*boundary 가 최대인 오프셋(in_image_min 충족)을 반환.
        """
        offs = np.asarray(offsets, float).reshape(-1, 2)
        a_c, b_c = self.to_ab(center)
        a, b = a_c + offs[:, 0], b_c + offs[:, 1]
        db = band_mm
        hb2, ha2 = max(h2 - db, 1.0), max(h1 - db, 1.0)
        # 오프셋마다 14개 rect 를 한 번에: 0 셀, 1 ring 바깥, 2-5 변 안쪽 띠(L,R,T,B), 6-9 변 바깥 띠, 10-13 변 ±db 띠
        CA = np.stack([a, a, a - h1 + db, a + h1 - db, a, a, a - h1 - db, a + h1 + db, a, a, a - h1, a + h1, a, a])
        CB = np.stack([b, b, b, b, b - h2 + db, b + h2 - db, b, b, b - h2 - db, b + h2 + db, b, b, b - h2, b + h2])
        WA = np.array([h1, h1 + ring_mm, db, db, ha2, ha2, db, db, ha2, ha2, db, db, ha2, ha2])[:, None]
        WB = np.array([h2, h2 + ring_mm, hb2, hb2, db, db, hb2, hb2, db, db, hb2, hb2, db, db])[:, None]
        idx = self._idx(CA, CB, WA, WB)
        S_in, S_top, S_valid = self._sum(self.I_in, idx), self._sum(self.I_top, idx), self._sum(self.I_valid, idx)
        n, n_top, n_valid = S_in[0], S_top[0], S_valid[0]
        n_exp = max((2 * h1 / self.res) * (2 * h2 / self.res), 1.0)
        in_image = np.minimum(n / n_exp, 1.0)
        ok = (n > 0) & (in_image >= in_image_min)
        if not ok.any():
            return None
        nn = np.maximum(n, 1.0)
        n_occ = self._sum(self.I_occ, (idx[0][0], idx[1][0], idx[2][0], idx[3][0]))
        support, contra, occ_frac = n_top / nn, (n_valid - n_top) / nn, n_occ / nn
        # 경계 일관성: 변마다 s = in*(1-out) + in*out*g - (1-in)*out  (틈 +1, 밀착 이음새 g, 가로지름 ~0, 지나침 -1)
        t_in = np.where(S_in[2:6] > 0, S_top[2:6] / np.maximum(S_in[2:6], 1.0), 0.0)
        t_out = np.where(S_in[6:10] > 0, S_top[6:10] / np.maximum(S_in[6:10], 1.0), 0.0)
        gsum = np.stack([self._sum(self.I_Ga, (idx[0][10], idx[1][10], idx[2][10], idx[3][10])),
                         self._sum(self.I_Ga, (idx[0][11], idx[1][11], idx[2][11], idx[3][11])),
                         self._sum(self.I_Gb, (idx[0][12], idx[1][12], idx[2][12], idx[3][12])),
                         self._sum(self.I_Gb, (idx[0][13], idx[1][13], idx[2][13], idx[3][13]))])
        gv = S_valid[10:14]
        g = np.clip(np.where(gv > 0, gsum / np.maximum(gv, 1.0), 0.0) / (0.5 * self.gscale), 0.0, 1.0)
        boundary = np.clip((t_in * (1.0 - t_out) + t_in * t_out * g - (1.0 - t_in) * t_out).mean(0), -1.0, 1.0)
        score = support - 2.0 * contra - 3.0 * occ_frac + bnd_w * boundary
        score[~ok] = -np.inf
        k = int(np.argmax(score))
        r_n, r_top = float(S_in[1, k] - n[k]), float(S_top[1, k] - n_top[k])
        ring_top = r_top / r_n if r_n > 0 else 0.0
        return {"offset": (float(offs[k, 0]), float(offs[k, 1])), "n": int(n[k]), "n_top": int(n_top[k]),
                "n_valid": int(n_valid[k]), "n_occ": int(n_occ[k]), "in_image": float(in_image[k]),
                "support": float(support[k]), "contra": float(contra[k]), "occ_frac": float(occ_frac[k]),
                "boundary": float(boundary[k]), "ring_top": ring_top, "score": float(score[k]),
                "ab": (float(a[k]), float(b[k]))}

    def occ_frac(self, ab, h1, h2):
        idx = self._idx(np.array([ab[0]]), np.array([ab[1]]), h1, h2)
        n = self._sum(self.I_in, idx)[0]
        return float(self._sum(self.I_occ, idx)[0] / n) if n > 0 else 0.0

    def add_occ(self, ab, h1, h2):
        ia0, ia1, ib0, ib1 = self._idx(np.array([ab[0]]), np.array([ab[1]]), h1, h2)
        self.occ[int(ib0[0]):int(ib1[0]), int(ia0[0]):int(ia1[0])] = 1
        self.I_occ = cv2.integral(self.occ)


def _cell_poly_px(M, center, e1, e2, h1, h2):
    corners = np.array([[-h1, -h2], [h1, -h2], [h1, h2], [-h1, h2]])
    mm = center + corners[:, :1] * e1 + corners[:, 1:] * e2
    return (np.c_[mm, np.ones(4)] @ M).astype(np.float32)


def top_layer_candidates(D, valid, bin_mm=10, roi_frac=0.25, k=4, min_rel=0.15):
    """ROI 깊이 히스토그램의 국소 최대 피크들(질량 내림차순, 정밀화된 깊이 mm). [0] = find_top_layer 결과."""
    hist, edges = _roi_depth_hist(D, valid, bin_mm, roi_frac)
    if len(hist) == 0:
        return []
    hp = np.pad(hist, 1)
    loc = np.nonzero((hist >= hp[:-2]) & (hist >= hp[2:]) & (hist >= min_rel * hist.max()))[0]
    loc = sorted(loc, key=lambda i: -hist[i])
    out = []
    for i in loc[:k]:
        d = _refine_peak(hist, edges, i)
        if all(abs(d - o) > 2 * bin_mm for o in out):
            out.append(d)
    return out


def _annotate_layer(sess, top_d, boxes_v1, tol_mm, conf_src):
    """v1 박스에 기하/신뢰도 부착. 반환 (top, valid, boxes, geoms, strong_idx)."""
    D, X, Y = sess["D"], sess["X"], sess["Y"]
    valid = valid_mask(sess)
    top = (np.abs(D - top_d) < tol_mm) & valid
    boxes, geoms = [], []
    for b in boxes_v1:
        g = _box_geometry(X, Y, D, valid, top, b)
        nb = dict(b)
        nb.update({"confidence": round(g["confidence"], 3), "source": "detected",
                   "center_mm": (round(g["center_mm"][0], 1), round(g["center_mm"][1], 1)),
                   "ang_deg": round(g["ang_deg"], 1), "plane_rms_mm": g["plane_rms_mm"],
                   "conf_components": {k: round(v, 3) for k, v in g["components"].items()}})
        boxes.append(nb)
        geoms.append(g)
    strong = [i for i, g in enumerate(geoms)
              if g["confidence"] >= conf_src and g["components"]["dims"] > 0.0 and g["has_center"]]
    return top, valid, boxes, geoms, strong


# 격자 이웃 제안: (e1 스텝, e2 스텝) — 1차 이웃(탐색 ±search_mm), 대각·2차 이웃(탐색 ±2*search_mm)
_NEIGHBOR_STEPS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1),
                   (2, 0), (-2, 0), (0, 2), (0, -2)]


def detect_boxes_v2(sess, tol_mm=40, min_area_px=700, conf_src=0.55, support_min=0.30,
                    contra_max=0.08, occ_max=0.10, valid_min=0.15, in_image_min=0.90,
                    ring_w=0.0, bnd_w=1.0, bnd_min=-0.25, search_mm=36.0, search_step_mm=12.0, max_rounds=2,
                    dense_scan=True, replace_frac=0.3, layer_fallback=True, fallback_min_strong=1,
                    fallback_k=8, conf_min=None, grid_res_mm=6.0, verbose=False, debug=None):
    """detect_boxes 래핑: (1) 박스별 신뢰도 (2) 격자 기반 결손 셀 보완('inferred').

    반환 (top_d, mask, boxes). boxes 의 각 dict 는 v1 키(area_px, dims_mm, depth_mm, rect_px)에
      confidence(0~1), source('detected'|'inferred'), center_mm, ang_deg, plane_rms_mm, conf_components 추가.
      - detected: v1 검출 그대로 (v1 키 값 불변). 단, 신뢰(strong) 조건(confidence >= conf_src 이고 치수가
        SKU 사전 ±15% 이내)에 못 미치는 '약한' 검출(결손으로 잘린 조각·병합체)이 채택된 격자 셀에 상면 픽셀의
        50% 이상 포함되면 그 셀로 대체된다(셀 dict 의 'replaces' 에 개수).
      - inferred: 신뢰 박스들로 격자(방향·치수·피치)를 추정하고, 각 신뢰 박스의 격자 이웃 셀
        (1차·대각·2차, 두 방향군) 및 팔레트 근방 약한 검출 위치의 셀을 후보로 삼아, 상면 지지도
        support(셀 내 상면 픽셀 비율, 무효 픽셀은 구멍으로 허용) >= support_min,
        모순 픽셀(유효하지만 상면 깊이 아님) 비율 <= contra_max, 기존 박스 겹침 <= occ_max,
        화면 내 비율 >= in_image_min 이면 채택. 셀 위치는 ±search_mm(원거리 후보 ±2*search_mm)에서
        지지도 최대 위치로 미세조정하고, (점수 - ring_w*경계띠 상면비율) 내림차순 그리디(겹침 억제)로 채택
        → 채택 셀은 다음 라운드의 소스가 된다(max_rounds).
      - layer_fallback: v1 상면층에 신뢰 박스가 하나도 없으면 ROI 히스토그램의 다른 피크 층(질량 상위 fallback_k개)을
        시도해 신뢰 박스 >= fallback_min_strong 인 층으로 교체 (박스가 1~2개 남아 팔레트 바닥/티어시트 피크가 이긴 경우.
        실측: 박스 1개 = ROI 질량 ~5% 로 팔레트 데크·바닥 피크들에 밀려 상위 4개 밖 → k=8 필요. 빈 크레이트 3세션·빈 팔레트·티어시트 프레임에서 오검출 0).
      - conf_min: 지정 시 그 미만 신뢰도의 박스(검출/추론 모두)를 출력에서 제외.
    """
    D, X, Y = sess["D"], sess["X"], sess["Y"]
    top_d, mask, boxes_v1 = detect_boxes(sess, tol_mm=tol_mm, min_area_px=min_area_px)
    top, valid, boxes, geoms, strong = _annotate_layer(sess, top_d, boxes_v1, tol_mm, conf_src)
    layer_switched = False
    if layer_fallback:
        # 층 선택: 히스토그램 질량이 가장 큰 피크가 항상 상면인 것은 아니다. 평평한 바닥·데크가
        # ROI 를 지배하면 질량 1위가 바닥이 된다(셀 트윈에서 발견: 바닥 4184mm 가 질량 1위이나
        # strong 1개, 실제 박스층 3254mm 는 strong 5개). 따라서 후보 피크들을 실제로 검출해 보고
        # **SKU 사전에 부합하는 신뢰(strong) 박스 수**가 최대인 층을 고른다(동수면 카메라에 가까운 층).
        # 기존 동작(strong==0 일 때만 재시도)은 엉뚱한 층이 그럴듯한 검출을 내면 재고하지 못했다.
        evaluated = [(top_d, mask, top, boxes, geoms, strong)]
        for td in top_layer_candidates(D, valid, k=fallback_k)[1:]:
            if any(abs(td - e[0]) < tol_mm for e in evaluated):
                continue
            _, mask2, bx2 = _detect_boxes_at(sess, td, tol_mm=tol_mm, min_area_px=min_area_px)
            top2, _, boxes2, geoms2, strong2 = _annotate_layer(sess, td, bx2, tol_mm, conf_src)
            if verbose:
                print("  layer try %.0f -> boxes=%d strong=%d" % (td, len(boxes2), len(strong2)))
            if debug is not None:
                debug.setdefault("fallback_tries", []).append((td, len(boxes2), len(strong2)))
            evaluated.append((td, mask2, top2, boxes2, geoms2, strong2))
        # 층 선택 규칙: 디팔레타이징에서 다음에 집을 층은 **카메라에 가장 가까운** 박스 층이다.
        # 질량 최대 피크를 그대로 쓰면 평평한 바닥/데크가 ROI 를 지배할 때 바닥을 상면으로 잡고
        # (셀 트윈에서 발견: 바닥 4184mm 질량 1위·strong 1개 vs 실제 박스층 3254mm·strong 5개),
        # 반대로 strong 최다를 쓰면 상층이 몇 개 안 남았을 때 가득 찬 아래층을 골라 버린다.
        # 따라서 'strong >= 2 인 층 중 최근접'을 우선하고, 없으면 'strong >= 1 중 최근접'으로 완화한다.
        pick = None
        for need in (2, max(int(fallback_min_strong), 1)):
            ok = [e for e in evaluated if len(e[5]) >= need]
            if ok:
                pick = min(ok, key=lambda e: e[0])
                break
        if pick is not None and pick[0] != top_d:
            top_d, mask, top, boxes, geoms, strong = pick
            layer_switched = True
    for b in boxes:
        b["layer_fallback"] = layer_switched

    def _finish(bxs):
        if conf_min is not None:
            bxs = [b for b in bxs if b["confidence"] >= conf_min]
        return top_d, mask, bxs

    if not strong:
        return _finish(boxes)
    M = _affine_px_from_mm(X, Y, top)
    if M is None:
        return _finish(boxes)
    lat = _estimate_lattice([geoms[i] for i in strong])
    e1, e2, ext = lat["e1"], lat["e2"], lat["ext"]
    if verbose:
        print("  lattice: theta=%.1f L=%.0f W=%.0f gap=%.0f pitch=%s strong=%d/%d" % (
            lat["theta_deg"], lat["L"], lat["W"], lat["gap"],
            {k: round(v) for k, v in lat["pitch"].items()}, len(strong), len(boxes)))

    occ = np.zeros(D.shape, bool)              # 신뢰 박스 점유 (약한 검출은 제외)
    for i in strong:
        occ |= geoms[i]["poly"]
    C_strong = np.array([geoms[i]["center_mm"] for i in strong])
    inten = np.log1p(np.clip(sess["I"], 0, None)).astype(np.float32)
    inten = cv2.GaussianBlur(cv2.normalize(inten, None, 0, 1, cv2.NORM_MINMAX), (3, 3), 0)
    inten[~valid] = 0.0
    G = _LatticeGrid(D.shape, M, C_strong.mean(0), e1, e2, top, valid, occ, inten, res=grid_res_mm)
    near_r = 1.6 * max(lat["pitch"].values())
    G.set_near([G.to_ab(c) for c in C_strong], near_r)
    steps = np.arange(-search_mm, search_mm + 1e-6, search_step_mm)
    offsets = [(a, b_) for a in steps for b_ in steps]
    steps_w = np.arange(-2 * search_mm, 2 * search_mm + 1e-6, search_step_mm)
    offsets_wide = [(a, b_) for a in steps_w for b_ in steps_w]
    steps_s = np.arange(-search_step_mm, search_step_mm + 1e-6, 0.5 * search_step_mm)
    offsets_small = [(a, b_) for a in steps_s for b_ in steps_s]
    weak = [i for i in range(len(boxes)) if i not in strong and geoms[i]["has_center"]
            and np.hypot(*(C_strong - np.array(geoms[i]["center_mm"])).T).min() <= near_r]
    dedupe_r = 0.35 * min(lat["L"], lat["W"])

    if debug is not None:
        debug.update({"lattice": lat, "M": M, "grid": G, "top": top, "valid": valid,
                      "strong": strong, "weak": weak, "geoms": geoms,
                      "offsets": offsets, "evals": [], "layer_switched": layer_switched})

    sources = [(np.array(geoms[i]["center_mm"]), lat["cls"][k]) for k, i in enumerate(strong)]
    weak_src = [np.array(geoms[i]["center_mm"]) for i in weak]
    accepted = []   # dict(center, cls, ev, poly, mask)
    for rnd in range(max_rounds):
        cands = []
        seen = [(a["center"], a["cls"]) for a in accepted]

        def _propose(cc, cls, offs):
            if np.hypot(*(C_strong - cc).T).min() > near_r:      # 팔레트(신뢰 박스 군) 근방만
                return
            if seen:
                sc = np.array([tc for tc, _ in seen])
                sk = np.array([k for _, k in seen])
                if np.any((sk == cls) & (np.hypot(sc[:, 0] - cc[0], sc[:, 1] - cc[1]) < dedupe_r)):
                    return
            seen.append((cc.copy(), cls))
            h1, h2 = 0.5 * ext[cls][0], 0.5 * ext[cls][1]
            ev = G.eval(cc, h1, h2, offs, in_image_min=in_image_min, bnd_w=bnd_w)
            if ev is None:
                return
            ok = (ev["occ_frac"] <= occ_max and ev["contra"] <= contra_max
                  and ev["support"] >= support_min and ev["n_valid"] / ev["n"] >= valid_min
                  and ev["boundary"] >= bnd_min)
            if verbose:
                print("    r%d cand c=(%.0f,%.0f) cls=%d -> off=%s sup=%.2f contra=%.2f occ=%.2f bnd=%.2f ring=%.2f img=%.2f %s" % (
                    rnd, cc[0], cc[1], cls, ev["offset"], ev["support"], ev["contra"],
                    ev["occ_frac"], ev["boundary"], ev["ring_top"], ev["in_image"], "ok" if ok else ""))
            if debug is not None:
                debug["evals"].append((cc.copy(), cls, ev, ok))
            if ok:
                cands.append((ev["score"] - ring_w * ev["ring_top"], cc, cls, ev, h1, h2))
        for (c0, cls0) in sources:
            for (i, j) in _NEIGHBOR_STEPS:
                for cls in (cls0, 1 - cls0):
                    cc = c0.copy()
                    for step, axis, ea in ((i, 0, e1), (j, 1, e2)):
                        if step == 0:
                            continue
                        shift = 0.5 * (ext[cls][axis] - ext[cls0][axis]) if cls != cls0 else 0.0
                        cc = cc + np.sign(step) * (abs(step) * lat["pitch"][(cls0, axis)] + shift) * ea
                    _propose(cc, cls, offsets if abs(i) + abs(j) == 1 else offsets_wide)
        if rnd == 0:
            for c0 in weak_src:
                for cls in (0, 1):
                    _propose(c0.copy(), cls, offsets_wide)
            if dense_scan:   # 격자 전역 매치드 필터: 팔레트 근방의 SKU 크기 상면 직사각형 (피치 추정과 무관)
                for cls in (0, 1):
                    h1, h2 = 0.5 * ext[cls][0], 0.5 * ext[cls][1]
                    for cc in G.dense_scan(h1, h2, support_min, contra_max, occ_max, in_image_min):
                        _propose(np.asarray(cc, float), cls, offsets_small)
        # 순위 내림차순 그리디 채택 (겹침 억제)
        new_sources = []
        for (rank, cc, cls, ev, h1, h2) in sorted(cands, key=lambda t: -t[0]):
            if G.occ_frac(ev["ab"], h1, h2) > occ_max:
                continue
            G.add_occ(ev["ab"], h1, h2)
            cfin = cc + ev["offset"][0] * e1 + ev["offset"][1] * e2
            poly = _cell_poly_px(M, cfin, e1, e2, h1, h2)
            cm = np.zeros(D.shape, np.uint8)
            cv2.fillPoly(cm, [poly.astype(np.int32)], 1)
            accepted.append({"center": cfin, "cls": cls, "ev": ev, "poly": poly, "mask": cm > 0})
            new_sources.append((cfin, cls))
        if not new_sources:
            break
        sources = new_sources

    # 약한 검출(조각·병합체) 중 채택 셀에 흡수된 것 제거 (상면 픽셀의 replace_frac 이상이 셀 안에 들면 대체)
    replaced = set()
    for a in accepted:
        a["replaces"] = 0
        for i in weak:
            tp = geoms[i]["top_px"]
            n_tp = int(tp.sum())
            if n_tp > 0 and i not in replaced and (tp & a["mask"]).sum() / n_tp >= replace_frac:
                replaced.add(i)
                a["replaces"] += 1
    out = [b for i, b in enumerate(boxes) if i not in replaced]
    for a in accepted:
        Lc, Wc = lat["L"], lat["W"]
        m = a["mask"] & top
        ev = a["ev"]
        rms = _plane_rms(X, Y, D, m)
        comp = {"fill": ev["n_valid"] / ev["n"], "plane": _plane_score(rms),
                "rect": ev["support"], "dims": _dims_score(Lc, Wc)}
        conf = float(np.mean(list(comp.values())))
        out.append({
            "area_px": int(m.sum()),
            "dims_mm": (round(Lc, 1), round(Wc, 1)),
            "depth_mm": round(float(np.median(D[m])), 1) if m.any() else round(float(top_d), 1),
            "rect_px": cv2.minAreaRect(a["poly"]),
            "confidence": round(conf, 3), "source": "inferred",
            "center_mm": (round(float(a["center"][0]), 1), round(float(a["center"][1]), 1)),
            "ang_deg": round((lat["theta_deg"] + (0.0 if a["cls"] == 0 else 90.0)) % 180.0, 1),
            "plane_rms_mm": rms,
            "conf_components": {k: round(v, 3) for k, v in comp.items()},
            "support": round(ev["support"], 3), "ring_top": round(ev["ring_top"], 3),
            "boundary": round(ev["boundary"], 3),
            "replaces": a["replaces"], "layer_fallback": layer_switched,
        })
    return _finish(out)


def render_overlay_v2(sess, top_d, mask, boxes, out_path):
    """v2 오버레이: detected=초록, inferred=주황, 박스별 신뢰도 표기."""
    D = sess["D"]
    valid = valid_mask(sess)
    lo, hi = np.percentile(D[valid], [2, 98]) if valid.any() else (0, 1)
    norm = np.clip((D - lo) / max(hi - lo, 1), 0, 1)
    img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_BONE)
    img[~valid] = (20, 20, 20)
    img[mask] = (0.55 * img[mask] + 0.45 * np.array([0, 140, 255])).astype(np.uint8)
    n_inf = 0
    for i, b in enumerate(boxes):
        pts = cv2.boxPoints(b["rect_px"]).astype(np.int32)
        inferred = b.get("source") == "inferred"
        n_inf += int(inferred)
        color = (0, 165, 255) if inferred else (0, 255, 0)
        cv2.polylines(img, [pts], True, color, 2)
        cx, cy = pts.mean(axis=0).astype(int)
        cv2.putText(img, f'{i+1}', (cx - 8, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(img, f'{b.get("confidence", 0):.2f}', (cx - 14, cy + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
    cv2.putText(img, f'top @ {top_d:.0f}mm, {len(boxes)} boxes ({n_inf} inferred)',
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imwrite(str(out_path), img)


if __name__ == "__main__":
    import os
    try:
        from local_paths import BINPICK_DIR
    except ImportError:
        BINPICK_DIR = os.environ["BINPICK_DIR"]
    root = Path(BINPICK_DIR)
    out = Path(r"E:\Robot_Sim\explore\topface")
    out.mkdir(parents=True, exist_ok=True)
    sessions = sorted([d for d in root.iterdir() if d.is_dir()])
    for s in sessions:
        sess = load_session(s)
        top_d, mask, boxes = detect_boxes(sess)
        render_overlay(sess, top_d, mask, boxes, out / f"{s.name}_topface.png")
        dims = ", ".join(f'{b["dims_mm"][0]:.0f}x{b["dims_mm"][1]:.0f}' for b in boxes[:6])
        print(f"{s.name}: top@{top_d:.0f}mm boxes={len(boxes)} dims(mm)=[{dims}{'...' if len(boxes)>6 else ''}]")
