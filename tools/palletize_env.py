# -*- coding: utf-8 -*-
"""팔레타이징 배치 계획 RL 환경 (T6).

실측 ToF 데이터(E:/Robot_Sim/explore/topface/results.json) 기반:
  - 박스 상면 L = 293.0 +/- 9.2 mm, W = 218.8 +/- 9.9 mm
  - 박스 높이 H = 283 mm (층간 깊이차 실측, 층내 깊이 산포 ~2 mm)
프레임워크 무관 순수 numpy 구현 (gymnasium 불필요).
좌표 규약: heightmap[ix, iy] — 첫 축이 x, 둘째 축이 y (좌하단 원점).
"""

import json
import os

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# 실측 통계 (explore/topface/results.json)
BOX_L_MU, BOX_L_SIGMA = 293.0, 9.2
BOX_W_MU, BOX_W_SIGMA = 218.8, 9.9
BOX_H_MU, BOX_H_SIGMA = 283.0, 2.0

PALLET_MM = 1100.0          # T-11 표준 팔레트
CELL_MM = 25.0
GRID = int(PALLET_MM / CELL_MM)  # 44


class PalletizeEnv:
    """팔레트 위 박스 배치 환경.

    상태: {"heightmap": (44,44) float mm, "box": (L,W,H) mm}
    액션: (grid_x, grid_y, rot) — 박스 좌하단 셀 위치, rot 0/1(=90도)
          평탄화 정수(shape (44,44,2) 기준 ravel)도 허용.
    """

    def __init__(self, max_height=1500.0, max_boxes=30,
                 support_ratio=0.7, support_tol=10.0, seed=None):
        self.max_height = float(max_height)
        self.max_boxes = int(max_boxes)
        self.support_ratio = float(support_ratio)
        self.support_tol = float(support_tol)
        self.rng = np.random.default_rng(seed)
        self.heightmap = np.zeros((GRID, GRID), dtype=np.float64)
        self.box = None
        self.n_placed = 0
        self.placed_volume = 0.0
        self.placements = []

    # ------------------------------------------------------------------
    def _sample_box(self):
        L = self.rng.normal(BOX_L_MU, BOX_L_SIGMA)
        W = self.rng.normal(BOX_W_MU, BOX_W_SIGMA)
        H = self.rng.normal(BOX_H_MU, BOX_H_SIGMA)
        return np.array([max(L, 50.0), max(W, 50.0), max(H, 50.0)])

    def _obs(self):
        return {"heightmap": self.heightmap.copy(), "box": self.box.copy()}

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.heightmap[:] = 0.0
        self.n_placed = 0
        self.placed_volume = 0.0
        self.placements = []
        self.box = self._sample_box()
        return self._obs()

    # ------------------------------------------------------------------
    def _footprint_cells(self, rot):
        L, W, _ = self.box
        a, b = (L, W) if rot == 0 else (W, L)
        return int(np.ceil(a / CELL_MM)), int(np.ceil(b / CELL_MM))

    def _placement_maps(self, rot):
        """rot에 대해 (지지면 최고높이, 유효 마스크) 44x44 배열 반환.

        범위 밖 좌하단 위치는 유효 False, 높이 inf.
        """
        nx, ny = self._footprint_cells(rot)
        hmax = np.full((GRID, GRID), np.inf)
        valid = np.zeros((GRID, GRID), dtype=bool)
        if nx > GRID or ny > GRID:
            return hmax, valid
        win = sliding_window_view(self.heightmap, (nx, ny))
        wmax = win.max(axis=(2, 3))
        support = (wmax[:, :, None, None] - win <= self.support_tol)
        ratio = support.mean(axis=(2, 3))
        ok = (ratio >= self.support_ratio) & \
             (wmax + self.box[2] <= self.max_height)
        hmax[: GRID - nx + 1, : GRID - ny + 1] = wmax
        valid[: GRID - nx + 1, : GRID - ny + 1] = ok
        return hmax, valid

    def action_mask(self):
        """모든 (grid_x, grid_y, rot)의 유효 여부 (44,44,2) bool 배열."""
        mask = np.zeros((GRID, GRID, 2), dtype=bool)
        for rot in (0, 1):
            _, valid = self._placement_maps(rot)
            mask[:, :, rot] = valid
        return mask

    # ------------------------------------------------------------------
    def step(self, action):
        if np.isscalar(action):
            gx, gy, rot = np.unravel_index(int(action), (GRID, GRID, 2))
        else:
            gx, gy, rot = (int(v) for v in action)

        hmax, valid = self._placement_maps(rot)
        if not (0 <= gx < GRID and 0 <= gy < GRID) or not valid[gx, gy]:
            return self._obs(), -0.1, False, {"placed": False,
                                              "n_placed": self.n_placed}

        L, W, H = self.box
        nx, ny = self._footprint_cells(rot)
        top = hmax[gx, gy] + H
        self.heightmap[gx:gx + nx, gy:gy + ny] = np.maximum(
            self.heightmap[gx:gx + nx, gy:gy + ny], top)
        self.placed_volume += L * W * H
        self.placements.append((gx, gy, int(rot), float(top)))
        self.n_placed += 1
        reward = (L * W * H) / (PALLET_MM * PALLET_MM * self.max_height)

        terminated = self.n_placed >= self.max_boxes
        if not terminated:
            self.box = self._sample_box()
            if not self.action_mask().any():
                terminated = True
        info = {"placed": True, "n_placed": self.n_placed,
                "utilization": self.utilization()}
        return self._obs(), reward, terminated, info

    # ------------------------------------------------------------------
    def utilization(self):
        """배치 부피 / (팔레트 면적 x 현재 최대 적재 높이)."""
        top = self.heightmap.max()
        if top <= 0:
            return 0.0
        return self.placed_volume / (PALLET_MM * PALLET_MM * top)

    def utilization_cap(self):
        """배치 부피 / (팔레트 면적 x max_height)."""
        return self.placed_volume / (PALLET_MM * PALLET_MM * self.max_height)

    def n_layers(self):
        return self.heightmap.max() / BOX_H_MU

    def render(self, path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.4, 5.6))
        im = ax.imshow(self.heightmap.T, origin="lower", cmap="viridis",
                       extent=[0, PALLET_MM, 0, PALLET_MM],
                       vmin=0, vmax=max(self.heightmap.max(), 1.0))
        fig.colorbar(im, ax=ax, label="height (mm)")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title("boxes=%d  utilization=%.1f%%  layers=%.1f"
                     % (self.n_placed, 100.0 * self.utilization(),
                        self.n_layers()))
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)


# ----------------------------------------------------------------------
def heuristic_dbl(env):
    """Deepest-bottom-left: 지지면이 가장 낮은 곳, 동률이면 (y, x, rot) 최소.

    유효 액션이 없으면 None.
    """
    best = None
    best_key = None
    for rot in (0, 1):
        hmax, valid = env._placement_maps(rot)
        if not valid.any():
            continue
        h = np.where(valid, hmax, np.inf)
        idx = np.argwhere(h == h.min())
        # 좌하단 우선: y 작은 순 -> x 작은 순
        gy, gx = min((int(i[1]), int(i[0])) for i in idx)
        key = (float(h.min()), gy, gx, rot)
        if best_key is None or key < best_key:
            best_key = key
            best = (gx, gy, rot)
    return best


def run_episode(env, seed):
    env.reset(seed=seed)
    total_r = 0.0
    while True:
        action = heuristic_dbl(env)
        if action is None:
            break
        _, r, done, _ = env.step(action)
        total_r += r
        if done:
            break
    return {"seed": seed,
            "utilization": env.utilization(),
            "utilization_cap": env.utilization_cap(),
            "boxes": env.n_placed,
            "layers": env.n_layers(),
            "reward": total_r}


# ----------------------------------------------------------------------
if __name__ == "__main__":
    out_dir = r"E:\Robot_Sim\explore\palletize"
    os.makedirs(out_dir, exist_ok=True)

    env = PalletizeEnv()
    episodes = [run_episode(env, seed) for seed in range(100)]

    util = np.array([e["utilization"] for e in episodes])
    util_cap = np.array([e["utilization_cap"] for e in episodes])
    boxes = np.array([e["boxes"] for e in episodes], dtype=float)
    layers = np.array([e["layers"] for e in episodes])

    stats = {
        "policy": "heuristic_dbl (deepest-bottom-left)",
        "n_episodes": len(episodes),
        "pallet_mm": PALLET_MM,
        "grid": GRID,
        "cell_mm": CELL_MM,
        "max_height_mm": env.max_height,
        "max_boxes": env.max_boxes,
        "support_ratio_min": env.support_ratio,
        "box_stats_mm": {"L": [BOX_L_MU, BOX_L_SIGMA],
                         "W": [BOX_W_MU, BOX_W_SIGMA],
                         "H": [BOX_H_MU, BOX_H_SIGMA]},
        "utilization_mean": float(util.mean()),
        "utilization_std": float(util.std()),
        "utilization_cap_mean": float(util_cap.mean()),
        "utilization_cap_std": float(util_cap.std()),
        "boxes_mean": float(boxes.mean()),
        "boxes_std": float(boxes.std()),
        "layers_mean": float(layers.mean()),
        "layers_std": float(layers.std()),
    }
    stats_path = os.path.join(out_dir, "baseline_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 대표 에피소드: 활용률 최저/중앙/최고 시드를 재실행해 렌더
    order = np.argsort(util)
    picks = {"low": int(order[0]),
             "mid": int(order[len(order) // 2]),
             "high": int(order[-1])}
    for name, seed in picks.items():
        run_episode(env, seed)
        env.render(os.path.join(out_dir, "ep_%s_seed%03d.png" % (name, seed)))

    print("baseline: heuristic_dbl, %d episodes" % len(episodes))
    print("utilization (vs stack envelope): %.3f +/- %.3f"
          % (stats["utilization_mean"], stats["utilization_std"]))
    print("utilization (vs 1500mm cap)   : %.3f +/- %.3f"
          % (stats["utilization_cap_mean"], stats["utilization_cap_std"]))
    print("boxes placed: %.2f +/- %.2f" % (stats["boxes_mean"],
                                           stats["boxes_std"]))
    print("layers      : %.2f +/- %.2f" % (stats["layers_mean"],
                                           stats["layers_std"]))
    print("saved: %s" % stats_path)
    for name, seed in picks.items():
        print("saved: ep_%s_seed%03d.png" % (name, seed))
