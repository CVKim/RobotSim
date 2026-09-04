# -*- coding: utf-8 -*-
"""CLI: python -m robotsim_perception run <session_dir> --json out.json --overlay out.png

콘솔 출력은 ASCII 만 사용 (Windows cp949 콘솔 호환).
"""
from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    from .detect import DEFAULT_SKU
    from .planner import DEFAULT_COL_TOL_MM, DEFAULT_DEST_XY_MM

    p = argparse.ArgumentParser(prog="python -m robotsim_perception",
                                description="ToF bin-picking perception: top layer -> boxes -> pick plan")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the pipeline on one session directory (.mim X/Y/D/I)")
    r.add_argument("session_dir")
    r.add_argument("--json", dest="json_path", default=None, help="write result JSON here")
    r.add_argument("--overlay", dest="overlay_path", default=None, help="write overlay PNG here")
    r.add_argument("--sku", type=float, nargs=3, metavar=("L", "W", "H"), default=list(DEFAULT_SKU),
                   help="box SKU L W H in mm (default %(default)s)")
    r.add_argument("--sku-tol", type=float, default=None,
                   help="drop boxes whose L/W deviate more than this fraction from SKU (default: off)")
    r.add_argument("--dest", type=float, nargs=2, metavar=("X", "Y"), default=list(DEFAULT_DEST_XY_MM),
                   help="destination stack center X Y in camera mm, negatives allowed (default %(default)s)")
    r.add_argument("--col-tol", type=float, default=DEFAULT_COL_TOL_MM, help="column tolerance mm (default %(default)s)")
    r.add_argument("--tol", type=float, default=40.0, help="top-layer depth tolerance mm (default %(default)s)")
    r.add_argument("--min-area", type=int, default=700, help="min component area px (default %(default)s)")
    r.add_argument("--min-confidence", type=float, default=0.0, help="drop boxes below this confidence")
    r.add_argument("--lattice", action="store_true",
                   help="v2: lattice gap completion + layer selection "
                        "(30 real frames: 152 -> 167 boxes; ~52 -> ~340 ms/frame)")
    r.add_argument("--quiet", action="store_true", help="suppress summary on stdout")

    sub.add_parser("version", help="print package and schema version")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "version":
        from . import __version__
        from .pipeline import SCHEMA_VERSION
        print(f"robotsim_perception {__version__} (schema {SCHEMA_VERSION})")
        return 0

    from .pipeline import run_session
    from .render import render_overlay

    sku = tuple(float(v) for v in args.sku)
    dest = tuple(float(v) for v in args.dest)
    try:
        res = run_session(args.session_dir, sku=sku, dest_xy_mm=dest, col_tol_mm=args.col_tol,
                          tol_mm=args.tol, min_area_px=args.min_area, sku_tol=args.sku_tol,
                          min_confidence=args.min_confidence, lattice=args.lattice)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.json_path:
        res.to_json(args.json_path)
    if args.overlay_path:
        render_overlay(res.frame, res, args.overlay_path)

    if not args.quiet:
        top = res.top_layer
        top_txt = f"{top.depth_mm:.1f} mm" if top.ok else "not found"
        print(f"session   : {args.session_dir}")
        print(f"valid px  : {res.frame.n_valid} ({100 * res.frame.valid.mean():.1f}%)")
        print(f"top layer : {top_txt}")
        print(f"boxes     : {len(res.boxes)}")
        for b in res.boxes:
            print(f"  [{b.id}] L{b.dims_mm[0]:.0f} x W{b.dims_mm[1]:.0f} mm  z={b.depth_mm:.0f}  "
                  f"tilt={b.tilt_deg:.1f}deg  conf={b.confidence:.2f}  {b.source:8s} "
                  f"center=({b.center_mm[0]:.0f},{b.center_mm[1]:.0f})")
        if res.plan:
            seq = " -> ".join(str(s.box_id) for s in res.plan)
            print(f"pick plan : {seq}  (first dist {res.plan[0].dist_mm:.0f} mm to dest)")
        bd = res.latency_breakdown_ms
        print(f"latency   : {res.latency_ms:.1f} ms  (load {bd.get('load', 0):.1f}, top {bd['top_layer']:.1f}, "
              f"detect {bd['detect_boxes']:.1f}, plan {bd['pick_plan']:.2f})")
        if args.json_path:
            print(f"json      : {args.json_path}")
        if args.overlay_path:
            print(f"overlay   : {args.overlay_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
