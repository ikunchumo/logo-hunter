"""
sweep.py — 参数网格搜索，找召回/精确率平衡最好的那一组。
图像只读一次，缓存在内存里。
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import cv2
import numpy as np

from detector import BadgeDetector

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "data" / "testset"


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def main() -> None:
    gt = json.loads((TEST / "gt.json").read_text(encoding="utf-8"))
    cache = {}
    for kind in ("pos", "neg"):
        for it in gt[kind]:
            cache[it["file"]] = cv2.imread(str(TEST / it["file"]))

    grid = list(itertools.product(
        [0.72, 0.75, 0.78],            # ratio
        [9, 11, 14],                   # min_inliers
        [0.18, 0.24, 0.30],            # min_inlier_ratio
    ))
    rows = []
    for ratio, mi, mir in grid:
        det = BadgeDetector(ratio=ratio, min_inliers=mi, min_inlier_ratio=mir)
        tp = fp = 0
        for it in gt["pos"]:
            h = det.detect(cache[it["file"]])
            if h.found and h.box and iou(h.box, it["box"]) >= 0.25:
                tp += 1
        for it in gt["neg"]:
            if det.detect(cache[it["file"]]).found:
                fp += 1
        rec = tp / len(gt["pos"])
        prec = tp / max(tp + fp, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        rows.append((f1, rec, prec, ratio, mi, mir, tp, fp))
        print(f"ratio={ratio} inl>={mi:2d} ir>={mir:.2f}  "
              f"recall={rec*100:5.1f}%  prec={prec*100:5.1f}%  F1={f1*100:5.1f}% "
              f"(tp={tp}, fp={fp})", flush=True)

    rows.sort(reverse=True)
    best = rows[0]
    print("\n最佳：F1 %.1f%%  ratio=%s min_inliers=%d min_inlier_ratio=%.2f"
          % (best[0] * 100, best[3], best[4], best[5]))
    (ROOT / "data" / "sweep.json").write_text(json.dumps(
        [{"f1": r[0], "recall": r[1], "precision": r[2], "ratio": r[3],
          "min_inliers": r[4], "min_inlier_ratio": r[5], "tp": r[6], "fp": r[7]}
         for r in rows], ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
