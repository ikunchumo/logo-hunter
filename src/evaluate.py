"""
evaluate.py — 在测试集上量检测器：召回、误检、IoU、耗时，并画可视化。

  python src/evaluate.py                 # 默认阈值
  python src/evaluate.py --min-inliers 7 --ratio 0.72   # 调参
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from detector import BadgeDetector

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "data" / "testset"
DOCS = ROOT / "docs"


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, default=0.72)
    ap.add_argument("--min-inliers", type=int, default=12)
    ap.add_argument("--min-inlier-ratio", type=float, default=0.26)
    ap.add_argument("--scales", default="1.0,1.7,2.6")
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--iou", type=float, default=0.25)
    args = ap.parse_args()

    det = BadgeDetector(ratio=args.ratio, min_inliers=args.min_inliers,
                        min_inlier_ratio=args.min_inlier_ratio,
                        scales=tuple(float(s) for s in args.scales.split(",")),
                        max_side=args.max_side)
    gt = json.loads((TEST / "gt.json").read_text(encoding="utf-8"))

    stats = {"pos": [], "neg": []}
    for kind in ("pos", "neg"):
        for item in gt[kind]:
            p = TEST / item["file"]
            hit = det.detect_file(p)
            rec = {"file": item["file"], "found": hit.found, "score": hit.score,
                   "inliers": hit.inliers, "matches": hit.good_matches,
                   "ms": hit.elapsed_ms, "box": hit.box}
            if kind == "pos":
                rec["iou"] = iou(hit.box, item["box"]) if hit.box else 0.0
                rec["hit_ok"] = bool(hit.found and rec["iou"] >= args.iou)
            stats[kind].append(rec)

    pos = stats["pos"]
    neg = stats["neg"]
    tp = sum(r["hit_ok"] for r in pos)
    miss = len(pos) - tp
    fp = sum(r["found"] for r in neg)
    tn = len(neg) - fp
    recall = tp / max(len(pos), 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    avg_ms = float(np.mean([r["ms"] for r in pos + neg]))

    print(f"\n阈值 ratio={args.ratio} min_inliers={args.min_inliers} "
          f"min_inlier_ratio={args.min_inlier_ratio} scales={args.scales}")
    print(f"  正样本 {tp}/{len(pos)} 检出（IoU≥{args.iou}），漏检 {miss}")
    print(f"  负样本 {fp}/{len(neg)} 误检")
    print(f"  召回 {recall*100:.1f}%   精确率 {precision*100:.1f}%   F1 {f1*100:.1f}%")
    print(f"  平均耗时 {avg_ms:.0f} ms/张")

    # 漏检 / 误检明细
    for r in pos:
        if not r["hit_ok"]:
            print(f"    [漏] {r['file']}  found={r['found']} iou={r['iou']:.2f} "
                  f"inl={r['inliers']} score={r['score']:.1f}")
    for r in neg:
        if r["found"]:
            print(f"    [误] {r['file']}  inl={r['inliers']} score={r['score']:.1f}")

    # 可视化
    DOCS.mkdir(exist_ok=True)
    tiles = []
    for kind, items in (("pos", pos), ("neg", neg)):
        for r in items[:14]:
            img = cv2.imread(str(TEST / r["file"]))
            if img is None:
                continue
            h, w = img.shape[:2]
            s = 220 / max(h, w)
            img = cv2.resize(img, (int(w * s), int(h * s)))
            if r["box"]:
                x, y, bw, bh = [int(v * s) for v in r["box"]]
                if kind == "pos":
                    col = (0, 220, 0) if r.get("hit_ok") else (0, 165, 255)
                else:
                    col = (0, 0, 230)
                cv2.rectangle(img, (x, y), (x + bw, y + bh), col, 2)
            if kind == "pos":
                label = "OK" if r.get("hit_ok") else "MISS"
            else:
                label = "FALSE+" if r["found"] else "TN"
            color = (0, 200, 0) if label in ("OK", "TN") else (0, 0, 230)
            cv2.putText(img, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(img)

    if tiles:
        cols = 7
        rows = (len(tiles) + cols - 1) // cols
        hh = max(t.shape[0] for t in tiles)
        ww = max(t.shape[1] for t in tiles)
        grid = np.full((rows * hh, cols * ww, 3), 24, np.uint8)
        for i, t in enumerate(tiles):
            r, c = divmod(i, cols)
            grid[r * hh:r * hh + t.shape[0], c * ww:c * ww + t.shape[1]] = t
        cv2.imwrite(str(DOCS / "detection_grid.png"), grid)
        print("  -> docs/detection_grid.png")

    (ROOT / "data" / "eval_last.json").write_text(json.dumps(
        {"args": vars(args), "recall": recall, "precision": precision, "f1": f1,
         "tp": tp, "fp": fp, "miss": miss, "avg_ms": avg_ms, "stats": stats},
        ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
