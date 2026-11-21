"""
classify_samples.py — 用 SIFT 检测器把抓来的图自动分成「含校徽 / 不含校徽」。

这一步本身就是个展示：检测器先跑一遍，人工只需抽查几张。
"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

import cv2

from detector import BadgeDetector

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "samples" / "raw"
POS = ROOT / "data" / "samples" / "with_badge"
NEG = ROOT / "data" / "samples" / "without_badge"
MAX_SIDE = 1100


def main() -> None:
    for d in (POS, NEG):
        d.mkdir(parents=True, exist_ok=True)
    det = BadgeDetector()
    rows = []
    files = sorted([p for p in RAW.glob("*") if p.suffix.lower() in
                    {".jpg", ".jpeg", ".png"}])
    print(f"待分类 {len(files)} 张")
    for p in files:
        img = cv2.imdecode(__import__("numpy").fromfile(str(p), dtype="uint8"),
                           cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        s = MAX_SIDE / max(h, w)
        small = cv2.resize(img, None, fx=s, fy=s) if s < 1 else img
        hit = det.detect(small)
        tag = "with_badge" if hit.found else "without_badge"
        dst = (POS if hit.found else NEG) / (p.stem + ".jpg")
        cv2.imwrite(str(dst), small, [cv2.IMWRITE_JPEG_QUALITY, 97])
        rows.append({"file": p.name, "saved": str(dst.name), "label": tag,
                     "inliers": hit.inliers, "score": round(hit.score, 1),
                     "box": list(map(int, hit.box)) if hit.box else "",
                     "ms": round(hit.elapsed_ms)})
        print(f"  {tag:14s} inl={hit.inliers:<4d} {p.name[:44]}")

    with open(ROOT / "data" / "samples" / "labels.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n_pos = sum(r["label"] == "with_badge" for r in rows)
    print(f"\n含校徽 {n_pos} 张 -> {POS}")
    print(f"不含校徽 {len(rows)-n_pos} 张 -> {NEG}")


if __name__ == "__main__":
    main()
