"""
scan.py — 给一个文件夹，把里面所有图片过一遍，挑出「含目标」的。

命令行
------
  python src/scan.py D:\\photos
  python src/scan.py D:\\photos --recursive --out results
  python src/scan.py D:\\photos --crop          # 顺便把目标裁出来

产物
----
  results/hits.csv            命中清单（文件、置信度、内点、耗时、框）
  results/annotated/*.jpg     标了框的图（命中的画绿框，未命中的不输出）
  results/crops/*.png         --crop 时裁出的目标

代码里
------
  from scan import scan_folder
  res = scan_folder("D:/photos")
"""
from __future__ import annotations

import argparse
import os
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from detector import BadgeDetector

ROOT = Path(__file__).resolve().parents[1]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def iter_images(folder: Path, recursive: bool):
    it = folder.rglob("*") if recursive else folder.glob("*")
    for p in sorted(it):
        if p.is_file() and p.suffix.lower() in EXTS:
            yield p


def scan_folder(folder: str | Path, recursive: bool = True,
                det: BadgeDetector | None = None, verbose: bool = True,
                workers: int | None = None) -> list[dict]:
    """并行扫描一个文件夹。workers=None 时按 CPU 核数自动定（上限 16）。"""
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(folder)
    files = list(iter_images(folder, recursive))
    n_w = workers or min(16, max(1, (os.cpu_count() or 4) - 2))
    if verbose:
        print(f"扫描 {folder}  共 {len(files)} 张图  （{n_w} 进程并行）", flush=True)
    t0 = time.time()

    from parallel_scan import scan_paths

    def prog(i, n):
        if verbose and (i % 10 == 0 or i == n):
            print(f"  {i}/{n}", flush=True)

    out = scan_paths(files, workers=n_w, progress=prog if verbose else None)

    if verbose:
        for i, r in enumerate(out, 1):
            flag = "✓ 命中" if r["found"] else "  --  "
            print(f"  [{i:>3}/{len(out)}] {flag}  {r['name'][:44]:46s} "
                  f"inl={r['inliers']:<4d} {r['ms']:6.0f}ms", flush=True)
        hits = sum(r["found"] for r in out)
        el = time.time() - t0
        print(f"\n完成：{hits}/{len(out)} 张命中，总耗时 {el:.1f}s "
              f"（平均 {el/max(len(out),1)*1000:.0f} ms/张，"
              f"单线程约 {sum(r['ms'] for r in out)/max(len(out),1):.0f} ms/张）")
    return out


def save_outputs(results: list[dict], src_root: Path, out_dir: Path,
                 crop: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ann_dir = out_dir / "annotated"
    ann_dir.mkdir(exist_ok=True)
    if crop:
        (out_dir / "crops").mkdir(exist_ok=True)

    with open(out_dir / "hits.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file", "found", "score", "inliers", "matches", "ms", "box"])
        for r in results:
            w.writerow([r["file"], r["found"], r["score"], r["inliers"],
                        r["matches"], r["ms"], r["box"]])

    (out_dir / "hits.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    n = 0
    for r in results:
        if not r["found"]:
            continue
        img = cv2.imread(r["file"], cv2.IMREAD_COLOR)
        if img is None:
            continue
        x, y, bw, bh = r["box"]
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 210, 0), 3)
        label = f"target  score={r['score']:.0f}"
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x, max(0, y - 26)), (x + tw + 10, y), (0, 210, 0), -1)
        cv2.putText(img, label, (x + 5, max(18, y - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        rel = Path(r["file"]).relative_to(src_root) if \
            str(r["file"]).startswith(str(src_root)) else Path(r["name"])
        dst = ann_dir / (str(rel).replace("\\", "_").replace("/", "_"))
        cv2.imwrite(str(dst.with_suffix(".jpg")), img)
        if crop:
            pad = int(0.06 * max(bw, bh))
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(img.shape[1], x + bw + pad), min(img.shape[0], y + bh + pad)
            cv2.imwrite(str(out_dir / "crops" / (dst.stem + "_badge.png")),
                        img[y0:y1, x0:x1])
        n += 1
    print(f"标注图 {n} 张 -> {ann_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--recursive", action="store_true", default=True)
    ap.add_argument("--no-recursive", dest="recursive", action="store_false")
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--crop", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--workers", type=int, default=0, help="并行进程数，0=自动")
    args = ap.parse_args()

    src = Path(args.folder).resolve()
    res = scan_folder(src, args.recursive, verbose=not args.quiet,
                      workers=args.workers or None)
    save_outputs(res, src, Path(args.out), crop=args.crop)


if __name__ == "__main__":
    main()
