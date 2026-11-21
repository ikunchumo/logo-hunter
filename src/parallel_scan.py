"""
parallel_scan.py — 多进程并行扫描。SIFT 是单线程的，32 核只跑 1 核太浪费。

用法
----
  from parallel_scan import scan_paths
  results = scan_paths(files, workers=12)

每个子进程只初始化一次 BadgeDetector（模板 SIFT 特征算一遍），
之后按文件流式处理，结果顺序与输入一致。
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from detector import BadgeDetector

_DET: BadgeDetector | None = None
_TPL: str | None = None


def _init(template_path: str | None) -> None:
    # OpenCV 默认在子进程里也会开多线程，导致 SIFT 的浮点累加顺序不同，
    # 边缘样本会在阈值附近抖动（内点 12±1）。锁成单线程换取可复现。
    import cv2
    cv2.setNumThreads(1)
    global _DET, _TPL
    _TPL = template_path
    _DET = BadgeDetector(template_path) if template_path else BadgeDetector()


def _work(path: str) -> dict:
    assert _DET is not None
    h = _DET.detect_file(path)
    return {
        "file": path,
        "name": Path(path).name,
        "found": bool(h.found),
        "score": round(h.score, 2),
        "inliers": int(h.inliers),
        "matches": int(h.good_matches),
        "ms": round(h.elapsed_ms, 1),
        "box": [int(v) for v in h.box] if h.box else None,
    }


def scan_paths(paths: list[str | Path], workers: int | None = None,
               template_path: str | None = None,
               progress=None) -> list[dict]:
    """并行扫一批文件。workers=None 时按 CPU 核数自动定（上限 16）。"""
    files = [str(p) for p in paths]
    if not files:
        return []
    n = workers or min(16, max(1, (os.cpu_count() or 4) - 2))

    if n <= 1:
        det = BadgeDetector(template_path) if template_path else BadgeDetector()
        out = []
        for f in files:
            h = det.detect_file(f)
            out.append({"file": f, "name": Path(f).name, "found": bool(h.found),
                        "score": round(h.score, 2), "inliers": int(h.inliers),
                        "matches": int(h.good_matches),
                        "ms": round(h.elapsed_ms, 1),
                        "box": [int(v) for v in h.box] if h.box else None})
            if progress:
                progress(len(out), len(files))
        return out

    out: list[dict] = []
    with ProcessPoolExecutor(max_workers=n, initializer=_init,
                             initargs=(template_path,)) as ex:
        for r in ex.map(_work, files, chunksize=1):
            out.append(r)
            if progress:
                progress(len(out), len(files))
    return out


if __name__ == "__main__":
    import sys
    import time
    from pathlib import Path

    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "data/samples")
    files = sorted([p for p in folder.rglob("*")
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])[:40]
    for w in (1, 4, 8, 16):
        t0 = time.time()
        r = scan_paths(files, workers=w)
        dt = time.time() - t0
        hits = sum(x["found"] for x in r)
        print(f"workers={w:2d}  {dt:6.2f}s  {dt/len(files)*1000:6.0f} ms/张  "
              f"命中 {hits}")
