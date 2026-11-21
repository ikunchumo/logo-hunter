"""
make_testset.py — 造一个「图库里混着各种图片」的测试集，用来量检测器准不准。

正样本：把校徽贴到校园照、纯色、纹理背景上，随机
        缩放（背景宽的 12%~55%）、旋转 ±35°、透视、亮度对比度、模糊、JPEG 压缩
负样本：校园风景、校庆标识、标准字、随机纹理、以及其他图形

产物：data/testset/{pos,neg}/*.jpg  +  data/testset/gt.json
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
OUT = ROOT / "data" / "testset"

BADGE = ASSETS / "nmu_badge_mask.png"


def load_backgrounds() -> list[np.ndarray]:
    """背景池：真实照片 + 合成背景。"""
    pool = []
    for name in ("candidate_4.jpg", "candidate_2.jpg", "candidate_3.jpg"):
        p = ASSETS / name
        if p.exists():
            im = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if im is not None:
                pool.append(im)
    rng = np.random.default_rng(3)
    # 渐变
    for _ in range(3):
        h, w = 640, 900
        c1 = rng.integers(20, 240, 3)
        c2 = rng.integers(20, 240, 3)
        t = np.linspace(0, 1, w)[None, :, None]
        pool.append(np.clip(c1 * (1 - t) + c2 * t, 0, 255).astype(np.uint8)
                    .repeat(h, 0))
    # 低频纹理（用噪声放大模拟）
    for _ in range(3):
        small = rng.integers(0, 256, (12, 16, 3)).astype(np.uint8)
        pool.append(cv2.resize(small, (900, 640), interpolation=cv2.INTER_CUBIC))
    # 纯色
    for c in [(60, 60, 60), (235, 235, 235), (30, 60, 120)]:
        pool.append(np.full((640, 900, 3), c, np.uint8))
    return pool


def rotate_badge(badge: np.ndarray, angle: float) -> np.ndarray:
    diag = int(np.hypot(*badge.shape[:2])) + 8
    canvas = np.zeros((diag, diag, 4), np.uint8)
    h, w = badge.shape[:2]
    y0, x0 = (diag - h) // 2, (diag - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = badge
    M = cv2.getRotationMatrix2D((diag / 2, diag / 2), angle, 1.0)
    return cv2.warpAffine(canvas, M, (diag, diag), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def perspective_badge(badge: np.ndarray, strength: float,
                      rng: np.random.Generator) -> np.ndarray:
    h, w = badge.shape[:2]
    d = strength * min(h, w)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + rng.uniform(-d, d, src.shape).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(badge, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(0, 0, 0, 0))


def paste(background: np.ndarray, badge: np.ndarray, rng: np.random.Generator):
    h, w = background.shape[:2]
    scale = float(rng.uniform(0.12, 0.55))
    tw = max(28, int(w * scale))
    th = max(28, int(tw * badge.shape[0] / badge.shape[1]))
    b = cv2.resize(badge, (tw, th), interpolation=cv2.INTER_AREA)
    b = rotate_badge(b, float(rng.uniform(-35, 35)))
    if rng.random() < 0.45:
        b = perspective_badge(b, 0.07, rng)

    ys, xs = np.where(b[:, :, 3] > 12)
    if len(xs) == 0:
        return np.zeros((h, w, 3), np.uint8), (0, 0, 1, 1)
    b = b[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    # 裁完之后可能比背景还大（高倍缩放 + 旋转外扩），先缩下来
    bh, bw = b.shape[:2]
    if bw > w * 0.92 or bh > h * 0.92:
        s = min(w * 0.9 / bw, h * 0.9 / bh)
        b = cv2.resize(b, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        bh, bw = b.shape[:2]

    out = background.copy()
    ox = int(rng.integers(0, max(1, w - bw)))
    oy = int(rng.integers(0, max(1, h - bh)))
    roi = out[oy:oy + bh, ox:ox + bw]
    alpha = b[:, :, 3:4].astype(np.float32) / 255.0
    roi[:] = (b[:, :, :3].astype(np.float32) * alpha
              + roi.astype(np.float32) * (1 - alpha)).astype(np.uint8)
    return out, (ox, oy, bw, bh)


def degrade(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    img = img.astype(np.float32) * float(rng.uniform(0.8, 1.2)) \
        + float(rng.uniform(-15, 15))
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        k = float(rng.uniform(0.6, 1.8))
        img = cv2.GaussianBlur(img, (0, 0), k)
    if rng.random() < 0.35:
        img = cv2.convertScaleAbs(img, alpha=float(rng.uniform(0.85, 1.15)),
                                  beta=float(rng.uniform(-20, 20)))
    return img


def main(n_pos: int = 40, n_neg: int = 30, seed: int = 11) -> None:
    rng = np.random.default_rng(seed)
    random.seed(seed)
    (OUT / "pos").mkdir(parents=True, exist_ok=True)
    (OUT / "neg").mkdir(parents=True, exist_ok=True)

    badge = cv2.imread(str(BADGE), cv2.IMREAD_UNCHANGED)
    if badge is None:
        raise SystemExit("缺少 assets/nmu_badge_mask.png，先跑 extract_template.py")
    bgs = load_backgrounds()
    print(f"背景池 {len(bgs)} 张")

    gt = {"pos": [], "neg": []}
    for i in range(n_pos):
        bg = bgs[i % len(bgs)]
        bg = cv2.resize(bg, (int(bg.shape[1] * 0.85), int(bg.shape[0] * 0.85)))
        img, box = paste(bg, badge, rng)
        img = degrade(img, rng)
        q = int(rng.integers(55, 92))
        name = f"pos_{i:03d}.jpg"
        cv2.imwrite(str(OUT / "pos" / name), img, [cv2.IMWRITE_JPEG_QUALITY, q])
        gt["pos"].append({"file": f"pos/{name}", "box": [int(v) for v in box]})

    for i in range(n_neg):
        bg = bgs[(i * 3 + 1) % len(bgs)]
        h, w = bg.shape[:2]
        # 随机裁剪一块 + 变换，模拟"不含校徽的普通图片"
        ch, cw = int(h * rng.uniform(0.6, 1.0)), int(w * rng.uniform(0.5, 0.9))
        y0 = int(rng.integers(0, max(1, h - ch)))
        x0 = int(rng.integers(0, max(1, w - cw)))
        img = bg[y0:y0 + ch, x0:x0 + cw].copy()
        if rng.random() < 0.5:
            ang = float(rng.uniform(-12, 12))
            M = cv2.getRotationMatrix2D((cw / 2, ch / 2), ang, 1.0)
            img = cv2.warpAffine(img, M, (cw, ch), borderMode=cv2.BORDER_REFLECT)
        img = degrade(img, rng)
        name = f"neg_{i:03d}.jpg"
        cv2.imwrite(str(OUT / "neg" / name), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        gt["neg"].append({"file": f"neg/{name}"})

    (OUT / "gt.json").write_text(json.dumps(gt, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    print(f"生成 {n_pos} 正样本 + {n_neg} 负样本 -> {OUT}")


if __name__ == "__main__":
    main()
