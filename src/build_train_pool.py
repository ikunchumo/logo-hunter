"""
build_train_pool.py — 造「含目标 / 不含目标」二分类训练集。

正样本：把目标贴到真实背景（官网抓的图）上，随机旋转/缩放/透视/亮度/模糊
负样本：同一批背景的随机裁块，外加「相似干扰」——校庆标识、其他高校 logo、文字块

产物：data/train_pool.npz  （x: uint8 N×S×S×3, y: 0/1）
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "samples"
BADGE = ROOT / "assets" / "template_mask.png"
BADGE_CLR = ROOT / "assets" / "template.png"
POOL = ROOT / "data" / "train_pool.npz"
SIZE = 96


def load_backgrounds() -> list[np.ndarray]:
    bgs = []
    for d in (SAMPLES / "without_badge", SAMPLES / "raw"):
        for p in sorted(d.glob("*")):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            s = 640 / max(h, w)
            if s < 1:
                img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            bgs.append(img)
    print(f"背景池 {len(bgs)} 张")
    return bgs


def rotate_alpha(img: np.ndarray, angle: float) -> np.ndarray:
    diag = int(np.hypot(*img.shape[:2])) + 6
    canvas = np.zeros((diag, diag, 4), np.uint8)
    h, w = img.shape[:2]
    y0, x0 = (diag - h) // 2, (diag - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = img
    M = cv2.getRotationMatrix2D((diag / 2, diag / 2), angle, 1.0)
    return cv2.warpAffine(canvas, M, (diag, diag), flags=cv2.INTER_LINEAR,
                          borderValue=(0, 0, 0, 0))


def make_positive(bg: np.ndarray, badge: np.ndarray,
                  rng: np.random.Generator) -> np.ndarray:
    h, w = bg.shape[:2]
    scale = float(rng.uniform(0.08, 0.34))
    tw = max(18, int(w * scale))
    th = max(18, int(tw * badge.shape[0] / badge.shape[1]))
    b = cv2.resize(badge, (tw, th), interpolation=cv2.INTER_AREA)
    b = rotate_alpha(b, float(rng.uniform(-30, 30)))
    if rng.random() < 0.4:
        d = 0.05 * min(b.shape[:2])
        src = np.float32([[0, 0], [b.shape[1], 0], [b.shape[1], b.shape[0]], [0, b.shape[0]]])
        dst = src + rng.uniform(-d, d, src.shape).astype(np.float32)
        b = cv2.warpPerspective(b, cv2.getPerspectiveTransform(src, dst),
                                (b.shape[1], b.shape[0]), borderValue=(0, 0, 0, 0))
    ys, xs = np.where(b[:, :, 3] > 12)
    if len(xs) == 0:
        return cv2.resize(bg, (SIZE, SIZE))
    b = b[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    out = bg.copy()
    bh, bw = b.shape[:2]
    if bh > h or bw > w:
        return cv2.resize(bg, (SIZE, SIZE))
    ox = int(rng.integers(0, w - bw + 1))
    oy = int(rng.integers(0, h - bh + 1))
    roi = out[oy:oy + bh, ox:ox + bw]
    a = b[:, :, 3:4].astype(np.float32) / 255.0
    roi[:] = (b[:, :, :3].astype(np.float32) * a
              + roi.astype(np.float32) * (1 - a)).astype(np.uint8)

    # 贴着目标裁一块（含少量上下文），模拟真实图片里的局部区域
    cx = ox + bw // 2 + int(rng.integers(-bw // 3, bw // 3 + 1))
    cy = oy + bh // 2 + int(rng.integers(-bh // 3, bh // 3 + 1))
    side = int(max(bw, bh) * rng.uniform(1.6, 3.2))
    side = min(side, min(h, w))
    x0 = int(np.clip(cx - side // 2, 0, max(0, w - side)))
    y0 = int(np.clip(cy - side // 2, 0, max(0, h - side)))
    crop = out[y0:y0 + side, x0:x0 + side]
    return cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def make_negative(bg: np.ndarray, badge_clr: np.ndarray,
                  rng: np.random.Generator) -> np.ndarray:
    h, w = bg.shape[:2]
    side = int(min(h, w) * rng.uniform(0.12, 0.5))
    x0 = int(rng.integers(0, max(1, w - side)))
    y0 = int(rng.integers(0, max(1, h - side)))
    crop = bg[y0:y0 + side, x0:x0 + side].copy()

    # 30% 加「相似干扰」：把目标模糊/极度缩小/裁剪，让人也难一眼认出
    if rng.random() < 0.3:
        s = float(rng.uniform(0.04, 0.12))
        tw = max(10, int(w * s))
        th = max(10, int(tw * badge_clr.shape[0] / badge_clr.shape[1]))
        b = cv2.resize(badge_clr, (tw, th))
        if rng.random() < 0.5:
            b = cv2.GaussianBlur(b, (0, 0), 1.6)
        if th < crop.shape[0] and tw < crop.shape[1]:
            ox = int(rng.integers(0, crop.shape[1] - tw))
            oy = int(rng.integers(0, crop.shape[0] - th))
            crop[oy:oy + th, ox:ox + tw] = b
    return cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def to_bgra(img: np.ndarray) -> np.ndarray:
    """把任意图转成带 alpha 的版本：白底变透明。

    用「到白色的距离」当不透明度，比单纯取灰度更通用 ——
    彩色 logo（白底红字）也能得到正确的前景遮罩。
    """
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return img
    dist = np.linalg.norm(img.astype(np.float32) - 255.0, axis=2)
    alpha = np.clip(dist / 110.0 * 255.0, 0, 255).astype(np.uint8)
    return np.dstack([img, alpha])


def build_pool(template_path: Path | None = None,
               out_path: Path = POOL, n_per_class: int = 1200,
               seed: int = 21, verbose: bool = True) -> dict:
    """用指定模板合成二分类训练池。返回统计信息。"""
    tpl_path = Path(template_path) if template_path else BADGE_CLR
    raw = cv2.imread(str(tpl_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"读不到模板：{tpl_path}")
    badge = to_bgra(raw)
    badge_clr = badge[:, :, :3].copy()

    rng = np.random.default_rng(seed)
    bgs = load_backgrounds()

    xs, ys = [], []
    for i in range(n_per_class):
        xs.append(make_positive(bgs[i % len(bgs)], badge, rng))
        ys.append(1)
        if verbose and i % 300 == 0:
            print(f"  正样本 {i}/{n_per_class}", flush=True)
    for i in range(n_per_class):
        xs.append(make_negative(bgs[(i * 7 + 3) % len(bgs)], badge_clr, rng))
        ys.append(0)
        if verbose and i % 300 == 0:
            print(f"  负样本 {i}/{n_per_class}", flush=True)

    x = np.stack(xs).astype(np.uint8)
    y = np.array(ys, dtype=np.int64)
    idx = rng.permutation(len(x))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, x=x[idx], y=y[idx])
    info = {"n": int(len(x)), "n_pos": int(y.sum()), "size": SIZE,
            "template": str(tpl_path), "path": str(out_path),
            "mb": round(out_path.stat().st_size / 1e6, 1)}
    if verbose:
        print(f"训练池 {x.shape} -> {out_path}  ({info['mb']} MB)")
    return info


def main(n_per_class: int = 1200, seed: int = 21) -> None:
    build_pool(BADGE_CLR, POOL, n_per_class=n_per_class, seed=seed)


if __name__ == "__main__":
    main()
