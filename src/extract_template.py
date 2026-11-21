"""
extract_template.py — 从官方标准图里裁出目标，做成检测模板。

输入：assets/candidate_1.jpg（目标 + 标准字）
输出：assets/template.png（纯目标，去白底）
      assets/template_mask.png（透明通道）
      docs/template_check.png（模板 + SIFT 特征点可视化）
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
DOCS = ROOT / "docs"


def main() -> None:
    src = cv2.imread(str(ASSETS / "candidate_1.jpg"))
    if src is None:
        raise SystemExit("缺少 assets/candidate_1.jpg")
    h, w = src.shape[:2]
    print("源图", w, "x", h)

    # 只在左半边找（右边是标准字）
    left = src[:, : int(w * 0.42)]
    gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)

    # 白底 → 取非白区域
    _, mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            np.ones((7, 7), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise SystemExit("没找到目标区域")
    c = max(cnts, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(c)
    pad = 6
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(left.shape[1], x + bw + pad), min(left.shape[0], y + bh + pad)
    badge = left[y0:y1, x0:x1].copy()
    print(f"裁剪区域 ({x0},{y0})-({x1},{y1})  {badge.shape[1]}x{badge.shape[0]}")

    # 放大 2 倍并锐化：JPEG 压缩过的边缘太软，SIFT 需要一点纹理
    badge = cv2.resize(badge, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(badge, (0, 0), 1.2)
    badge = cv2.addWeighted(badge, 1.55, blur, -0.55, 0)
    cv2.imwrite(str(ASSETS / "template.png"), badge)

    # 透明底版本（白底转 alpha）
    g = cv2.cvtColor(badge, cv2.COLOR_BGR2GRAY)
    alpha = cv2.threshold(g, 246, 255, cv2.THRESH_BINARY_INV)[1]
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    bgra = cv2.cvtColor(badge, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha
    cv2.imwrite(str(ASSETS / "template_mask.png"), bgra)

    # 模板的 SIFT 质量检查
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(cv2.cvtColor(badge, cv2.COLOR_BGR2GRAY), None)
    print(f"SIFT 关键点 {len(kp)} 个，描述子 {None if des is None else des.shape}")

    vis = cv2.drawKeypoints(badge, kp, None,
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    stack = np.hstack([badge, vis])
    cv2.putText(stack, "template + SIFT keypoints", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    DOCS.mkdir(exist_ok=True)
    cv2.imwrite(str(DOCS / "template_check.png"), stack)
    print("-> assets/template.png  docs/template_check.png")


if __name__ == "__main__":
    main()
