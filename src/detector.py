"""
detector.py — 用 SIFT 特征匹配在任意图片里定位「南京医科大学校徽」。

为什么用特征匹配而不是训练一个检测网络？
----------------------------------------
校徽是一个**固定图案**：可能有旋转、缩放、透视、光照变化、轻微遮挡，
但图案本身的局部纹理（盾形轮廓、蛇杖、麦穗、文字）不会变。
SIFT 恰好对这些变换不变，而且**不需要任何训练数据**——只要一张模板。

流程
----
  模板 SIFT 特征
      ↓
  待检图多尺度 SIFT 特征（1.0x / 1.7x，兼顾远拍小目标和近拍大目标）
      ↓
  FLANN KNN 匹配 + Lowe ratio 筛选
      ↓
  RANSAC 单应性估计（把模板四角投影到图上）
      ↓
  几何校验：内点数、内点比例、投影框长宽比、面积
      ↓
  输出：是否命中 + 外框 + 置信度
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "assets" / "nmu_badge.png"


@dataclass
class Hit:
    found: bool
    score: float = 0.0
    box: tuple[int, int, int, int] | None = None      # x, y, w, h
    quad: np.ndarray | None = None                    # 4×2 角点
    inliers: int = 0
    good_matches: int = 0
    scale: float = 1.0
    elapsed_ms: float = 0.0
    extra: dict = field(default_factory=dict)


class BadgeDetector:
    def __init__(self, template_path: Path = TEMPLATE,
                 nfeatures: int = 0, ratio: float = 0.72,
                 min_inliers: int = 12, min_inlier_ratio: float = 0.26,
                 scales: tuple[float, ...] = (1.0, 1.7, 2.6),
                 max_side: int = 1280):
        img = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"缺少模板 {template_path}")
        self.template = img
        self.th, self.tw = img.shape[:2]
        self.aspect = self.tw / self.th

        self.sift = cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=0.01)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.kp_t, self.des_t = self.sift.detectAndCompute(gray, None)
        if self.des_t is None:
            raise RuntimeError("模板提取不到 SIFT 特征")
        self.pt_t = np.float32([k.pt for k in self.kp_t])

        index_params = dict(algorithm=1, trees=5)      # FLANN_INDEX_KDTREE
        search_params = dict(checks=64)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        self.ratio = ratio
        self.min_inliers = min_inliers
        self.min_inlier_ratio = min_inlier_ratio
        self.scales = scales
        self.max_side = max_side
        self.corners_t = np.float32(
            [[0, 0], [self.tw, 0], [self.tw, self.th], [0, self.th]]).reshape(-1, 1, 2)

    # ------------------------------------------------------------------
    def _match_at(self, gray: np.ndarray, scale: float) -> Hit:
        kp, des = self.sift.detectAndCompute(gray, None)
        if des is None or len(kp) < 8:
            return Hit(False, scale=scale)
        matches = self.flann.knnMatch(self.des_t, des, k=2)
        good = [m for m, n in matches
                if m.distance < self.ratio * n.distance] if matches else []
        if len(good) < max(self.min_inliers, 6):
            return Hit(False, good_matches=len(good), scale=scale)

        src = self.pt_t[[m.queryIdx for m in good]].reshape(-1, 1, 2)
        dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None or mask is None:
            return Hit(False, good_matches=len(good), scale=scale)
        inl = int(mask.sum())
        if inl == 0:
            return Hit(False, good_matches=len(good), scale=scale)

        proj = cv2.perspectiveTransform(self.corners_t, H)
        proj = proj.reshape(-1, 2) / scale
        x, y, w, h = cv2.boundingRect(proj.astype(np.float32))
        if w < 12 or h < 12:
            return Hit(False, good_matches=len(good), scale=scale)

        # 单应性投影的框经常把背景圈进来（匹配点集中在图案中部时会外扩）。
        # 用「内点在图上的外接框」兜底：内点只覆盖有纹理的区域，外扩一点更贴合。
        inl_pts = (dst.reshape(-1, 2)[mask.ravel() == 1]) / scale
        ix, iy, iw, ih = cv2.boundingRect(inl_pts.astype(np.float32))
        px, py = int(iw * 0.22), int(ih * 0.22)
        ix, iy = max(0, ix - px), max(0, iy - py)
        iw, ih = iw + 2 * px, ih + 2 * py
        if w * h > iw * ih * 2.2:
            x, y, w, h = ix, iy, iw, ih

        ar = w / max(h, 1)
        # 盾形长宽比约 0.77，允许透视带来的形变
        if not (0.45 <= ar <= 1.45):
            return Hit(False, good_matches=len(good), scale=scale)

        inlier_ratio = inl / len(good)
        score = inl * inlier_ratio
        ok = (inl >= self.min_inliers) and (inlier_ratio >= self.min_inlier_ratio)
        return Hit(ok, score=score, box=(int(x), int(y), int(w), int(h)),
                   quad=proj, inliers=inl, good_matches=len(good), scale=scale)

    # ------------------------------------------------------------------
    def detect(self, img_bgr: np.ndarray, max_side: int | None = None) -> Hit:
        import time
        t0 = time.perf_counter()
        h, w = img_bgr.shape[:2]
        lim = max_side or self.max_side
        base = min(1.0, lim / max(h, w))
        best = Hit(False)
        cache: dict[float, np.ndarray] = {}

        for s in self.scales:
            real = base * s
            if max(h, w) * real > 2400:      # 太大就跳过，SIFT 会很慢
                continue
            key = round(real, 3)
            if key not in cache:
                if abs(real - 1.0) < 1e-3:
                    cache[key] = img_bgr
                else:
                    cache[key] = cv2.resize(img_bgr, None, fx=real, fy=real,
                                            interpolation=cv2.INTER_AREA if real < 1
                                            else cv2.INTER_CUBIC)
            gray = cv2.cvtColor(cache[key], cv2.COLOR_BGR2GRAY)
            hit = self._match_at(gray, real)
            if hit.found and hit.score > best.score:
                best = hit
            # 级联早停：已经在原尺度上很确定，就不用再试放大的尺度了
            if hit.found and hit.inliers >= self.min_inliers * 2.2:
                break

        if not best.found:
            # 记录最接近的一次，便于调试
            for s in self.scales:
                real = base * s
                key = round(real, 3)
                if key in cache:
                    h2 = self._match_at(cv2.cvtColor(cache[key], cv2.COLOR_BGR2GRAY), real)
                    if h2.inliers > best.inliers:
                        best = h2
        best.elapsed_ms = (time.perf_counter() - t0) * 1000
        return best

    # ------------------------------------------------------------------
    def detect_file(self, path: str | Path) -> Hit:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return Hit(False, extra={"error": "无法读取"})
        return self.detect(img)


if __name__ == "__main__":
    d = BadgeDetector()
    print(f"模板 {d.tw}×{d.th}，SIFT 点 {len(d.kp_t)}")
    print("长宽比", round(d.aspect, 3))
