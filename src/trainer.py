"""
trainer.py — 可在网页上实时调参的「含目标 / 不含目标」二分类训练器。

和 SIFT 检测器的分工：
  SIFT    : 不需要训练，直接框出目标位置，换 logo 只换模板
  CNN     : 需要训练，判断「这张图里有没有目标」，适合当粗筛（快、可批量）
两者在网页上可以对照看。

状态放在全局 STATE 里，网页通过 /api/train/status 轮询。
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "data" / "train_pool.npz"
MODEL = ROOT / "models" / "badge_cnn.pt"
# 训练目标可以和扫描页的 SIFT 模板不同：CNN 只需要"图里长什么样"，
# 不需要 SIFT 那样的强纹理，所以这里单独存一份。
TRAIN_TPL = ROOT / "assets" / "train_template.png"

STATE: dict = {
    "running": False, "done": False, "error": None,
    "epoch": 0, "total_epochs": 0,
    "loss": [], "train_acc": [], "val_acc": [], "val_loss": [],
    "log": [], "params": {}, "result": None, "seconds": 0.0,
    "device": "",
}
_LOCK = threading.Lock()
_RESET_EVENTS = ("start",)


class SmallCNN(nn.Module):
    def __init__(self, n_cls: int = 2, width: int = 48):
        super().__init__()

        def block(ci, co, n=2):
            layers = []
            for k in range(n):
                layers += [nn.Conv2d(ci if k == 0 else co, co, 3, padding=1,
                                     bias=False),
                           nn.BatchNorm2d(co), nn.ReLU(inplace=True)]
            layers.append(nn.MaxPool2d(2))
            return layers

        self.features = nn.Sequential(
            *block(3, width), *block(width, width * 2),
            *block(width * 2, width * 4, n=1), *block(width * 4, width * 4, n=1))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.2), nn.Linear(width * 4, n_cls))

    def forward(self, x):
        return self.head(self.features(x))


def _load_pool():
    d = np.load(POOL)
    return d["x"], d["y"]


def _to_tensor(x: np.ndarray, size: int) -> torch.Tensor:
    t = torch.from_numpy(x).permute(0, 3, 1, 2).float().div_(255.0)
    if size != t.shape[-1]:
        t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return (t - 0.5) / 0.25


def _rand_affine(x: torch.Tensor, deg: float, shift: float, scale: float):
    b = x.shape[0]
    dev = x.device
    ang = (torch.rand(b, device=dev) * 2 - 1) * deg * np.pi / 180
    sh = (torch.rand(b, 2, device=dev) * 2 - 1) * shift
    sc = 1 + (torch.rand(b, device=dev) * 2 - 1) * scale
    c, s = torch.cos(ang), torch.sin(ang)
    th = torch.zeros(b, 2, 3, device=dev)
    th[:, 0, 0] = c / sc
    th[:, 0, 1] = -s / sc
    th[:, 0, 2] = sh[:, 0]
    th[:, 1, 0] = s / sc
    th[:, 1, 1] = c / sc
    th[:, 1, 2] = sh[:, 1]
    grid = F.affine_grid(th, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="border")


def _rand_color(x: torch.Tensor, strength: float):
    b = x.shape[0]
    s = 1 + (torch.rand(b, 1, 1, 1, device=x.device) * 2 - 1) * strength
    o = (torch.rand(b, 3, 1, 1, device=x.device) * 2 - 1) * strength * 1.5
    return x * s + o


def _rand_erase(x: torch.Tensor, p: float, rng_ratio: float = 0.25):
    b, _, h, w = x.shape
    for i in range(b):
        if torch.rand(1).item() > p:
            continue
        eh = int(h * torch.empty(1).uniform_(0.08, rng_ratio).item())
        ew = int(w * torch.empty(1).uniform_(0.08, rng_ratio).item())
        y = int(torch.randint(0, max(1, h - eh), (1,)).item())
        xx = int(torch.randint(0, max(1, w - ew), (1,)).item())
        x[i, :, y:y + eh, xx:xx + ew] = 0.0
    return x


@torch.no_grad()
def _evaluate(model, x, y, device, bs=512):
    model.eval()
    preds = []
    for i in range(0, len(x), bs):
        preds.append(model(x[i:i + bs].to(device)).argmax(1).cpu())
    p = torch.cat(preds)
    yt = y.cpu()
    acc = (p == yt).float().mean().item()
    tp = int(((p == 1) & (yt == 1)).sum())
    fp = int(((p == 1) & (yt == 0)).sum())
    fn = int(((p == 0) & (yt == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return acc, prec, rec, p, yt


def _log(msg: str) -> None:
    with _LOCK:
        STATE["log"].append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        if len(STATE["log"]) > 400:
            STATE["log"] = STATE["log"][-400:]


def status() -> dict:
    with _LOCK:
        s = dict(STATE)
    if not s.get("device"):
        try:
            s["device"] = (torch.cuda.get_device_name(0)
                           if torch.cuda.is_available() else "CPU")
        except Exception:  # noqa: BLE001
            s["device"] = "CPU"
    return s


def start(params: dict) -> dict:
    with _LOCK:
        if STATE["running"]:
            return {"ok": False, "error": "已有训练在跑，等它结束或先点停止"}
        STATE.update({
            "running": True, "done": False, "error": None,
            "epoch": 0, "total_epochs": int(params.get("epochs", 12)),
            "loss": [], "train_acc": [], "val_acc": [], "val_loss": [],
            "log": [], "params": dict(params), "result": None,
            "seconds": 0.0,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        })
    threading.Thread(target=_run, args=(dict(params),), daemon=True).start()
    return {"ok": True, "device": STATE["device"]}


def _run(p: dict) -> None:
    t0 = time.time()
    try:
        if not POOL.exists():
            raise RuntimeError("缺少 data/train_pool.npz，先运行 src/build_train_pool.py")
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(int(p.get("seed", 7)))

        per_class = int(p.get("per_class", 600))
        epochs = int(p.get("epochs", 12))
        lr = float(p.get("lr", 3e-3))
        batch = int(p.get("batch", 64))
        aug = float(p.get("aug", 0.6))
        size = int(p.get("size", 96))
        width = int(p.get("width", 48))

        x, y = _load_pool()
        idx1 = np.where(y == 1)[0][:per_class]
        idx0 = np.where(y == 0)[0][:per_class]
        idx = np.concatenate([idx1, idx0])
        rng = np.random.default_rng(int(p.get("seed", 7)))
        rng.shuffle(idx)
        xt = _to_tensor(x[idx], size).to(dev)
        yt = torch.from_numpy(y[idx]).long().to(dev)
        n = len(xt)
        n_te = max(40, int(n * 0.18))
        n_va = max(40, int(n * 0.12))
        te_sl = slice(0, n_te)
        va_sl = slice(n_te, n_te + n_va)
        tr_sl = slice(n_te + n_va, n)
        _log(f"数据 {n} 张（训练 {n - n_te - n_va} / 验证 {n_va} / 测试 {n_te}）"
             f"  {size}px")
        _log(f"模型 SmallCNN(width={width})  参数 "
             f"{sum(q.numel() for q in SmallCNN(2, width).parameters())/1e3:.0f}K")

        model = SmallCNN(2, width).to(dev)
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                              weight_decay=5e-4, nesterov=True)
        steps = max(1, (n - n_te - n_va) // batch)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr, total_steps=epochs * steps, pct_start=0.25)
        crit = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler("cuda", enabled=(dev.type == "cuda"))

        xtr, ytr = xt[tr_sl], yt[tr_sl]
        xva, yva = xt[va_sl], yt[va_sl]
        xte, yte = xt[te_sl], yt[te_sl]

        for ep in range(1, epochs + 1):
            if not STATE["running"]:
                _log("训练被中止")
                break
            model.train()
            perm = torch.randperm(len(xtr), device=dev)
            tot = cor = 0
            lsum = 0.0
            for i in range(0, len(xtr) - batch + 1, batch):
                idb = perm[i:i + batch]
                xb = xtr[idb]
                if aug > 0:
                    xb = _rand_affine(xb, 14 * aug, 0.07 * aug, 0.12 * aug)
                    xb = _rand_color(xb, 0.18 * aug)
                    xb = _rand_erase(xb, 0.25 * aug)
                yb = ytr[idb]
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.float16,
                                        enabled=(dev.type == "cuda")):
                    out = model(xb)
                    loss = crit(out, yb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                lsum += float(loss.detach()) * len(yb)
                cor += int((out.argmax(1) == yb).sum())
                tot += len(yb)
            va_acc, _, _, _, _ = _evaluate(model, xva, yva, dev)
            with torch.no_grad():
                model.eval()
                vl = 0.0
                for i in range(0, len(xva), 256):
                    vl += float(F.cross_entropy(model(xva[i:i + 256]), yva[i:i + 256],
                                                reduction="sum"))
                vl /= max(len(xva), 1)
            with _LOCK:
                STATE["epoch"] = ep
                STATE["loss"].append(round(lsum / max(tot, 1), 4))
                STATE["train_acc"].append(round(cor / max(tot, 1), 4))
                STATE["val_acc"].append(round(float(va_acc), 4))
                STATE["val_loss"].append(round(float(vl), 4))
                STATE["seconds"] = round(time.time() - t0, 1)
            _log(f"epoch {ep:02d}/{epochs}  loss={lsum/max(tot,1):.4f}  "
                 f"train={cor/max(tot,1)*100:.1f}%  val={va_acc*100:.1f}%")

        acc, prec, rec, pred, truth = _evaluate(model, xte, yte, dev)
        cm = np.zeros((2, 2), int)
        for t, q in zip(truth.tolist(), pred.tolist()):
            cm[t, q] += 1
        MODEL.parent.mkdir(exist_ok=True)
        torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "classes": ["without_badge", "with_badge"],
                    "size": size, "width": width, "test_acc": acc},
                   MODEL)
        with _LOCK:
            STATE["result"] = {
                "test_acc": round(float(acc), 4),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "confusion": cm.tolist(),
                "n_train": int(n - n_te - n_va), "n_test": int(n_te),
                "seconds": round(time.time() - t0, 1),
                "params": int(sum(q.numel() for q in model.parameters())),
                "device": STATE["device"],
            }
            STATE["running"] = False
            STATE["done"] = True
            STATE["seconds"] = round(time.time() - t0, 1)
        _log(f"完成：测试准确率 {acc*100:.2f}%  精确率 {prec*100:.1f}%  "
             f"召回 {rec*100:.1f}%  用时 {time.time()-t0:.1f}s")
    except Exception as e:  # noqa: BLE001
        with _LOCK:
            STATE["running"] = False
            STATE["error"] = f"{type(e).__name__}: {e}"
        _log(f"出错：{STATE['error']}")


def stop() -> dict:
    with _LOCK:
        STATE["running"] = False
    _log("已请求停止")
    return {"ok": True}


def rebuild_pool(template_path: str | Path | None = None,
                 n_per_class: int = 600) -> dict:
    """用指定模板重建二分类训练池（正样本贴该模板，负样本随机块）。

    默认用「训练目标」图（TRAIN_TPL）；没有的话回退到扫描页的 SIFT 模板。
    """
    from build_train_pool import build_pool, POOL

    if template_path:
        tpl = Path(template_path)
    elif TRAIN_TPL.exists():
        tpl = TRAIN_TPL
    else:
        tpl = ROOT / "assets" / "template.png"
    if not tpl.exists():
        raise FileNotFoundError(f"找不到模板：{tpl}")

    t0 = time.time()
    info = build_pool(tpl, POOL, n_per_class=n_per_class, verbose=False)
    info["seconds"] = round(time.time() - t0, 1)
    info["source"] = "custom" if tpl == TRAIN_TPL else "default"
    with _LOCK:
        STATE["done"] = False
        STATE["result"] = None
        STATE["loss"] = []
        STATE["val_acc"] = []
        STATE["train_acc"] = []
        STATE["val_loss"] = []
        STATE["epoch"] = 0
        STATE["log"] = []
    _log(f"训练池已重建：{info['n']} 张（{n_per_class}/类），模板 {tpl.name}")
    return info


def set_train_template(img_bgr) -> dict:
    """把上传的图设为训练目标（不影响 SIFT 检测模板）。"""
    import cv2

    h, w = img_bgr.shape[:2]
    if max(h, w) > 800:
        s = 800 / max(h, w)
        img_bgr = cv2.resize(img_bgr, None, fx=s, fy=s,
                             interpolation=cv2.INTER_AREA)
    TRAIN_TPL.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(TRAIN_TPL), img_bgr)
    return {"path": str(TRAIN_TPL), "width": img_bgr.shape[1],
            "height": img_bgr.shape[0], "source": "custom"}


def clear_train_template() -> dict:
    if TRAIN_TPL.exists():
        TRAIN_TPL.unlink()
    return {"ok": True, "source": "default"} 


def train_template_info() -> dict:
    """返回训练目标的基本信息（缩略图由调用方按 path 提供）。"""
    use_custom = TRAIN_TPL.exists()
    p = TRAIN_TPL if use_custom else ROOT / "assets" / "template.png"
    return {"path": str(p), "source": "custom" if use_custom else "default"}


# --------------------------------------------------------------------------
# CNN 搜索演示：用训好的模型扫一遍示例图库，和 SIFT 的结果对照
# --------------------------------------------------------------------------
def scan_with_model(limit: int = 60) -> dict:
    """用 models/badge_cnn.pt 对示例图库做二分类推理。

    返回每张图的 P(有目标)，并附真实标签（按所在目录）。
    调用方可以再合并 SIFT 的检测结果做对照 —— 这正是"CNN 粗筛"方案的实测：
    整图缩到 96px 后，小目标会被漏掉。
    """
    import cv2

    if not MODEL.exists():
        raise FileNotFoundError("还没有训练好的模型，先在训练页跑一次")
    blob = torch.load(MODEL, map_location="cpu")
    net = SmallCNN(2, blob.get("width", 48))
    net.load_state_dict(blob["state_dict"])
    net.eval()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(dev)
    size = int(blob.get("size", 96))

    samples = ROOT / "data" / "samples"
    items: list[tuple[Path, int]] = []
    for name, label in (("with_badge", 1), ("without_badge", 0)):
        d = samples / name
        if not d.exists():
            continue
        for p in sorted(d.glob("*.jpg")):
            items.append((p, label))
    items = items[:limit]
    if not items:
        raise FileNotFoundError("示例图库为空：data/samples/")

    t0 = time.time()
    out: list[dict] = []
    batch: list[np.ndarray] = []
    metas: list[tuple[Path, int]] = []

    def flush():
        if not batch:
            return
        x = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).float().div_(255.0)
        x = ((x - 0.5) / 0.25).to(dev)
        with torch.no_grad():
            prob = F.softmax(net(x), dim=1)[:, 1].cpu().numpy()
        for (p, truth), pr in zip(metas, prob):
            out.append({"file": str(p), "name": p.name, "prob": round(float(pr), 4),
                        "cnn_found": bool(pr >= 0.5), "truth": truth})
        batch.clear()
        metas.clear()

    for p, truth in items:
        data = np.fromfile(str(p), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            continue
        # 统一到模型输入尺寸，否则 batch 拼不起来
        img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (size, size),
                         interpolation=cv2.INTER_AREA)
        batch.append(img)
        metas.append((p, truth))
        if len(batch) >= 64:
            flush()
    flush()

    n_pos = sum(1 for r in out if r["truth"] == 1)
    tp = sum(1 for r in out if r["truth"] == 1 and r["cnn_found"])
    fp = sum(1 for r in out if r["truth"] == 0 and r["cnn_found"])
    return {
        "total": len(out), "n_pos": n_pos, "tp": tp, "fp": fp,
        "recall": round(tp / max(n_pos, 1), 4),
        "precision": round(tp / max(tp + fp, 1), 4),
        "seconds": round(time.time() - t0, 2),
        "size": size, "device": str(dev),
        "results": out,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(start({}), ensure_ascii=False))
    while True:
        time.sleep(1.5)
        s = status()
        print(f"ep {s['epoch']}/{s['total_epochs']}  "
              f"loss={s['loss'][-1] if s['loss'] else '-'}  "
              f"val={s['val_acc'][-1] if s['val_acc'] else '-'}")
        if s["done"] or s["error"]:
            print(json.dumps(s["result"] or s["error"], ensure_ascii=False, indent=2))
            break
