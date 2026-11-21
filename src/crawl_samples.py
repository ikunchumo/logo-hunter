"""
crawl_samples.py — 从南医大官网抓一批真实图片，作为「示例图库」。

抓完之后用 SIFT 检测器自动分成「含校徽 / 不含校徽」，人工只需抽查。
"""
from __future__ import annotations

import re
import ssl
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "samples" / "raw"

PAGES = [
    "https://90.njmu.edu.cn/",
    "https://90.njmu.edu.cn/BSXT/",
    "https://jwc.njmu.edu.cn/",
    "https://jwc.njmu.edu.cn/1053/list.htm",
    "https://www.njmu.edu.cn/xxgk/xxjj.htm",
]

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")


def opener() -> urllib.request.OpenerDirector:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    op = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
    op.addheaders = [("User-Agent", UA), ("Accept", "text/html,image/*,*/*")]
    return op


def fetch(op, url: str, timeout: int = 25) -> bytes:
    return op.open(url, timeout=timeout).read()


def main(max_images: int = 60) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    op = opener()
    seen: set[str] = set()
    saved = 0

    for page in PAGES:
        if saved >= max_images:
            break
        try:
            html = fetch(op, page).decode("utf-8", "ignore")
        except Exception as e:  # noqa: BLE001
            print(f"[跳过] {page}  {type(e).__name__}: {e}")
            continue
        print(f"[页面] {page}  {len(html)} 字节")
        # 页面里的图片 + CSS 背景图
        cands = re.findall(r'src="([^"]+\.(?:jpg|jpeg|png))"', html, re.I)
        cands += re.findall(r'url\(["\']?([^"\')]+\.(?:jpg|jpeg|png))', html, re.I)
        for src in cands:
            if saved >= max_images:
                break
            url = urllib.parse.urljoin(page, src)
            if url in seen:
                continue
            seen.add(url)
            name = urllib.parse.urlparse(url).path.rsplit("/", 1)[-1]
            name = re.sub(r"[^\w.\-]", "_", name)[:60]
            dst = OUT / name
            if dst.exists():
                continue
            try:
                data = fetch(op, url, timeout=20)
            except Exception:  # noqa: BLE001
                continue
            if len(data) < 6000 or len(data) > 6_000_000:
                continue
            dst.write_bytes(data)
            saved += 1
            print(f"  ✓ {name}  {len(data)//1024} KB")
            time.sleep(0.15)

    print(f"\n共下载 {saved} 张 -> {OUT}")


if __name__ == "__main__":
    main()
