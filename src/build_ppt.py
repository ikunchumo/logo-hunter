"""
build_ppt.py — 生成项目汇报 PPT（16:9，16 页，含讲稿备注）。

数据来源：项目实测结果 + docs/*.png
"""
from __future__ import annotations

from pathlib import Path

from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
import os
OUT = Path(os.environ.get("PPT_OUT", DOCS / "目标检测器_项目汇报.pptx"))

FONT = "Microsoft YaHei"
INK = RGBColor(0x1D, 0x1D, 0x1F)
SUB = RGBColor(0x6E, 0x6E, 0x73)
BLUE = RGBColor(0x00, 0x71, 0xE3)
GREEN = RGBColor(0x34, 0xC7, 0x59)
AMBER = RGBColor(0xFF, 0x95, 0x00)
RED = RGBColor(0xFF, 0x3B, 0x30)
DARK = RGBColor(0x0F, 0x2A, 0x44)
LIGHT = RGBColor(0xF5, 0xF5, 0xF7)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LINE = RGBColor(0xD2, 0xD2, 0xD7)

W, H = Inches(13.333), Inches(7.5)


# ---------------------------------------------------------------- helpers
def set_font(run, size, bold=False, color=INK, name=FONT):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = name
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = etree.SubElement(rPr, qn(tag))
        el.set("typeface", name)


def textbox(slide, l, t, w, h, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    return tf


def para(tf, text, size, bold=False, color=INK, before=0, after=6,
         align=PP_ALIGN.LEFT, first=False, line=1.25):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.space_before = Pt(before)
    p.space_after = Pt(after)
    p.line_spacing = line
    r = p.add_run()
    r.text = text
    set_font(r, size, bold, color)
    return p


def rect(slide, l, t, w, h, fill, shape=MSO_SHAPE.RECTANGLE, radius=None,
         line_color=None):
    s = slide.shapes.add_shape(shape, l, t, w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line_color is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line_color
        s.line.width = Pt(0.75)
    s.shadow.inherit = False
    if radius is not None and shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    return s


def pic(slide, path, l, t, max_w, max_h):
    p = Path(path)
    if not p.exists():
        return None
    iw, ih = Image.open(p).size
    k = min(max_w / iw, max_h / ih)
    w, h = int(iw * k), int(ih * k)
    return slide.shapes.add_picture(str(p), int(l + (max_w - w) / 2),
                                    int(t + (max_h - h) / 2), w, h)


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def page(prs, kicker, title, no, total=16):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    rect(s, 0, 0, W, Inches(0.08), BLUE)
    tf = textbox(s, Inches(0.72), Inches(0.38), Inches(11.9), Inches(0.3))
    para(tf, kicker.upper(), 11.5, True, BLUE, first=True, after=0)
    tf = textbox(s, Inches(0.72), Inches(0.68), Inches(11.9), Inches(0.7))
    para(tf, title, 26, True, INK, first=True, after=0)
    tf = textbox(s, Inches(0.72), Inches(7.05), Inches(8), Inches(0.28))
    para(tf, "目标检测器 · 项目汇报", 9, False, RGBColor(0x9A, 0x9A, 0x9E),
         first=True, after=0)
    tf = textbox(s, Inches(11.9), Inches(7.05), Inches(0.7), Inches(0.28))
    para(tf, f"{no}/{total}", 9, False, RGBColor(0x9A, 0x9A, 0x9E),
         first=True, after=0, align=PP_ALIGN.RIGHT)
    return s


def table(slide, rows, cols, l, t, w, h, header_fill=DARK, size=11):
    tb = slide.shapes.add_table(rows, cols, l, t, w, h).table
    return tb


def fill_table(tb, header, body, size=10.5, first_col_bold=False,
               header_size=11):
    for j, htxt in enumerate(header):
        c = tb.cell(0, j)
        c.text = ""
        r = c.text_frame.paragraphs[0].add_run()
        r.text = htxt
        set_font(r, header_size, True, WHITE)
        c.fill.solid()
        c.fill.fore_color.rgb = DARK
        c.vertical_anchor = MSO_ANCHOR.MIDDLE
        c.margin_left = c.margin_right = Inches(0.09)
    for i, row in enumerate(body, start=1):
        for j, txt in enumerate(row):
            c = tb.cell(i, j)
            c.text = ""
            p = c.text_frame.paragraphs[0]
            r = p.add_run()
            r.text = txt
            set_font(r, size, (first_col_bold and j == 0), INK if j == 0 else SUB)
            c.fill.solid()
            c.fill.fore_color.rgb = WHITE if i % 2 else LIGHT
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.margin_left = c.margin_right = Inches(0.09)


# ---------------------------------------------------------------- build
def build() -> None:
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    TOTAL = 17

    # ---- 1 封面 ----
    s = prs.slides.add_slide(prs.slide_layouts[6])
    rect(s, 0, 0, W, H, DARK)
    rect(s, Inches(0.95), Inches(2.5), Inches(0.12), Inches(2.1), BLUE)
    tf = textbox(s, Inches(1.35), Inches(2.3), Inches(10.5), Inches(0.35))
    para(tf, "IMAGE RETRIEVAL · 项目汇报", 12.5, True,
         RGBColor(0x8A, 0xC4, 0xFF), first=True, after=0)
    tf = textbox(s, Inches(1.35), Inches(2.68), Inches(11), Inches(1.5))
    para(tf, "在一堆图片里", 44, True, WHITE, first=True, after=2)
    para(tf, "找出含指定目标的那几张", 44, True, WHITE, after=8)
    tf = textbox(s, Inches(1.35), Inches(4.85), Inches(10), Inches(1.2))
    para(tf, "SIFT 特征匹配 + 单应性验证，附可视化训练页与网页版检索器",
         15, False, RGBColor(0xC7, 0xD4, 0xE2), first=True, after=14)
    para(tf, "汇报人：________　　日期：2026.11", 12.5, False,
         RGBColor(0x9F, 0xB0, 0xC4))
    notes(s, "开场：这次汇报的项目是「图片目标检索」——给一个装满各种图片的文件夹，"
             "自动挑出里面含某个特定 logo / 标识的那几张，并且框出位置。\n"
             "技术上用了两条路线：一条是不需要训练的 SIFT 特征匹配，"
             "另一条是可训练 CNN 二分类，最后对比了两者的适用场景。")

    # ---- 2 问题 ----
    s = page(prs, "Problem", "要解决的问题", 2, TOTAL)
    tf = textbox(s, Inches(0.72), Inches(1.55), Inches(11.9), Inches(0.6))
    para(tf, "不是「识别这是什么」，而是「在一堆无关图片里，把含它的挑出来」。",
         15, False, SUB, first=True, after=0)
    cards = [
        ("输入", "一个文件夹\n几百到几万张图片", "混着风景、海报、\n截图、二维码", BLUE),
        ("要找的", "一个固定图案\n（logo / 标识 / 徽章）", "可能被旋转、缩放、\n压缩、贴在小角落", AMBER),
        ("输出", "命中的图片列表", "并框出目标的\n位置", GREEN),
    ]
    for i, (t, d1, d2, color) in enumerate(cards):
        x = Inches(0.72 + i * 4.06)
        rect(s, x, Inches(2.35), Inches(3.72), Inches(2.5), LIGHT,
             MSO_SHAPE.ROUNDED_RECTANGLE, 0.06)
        rect(s, x, Inches(2.35), Inches(3.72), Inches(0.5), color)
        tf = textbox(s, x + Inches(0.22), Inches(2.45), Inches(3.2), Inches(0.35))
        para(tf, t, 13.5, True, WHITE, first=True, after=0)
        tf = textbox(s, x + Inches(0.26), Inches(3.0), Inches(3.2), Inches(0.9))
        para(tf, d1, 13, True, INK, first=True, after=0, line=1.35)
        tf = textbox(s, x + Inches(0.26), Inches(3.95), Inches(3.2), Inches(0.8))
        para(tf, d2, 11.5, False, SUB, first=True, after=0, line=1.35)

    tf = textbox(s, Inches(0.72), Inches(5.2), Inches(11.9), Inches(1.6))
    para(tf, "难点在哪", 13.5, True, BLUE, first=True, after=10)
    for line in [
        "同一张图，可能被缩到 80px 藏在海报角落，也可能被旋转 30°、压成低质量 JPEG",
        "目标本身是固定图案，但背景千变万化 —— 不能靠「像不像照片」，只能靠「图案的局部纹理」",
        "要能框出位置，而不只是回答「有没有」",
    ]:
        para(tf, "· " + line, 12.5, False, INK, after=8, line=1.35)
    notes(s, "强调任务定义：检索 + 定位，不是分类。\n"
             "难点在于目标尺寸差异极大、变换多，且必须给出位置。")

    # ---- 3 两条路线 ----
    s = page(prs, "Approach", "两条技术路线，各管一段", 3, TOTAL)
    hdr = ["", "SIFT 特征匹配", "CNN 二分类"]
    body = [
        ("要不要训练", "不用，一张模板图就够", "要，但网页上点几下就行"),
        ("输出", "框出目标位置", "只回答「有 / 没有」"),
        ("应对旋转缩放", "天然不变", "要靠数据增强覆盖"),
        ("换一个目标", "换一张模板图", "要重新训练"),
        ("速度", "333 ms/张（8 进程）", "约 3 ms/张（GPU）"),
        ("适合", "精确定位、中小批量", "大批量粗筛"),
    ]
    tb = s.shapes.add_table(len(body) + 1, 3, Inches(0.72), Inches(1.65),
                            Inches(11.9), Inches(3.6)).table
    tb.columns[0].width = Inches(2.6)
    tb.columns[1].width = Inches(4.65)
    tb.columns[2].width = Inches(4.65)
    fill_table(tb, hdr, body, size=11.5, header_size=11.5)

    tf = textbox(s, Inches(0.72), Inches(5.55), Inches(11.9), Inches(1.3))
    para(tf, "为什么保留两条？", 13, True, BLUE, first=True, after=8)
    para(tf, "SIFT 负责「准」——不需要标注数据，换目标只换一张图，能给出精确外框；"
             "CNN 负责「快」——适合先在大批量图片上做粗筛。"
             "本项目把两条都实现了，并用实验说明各自的能力边界。",
         12.5, False, INK, after=0, line=1.4)
    notes(s, "这一页定调：不是二选一，而是分工。\n"
             "后面会先讲 SIFT 方案（主线），再讲 CNN 方案，最后用实验数据说明"
             "为什么「CNN 粗筛 + SIFT 精定位」这个看起来很美组合在本项目里没走通。")

    # ---- 4 SIFT 原理 ----
    s = page(prs, "Method", "SIFT 方案：五步定位", 4, TOTAL)
    steps = [
        ("① 模板特征", "从一张干净的\n目标图提取 SIFT\n关键点与描述子", BLUE),
        ("② 多尺度提取", "待检图在 1.0× /\n1.7× / 2.6× 上\n各提取一次", RGBColor(0x5E, 0x5C, 0xE6)),
        ("③ 匹配筛选", "FLANN KNN 匹配\n+ Lowe ratio 0.72\n过滤掉弱匹配", AMBER),
        ("④ 单应性估计", "RANSAC 求变换矩阵\n把模板四角投影\n到待检图上", RGBColor(0xE8, 0x5D, 0x2A)),
        ("⑤ 几何校验", "内点≥12、内点比例\n≥0.26、长宽比\n0.45~1.45", GREEN),
    ]
    bw = Inches(2.28)
    gap = Inches(0.13)
    for i, (t, d, color) in enumerate(steps):
        x = Inches(0.72) + i * (bw + gap)
        rect(s, x, Inches(1.7), bw, Inches(1.95), LIGHT,
             MSO_SHAPE.ROUNDED_RECTANGLE, 0.06)
        rect(s, x, Inches(1.7), bw, Inches(0.42), color)
        tf = textbox(s, x + Inches(0.12), Inches(1.79), bw - Inches(0.2), Inches(0.3))
        para(tf, t, 11.5, True, WHITE, first=True, after=0)
        tf = textbox(s, x + Inches(0.14), Inches(2.25), bw - Inches(0.28), Inches(1.3))
        para(tf, d, 10.5, False, INK, first=True, after=0, line=1.35)
        if i < 4:
            ar = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW,
                                    x + bw - Inches(0.02), Inches(2.52),
                                    Inches(0.17), Inches(0.34))
            ar.fill.solid()
            ar.fill.fore_color.rgb = RGBColor(0xC7, 0xD4, 0xE2)
            ar.line.fill.background()
            ar.shadow.inherit = False

    tf = textbox(s, Inches(0.72), Inches(4.45), Inches(11.9), Inches(2.4))
    para(tf, "为什么不需要训练", 13, True, BLUE, first=True, after=8)
    para(tf, "SIFT 描述子本身对旋转、尺度、光照变化不变，因此「模板 ↔ 待检图」可以直接比对。"
             "RANSAC 单应性进一步保证：只有那些位置关系自洽的匹配点才算数，"
             "零散误匹配会被直接淘汰。",
         12.5, False, INK, after=12, line=1.4)
    para(tf, "代价：SIFT 是纯 CPU 的稠密运算，没有 GPU 实现；且对极小目标（<40px）敏感。",
         12.5, True, RED, after=0, line=1.4)
    notes(s, "重点讲第③④步：ratio 筛选 + RANSAC 是鲁棒性的来源。\n"
             "把「为什么不用训练」讲清楚——这是 SIFT 路线最大的优势，也是它的局限"
             "（模板固定，无法识别「任意 logo」）。")

    # ---- 5 模板制作 ----
    s = page(prs, "Setup", "模板：从官方标准图裁出来", 5, TOTAL)
    pic(s, DOCS / "template_check.png", Inches(0.6), Inches(1.7), Inches(7.2), Inches(4.5))
    tf = textbox(s, Inches(8.1), Inches(1.85), Inches(4.6), Inches(4.5))
    para(tf, "处理流程", 13, True, BLUE, first=True, after=10)
    for line in [
        "白底阈值分割，取最大连通域 → 自动裁出图案",
        "放大 2.2 倍（原图是 JPEG，边缘太软）",
        "反锐化掩模（unsharp mask）提升边缘对比",
        "同时保存带 alpha 的版本，用于合成训练数据",
    ]:
        para(tf, "· " + line, 11.5, False, INK, after=9, line=1.35)
    para(tf, "结果", 13, True, BLUE, before=8, after=8)
    for k, v in [("模板尺寸", "337 × 436 px"), ("SIFT 关键点", "2072 个"),
                 ("长宽比", "0.773")]:
        para(tf, f"{k}：{v}", 11.5, False, SUB, after=6)
    para(tf, "锐化很关键：不做这一步，关键点数量会少一半，"
             "小尺寸目标的召回会明显下降。", 11.5, True, AMBER, before=8, after=0,
         line=1.35)
    notes(s, "这页说明模板质量直接决定上限。\n"
             "左图是模板和它的 SIFT 关键点可视化——可以看到关键点集中在盾形轮廓、"
             "中间徽标、文字这些高对比区域。")

    # ---- 6 数据集 ----
    s = page(prs, "Dataset", "三组数据，各管一件事", 6, TOTAL)
    data_rows = [
        ("合成评测集", "70 张", "40 正 + 30 负，含旋转 ±35°、透视、缩放 12~55%、模糊、JPEG 压缩",
         "量准确率/召回率"),
        ("真实素材集", "58 张", "从学校官网抓的新闻图、海报、校园照、二维码",
         "验证真实场景"),
        ("二分类训练池", "2400 张", "贴目标的局部块 1200 + 随机块 1200（30% 带模糊小目标干扰）",
         "训练 CNN"),
    ]
    y = Inches(1.7)
    for name, cnt, desc, use in data_rows:
        rect(s, Inches(0.72), y, Inches(11.9), Inches(1.35), LIGHT,
             MSO_SHAPE.ROUNDED_RECTANGLE, 0.05)
        tf = textbox(s, Inches(1.0), y + Inches(0.2), Inches(2.4), Inches(0.9))
        para(tf, name, 14, True, INK, first=True, after=4)
        para(tf, cnt, 12, True, BLUE)
        tf = textbox(s, Inches(3.5), y + Inches(0.28), Inches(6.6), Inches(0.9))
        para(tf, desc, 11, False, SUB, first=True, after=0, line=1.35)
        tf = textbox(s, Inches(10.4), y + Inches(0.45), Inches(2.0), Inches(0.5))
        para(tf, use, 11.5, True, GREEN, first=True, after=0,
             align=PP_ALIGN.RIGHT)
        y += Inches(1.5)

    tf = textbox(s, Inches(0.72), Inches(6.15), Inches(11.9), Inches(0.8))
    para(tf, "负样本里故意放了最容易混淆的东西：同一套视觉体系的周年标识"
             "（同样的红色、同样的校名文字、同样的轮廓），用来检验会不会误报。",
         11.5, False, INK, first=True, after=0, line=1.35)
    notes(s, "强调评测集是「合成但可控」：可以精确控制旋转角度、缩放比例，"
             "从而知道检测器在哪一档开始失效。\n"
             "真实素材集则是「不可控但真实」，用来验证结论不是合成数据上的自嗨。")

    # ---- 7 主结果 ----
    s = page(prs, "Result 1", "合成集：召回 100%，精确率 100%", 7, TOTAL)
    pic(s, DOCS / "detection_grid.png", Inches(0.5), Inches(1.55), Inches(12.3), Inches(4.0))
    kpis = [("100%", "召回率", GREEN), ("100%", "精确率", GREEN),
            ("0", "漏检 / 误报", SUB), ("27 组", "参数网格搜索", BLUE)]
    for i, (v, k, c) in enumerate(kpis):
        x = Inches(0.72 + i * 3.05)
        rect(s, x, Inches(5.75), Inches(2.85), Inches(1.05), LIGHT,
             MSO_SHAPE.ROUNDED_RECTANGLE, 0.08)
        tf = textbox(s, x + Inches(0.25), Inches(5.9), Inches(2.4), Inches(0.45))
        para(tf, v, 22, True, c, first=True, after=0)
        tf = textbox(s, x + Inches(0.25), Inches(6.38), Inches(2.4), Inches(0.3))
        para(tf, k, 10.5, False, SUB, first=True, after=0)
    notes(s, "上图是检测结果网格：绿框=正确检出，橙色框=漏检，红框=误报。"
             "这一页没有错例。\n"
             "参数不是手调的，是 27 组网格搜索选出来的（下一页）。")

    # ---- 8 调参 ----
    s = page(prs, "Result 2", "阈值不是拍出来的：27 组网格搜索", 8, TOTAL)
    tf = textbox(s, Inches(0.72), Inches(1.5), Inches(11.9), Inches(0.4))
    para(tf, "扫了 ratio × min_inliers × min_inlier_ratio 三维共 27 组，取 F1 最高的组合。",
         13, False, SUB, first=True, after=0)
    hdr = ["ratio", "内点≥", "内点比例≥", "召回", "精确率", "F1"]
    body = [
        ("0.72", "9", "0.18", "100%", "97.6%", "98.8%"),
        ("0.72", "11", "0.24", "100%", "100%", "100% ← 采用"),
        ("0.72", "14", "0.30", "100%", "100%", "100%"),
        ("0.75", "9", "0.24", "97.5%", "100%", "98.7%"),
        ("0.75", "14", "0.30", "97.5%", "100%", "98.7%"),
        ("0.78", "9", "0.18", "97.5%", "95.1%", "96.3%"),
        ("0.78", "14", "0.24", "97.5%", "100%", "98.7%"),
    ]
    tb = s.shapes.add_table(len(body) + 1, 6, Inches(0.72), Inches(2.05),
                            Inches(11.9), Inches(3.5)).table
    fill_table(tb, hdr, body, size=11)

    tf = textbox(s, Inches(0.72), Inches(5.85), Inches(11.9), Inches(1.1))
    para(tf, "观察", 13, True, BLUE, first=True, after=8)
    para(tf, "ratio 越松（0.78）误报越多，越紧（0.72）召回越稳；内点阈值从 9 提到 11 "
             "就消掉了唯一的误报。最终取 ratio=0.72 / 内点≥12 / 内点比例≥0.26，"
             "在留出余量的同时保持 100%。",
         12.5, False, INK, after=0, line=1.4)
    notes(s, "说明参数是用实验定的，不是凭感觉。\n"
             "重点：内点阈值只提了 2 个，就把唯一的误报消掉了——说明误报和正例的内点数"
             "本身有区分度，只是刚好卡在边界。")

    # ---- 9 真实场景 ----
    s = page(prs, "Result 3", "真实素材：藏在海报角落里也能找到", 9, TOTAL)
    pic(s, DOCS / "ui_scan.png", Inches(0.6), Inches(1.5), Inches(7.0), Inches(5.2))
    tf = textbox(s, Inches(7.9), Inches(1.7), Inches(4.8), Inches(5.0))
    para(tf, "58 张真实图片，8 张含目标", 13, True, BLUE, first=True, after=12)
    for line in [
        "会议海报：目标藏在一排 8 个合作 logo 的第 3 个，尺寸约 90×115 px",
        "周年活动图：目标在左上角，与同体系的周年标识并排出现",
        "二维码图：目标嵌在二维码中央，被严重遮挡",
        "校园风景、纯色底图、纹理背景：全部正确排除",
    ]:
        para(tf, "· " + line, 11.5, False, INK, after=11, line=1.35)
    para(tf, "真实素材 5/5 判断正确（另做单张验证）", 12.5, True, GREEN,
         before=6, after=10)
    para(tf, "对比：合成集上 100% 不代表真实场景 100%，"
             "但这一组的实际表现说明方法可用。", 11.5, False, SUB, after=0,
         line=1.35)
    notes(s, "这一页是「可信度」页：合成数据自己造的，容易自嗨；\n"
             "真实官网上抓的图片里，目标往往只有 90px 宽、被压缩、被其他 logo 包围，"
             "这才是真实难度。")

    # ---- 10 CNN 方案 ----
    s = page(prs, "Method", "CNN 方案：可视化调参的训练页", 10, TOTAL)
    pic(s, DOCS / "ui_train.png", Inches(0.55), Inches(1.55), Inches(8.1), Inches(4.9))
    tf = textbox(s, Inches(8.9), Inches(1.75), Inches(3.8), Inches(4.8))
    para(tf, "能调什么", 13, True, BLUE, first=True, after=10)
    for line in ["每类样本数 100~1200", "训练轮数 3~40", "学习率 1e-4~1e-2（对数）",
                 "批大小 16~128", "增强强度 0~1", "输入尺寸 64/96/128"]:
        para(tf, "· " + line, 11, False, INK, after=7)
    para(tf, "默认参数结果", 13, True, BLUE, before=8, after=8)
    for k, v in [("测试准确率", "96.5%"), ("召回率", "99.3%"),
                 ("训练耗时", "约 13 秒（RTX 5070）")]:
        para(tf, f"{k}：{v}", 11.5, False, SUB, after=6)
    para(tf, "后端在 GPU 上训练，前端每秒轮询，曲线逐 epoch 长出来。",
         11, True, AMBER, before=6, after=0, line=1.35)
    notes(s, "这一页展示「可交互」的部分：拖滑块 → 点开始 → 看曲线。\n"
             "左侧是参数，右侧是实时曲线和结果指标。默认配置 13 秒跑完。")

    # ---- 11 关键发现 ----
    s = page(prs, "Finding", "关键发现：CNN 粗筛这条路走不通", 11, TOTAL)
    tf = textbox(s, Inches(0.72), Inches(1.5), Inches(11.9), Inches(0.5))
    para(tf, "本来设想「CNN 先粗筛、SIFT 再精定位」，实测发现 CNN 在整图上召回只有 57%。",
         14, False, INK, first=True, after=0)
    hdr = ["样本", "P(有目标)", "判定", "说明"]
    body = [
        ("周年活动图 ×4", "0.901", "✓ 正确", "目标在画面中占比较大"),
        ("会议海报（8 logo 排排站）", "0.187", "✗ 漏检", "目标只占画面 3%"),
        ("校园活动图（角落）", "0.108", "✗ 漏检", "同上"),
        ("小尺寸目标图", "0.014", "✗ 漏检", "缩到 96px 后几乎看不见"),
        ("红底周年图（不含目标）", "0.963", "✗ 误报", "颜色/版式相似骗过了模型"),
    ]
    tb = s.shapes.add_table(len(body) + 1, 4, Inches(0.72), Inches(2.05),
                            Inches(11.9), Inches(3.3)).table
    tb.columns[0].width = Inches(4.4)
    tb.columns[1].width = Inches(2.0)
    tb.columns[2].width = Inches(1.8)
    tb.columns[3].width = Inches(3.7)
    fill_table(tb, hdr, body, size=11)

    tf = textbox(s, Inches(0.72), Inches(5.6), Inches(11.9), Inches(1.4))
    para(tf, "为什么", 13, True, BLUE, first=True, after=8)
    para(tf, "训练集的正样本是「紧贴目标裁下来的局部块」，目标占画面 30~60%；"
             "而真实图片整图缩到 96×96 后，目标只占 3~5%。训练分布和推理分布不一致。",
         12.5, False, INK, after=10, line=1.4)
    para(tf, "要救它得改成滑窗推理（切块 + 批量送 GPU），那是另一个量级的工程 —— "
             "所以本项目最终没有采用「CNN 粗筛」这条路。",
         12.5, True, RED, after=0, line=1.4)
    notes(s, "这一页是汇报里最有价值的「负结果」。\n"
             "很多方案在纸面上很漂亮（CNN 3ms 粗筛 + SIFT 精定位），但实测数据直接否掉了。\n"
             "诚实报告负结果比硬吹方案更可信。")

    # ---- 12 性能优化 ----
    s = page(prs, "Engineering", "三个把速度做上去的决定", 12, TOTAL)
    items = [
        ("纯 CPU 任务，用多进程而不是 GPU",
         "SIFT 在 OpenCV 里没有 CUDA 实现，而它本身是单线程的 —— 32 核只用了 1 核。"
         "改成 8 进程后：118 张从 118 秒降到 39 秒，命中数一致。", BLUE),
        ("nfeatures 必须设 0（不限）",
         "设成 3000 时 SIFT 按响应值截断，在并列值上取舍不稳定，同一张图在主进程和"
         "子进程内点数能差 5~10 个，边缘样本判定会翻转。改后 1 进程与 8 进程结果完全对齐。", AMBER),
        ("中间图片别过度压缩",
         "示例图存成 JPEG quality 88 时，7 张含目标的图有 4 张掉到内点阈值以下被漏检；"
         "换成 quality 97 后全部检出。", RED),
    ]
    y = Inches(1.65)
    for t, d, c in items:
        rect(s, Inches(0.72), y, Inches(11.9), Inches(1.5), LIGHT,
             MSO_SHAPE.ROUNDED_RECTANGLE, 0.05)
        rect(s, Inches(0.72), y, Inches(0.08), Inches(1.5), c)
        tf = textbox(s, Inches(1.05), y + Inches(0.2), Inches(11.2), Inches(0.4))
        para(tf, t, 14, True, c, first=True, after=0)
        tf = textbox(s, Inches(1.05), y + Inches(0.72), Inches(11.2), Inches(0.7))
        para(tf, d, 11.5, False, INK, first=True, after=0, line=1.35)
        y += Inches(1.65)

    tf = textbox(s, Inches(0.72), Inches(6.65), Inches(11.9), Inches(0.5))
    para(tf, "结论：这个任务的瓶颈是特征匹配的纯 CPU 运算，加多进程比折腾 GPU 收益大得多。",
         12.5, True, BLUE, first=True, after=0)
    notes(s, "这三条都是踩坑踩出来的，不是教科书结论。\n"
             "第 2 条最隐蔽：结果不稳定时很难想到是「特征点数量上限」导致的。")

    # ---- 13 产品形态 ----
    s = page(prs, "Product", "最终形态：一个本地网页工具", 13, TOTAL)
    feats = [
        ("扫描页", "填文件夹路径 → 逐张检测 → 命中的排前面、绿框标位置\n支持 ?dir=路径 直接打开结果", BLUE),
        ("训练页", "六个参数滑块 + 实时曲线 + 结果指标卡片\n训练完自动保存模型", GREEN),
        ("命令行", "python src/scan.py D:\\photos --crop\n输出 CSV + 标注图 + 裁剪结果", AMBER),
        ("可替换", "换目标只需替换 assets 里的模板图，代码零改动", RGBColor(0x5E, 0x5C, 0xE6)),
    ]
    for i, (t, d, c) in enumerate(feats):
        x = Inches(0.72 + (i % 2) * 6.05)
        yy = Inches(1.7 + (i // 2) * 2.3)
        rect(s, x, yy, Inches(5.85), Inches(2.05), LIGHT,
             MSO_SHAPE.ROUNDED_RECTANGLE, 0.06)
        tf = textbox(s, x + Inches(0.3), yy + Inches(0.22), Inches(5.2), Inches(0.4))
        para(tf, t, 14, True, c, first=True, after=0)
        tf = textbox(s, x + Inches(0.3), yy + Inches(0.78), Inches(5.3), Inches(1.1))
        para(tf, d, 11.5, False, INK, first=True, after=0, line=1.45)

    tf = textbox(s, Inches(0.72), Inches(6.3), Inches(11.9), Inches(0.8))
    para(tf, "技术栈：Python 3.11 · OpenCV 4.11（SIFT/FLANN/RANSAC）· PyTorch 2.7（CNN 训练）"
             "· 标准库 http.server（零 Web 框架依赖）",
         11.5, False, SUB, first=True, after=0)
    notes(s, "产品形态这页快速带过，重点说「可替换」——换目标只换一张图，"
             "这是 SIFT 路线最大的工程价值。")

    # ---- 14 工程结构 ----
    s = page(prs, "Codebase", "代码结构", 14, TOTAL)
    tree = [
        ("src/detector.py", "SIFT + 单应性检测器（核心，约 200 行）"),
        ("src/parallel_scan.py", "多进程并行扫描"),
        ("src/scan.py", "批量扫描 CLI（CSV / 标注图 / 裁剪）"),
        ("src/app.py", "网页版：扫描 + 训练双页"),
        ("src/trainer.py", "CNN 训练器（供网页调参）"),
        ("src/extract_template.py", "从标准图裁出模板"),
        ("src/make_testset.py", "合成评测集（贴图 + 变换）"),
        ("src/evaluate.py + sweep.py", "评测与参数网格搜索"),
        ("src/crawl_samples.py + classify_samples.py", "抓真实图并自动分类"),
        ("src/build_train_pool.py", "合成二分类训练池"),
    ]
    y = Inches(1.65)
    for name, desc in tree:
        tf = textbox(s, Inches(0.72), y, Inches(5.0), Inches(0.32))
        para(tf, name, 11.5, True, BLUE, first=True, after=0)
        tf = textbox(s, Inches(5.9), y, Inches(6.7), Inches(0.32))
        para(tf, desc, 11, False, SUB, first=True, after=0)
        y += Inches(0.5)

    tf = textbox(s, Inches(0.72), Inches(6.55), Inches(11.9), Inches(0.5))
    para(tf, "每个脚本都能单独跑，串起来就是完整流水线；数据与模型不入库，可由脚本再生成。",
         11.5, False, INK, first=True, after=0)
    notes(s, "结构这页可以快速过，重点说「每个脚本都能单独跑」。")

    # ---- 15 开源信息 ----
    s = page(prs, "Open Source", "开源信息", 15, TOTAL)
    tf = textbox(s, Inches(0.72), Inches(1.6), Inches(11.9), Inches(0.55))
    para(tf, "github.com/ikunchumo/logo-hunter", 30, True, BLUE, first=True,
         after=0)
    tf = textbox(s, Inches(0.72), Inches(2.2), Inches(11.9), Inches(0.4))
    para(tf, "MIT License · 公开仓库 · 91 个文件 · 16.6 MB", 13, False, SUB,
         first=True, after=0)

    # 左：仓库里有什么
    rect(s, Inches(0.72), Inches(2.85), Inches(5.85), Inches(3.35), LIGHT,
         MSO_SHAPE.ROUNDED_RECTANGLE, 0.06)
    tf = textbox(s, Inches(1.0), Inches(3.02), Inches(5.3), Inches(0.4))
    para(tf, "仓库里有什么", 13.5, True, INK, first=True, after=0)
    tf = textbox(s, Inches(1.0), Inches(3.5), Inches(5.35), Inches(2.5))
    for line in ["13 个脚本：检测器 / 并行扫描 / 网页 / 训练 / 评测 / 调参",
                 "58 张真实样本 + 检测模板（含 alpha 版）",
                 "16 页汇报 PPT（带逐页讲稿备注）",
                 "大文件不入库，跑脚本可再生成",
                 "完整 README：原理、复现、局限、FAQ"]:
        para(tf, "· " + line, 11, False, SUB, first=True if line.startswith("13") else False,
             after=8, line=1.3)

    # 右：三条命令
    rect(s, Inches(6.77), Inches(2.85), Inches(5.85), Inches(3.35), LIGHT,
         MSO_SHAPE.ROUNDED_RECTANGLE, 0.06)
    tf = textbox(s, Inches(7.05), Inches(3.02), Inches(5.3), Inches(0.4))
    para(tf, "三条命令跑起来", 13.5, True, INK, first=True, after=0)
    tf = textbox(s, Inches(7.05), Inches(3.55), Inches(5.35), Inches(2.4))
    for i, (cmd, col) in enumerate([
        ("pip install -r requirements.txt", SUB),
        ("python src/scan.py D:\\photos", BLUE),
        ("python src/app.py  →  :8770", GREEN),
    ]):
        para(tf, cmd, 11.5, True, col, first=(i == 0), after=12,
             line=1.3)
    para(tf, "换目标：替换 assets/ 里的模板图，代码零改动。",
         11, False, AMBER, before=4, after=0, line=1.3)

    tf = textbox(s, Inches(0.72), Inches(6.45), Inches(11.9), Inches(0.5))
    para(tf, "代码、数据、汇报材料一并开源；欢迎 Issue / PR。",
         12.5, True, BLUE, first=True, after=0)
    notes(s, "这一页给听众「去哪找代码」。\n"
             "强调三点：MIT 协议可自由使用；大文件不入库但脚本能再生成；"
             "换目标只需替换一张模板图。")

    # ---- 16 局限 ----
    s = page(prs, "Limitation", "局限与下一步", 16, TOTAL)
    left = [
        ("极小目标", "小于 40px 容易漏，需加大上采样尺度"),
        ("单色剪影", "纹理丢失，需额外做一版剪影模板"),
        ("严重遮挡", "挡掉一半以上会漏，需降低内点阈值"),
    ]
    right = [
        ("滑窗 CNN", "重新设计正样本比例，把 CNN 变成可用的粗筛"),
        ("多模板", "支持多张模板（彩色/单色/反白）取最大"),
        ("批量工具化", "加进度条、断点续扫、结果去重"),
    ]
    for col, items, color, title in ((0.72, left, RED, "已知局限"),
                                     (6.75, right, GREEN, "下一步")):
        tf = textbox(s, Inches(col), Inches(1.65), Inches(5.7), Inches(0.4))
        para(tf, title, 14, True, color, first=True, after=0)
        y = Inches(2.2)
        for t, d in items:
            rect(s, Inches(col), y, Inches(5.85), Inches(1.25), LIGHT,
                 MSO_SHAPE.ROUNDED_RECTANGLE, 0.06)
            tf = textbox(s, Inches(col + 0.25), y + Inches(0.15), Inches(5.3), Inches(0.35))
            para(tf, t, 12.5, True, INK, first=True, after=0)
            tf = textbox(s, Inches(col + 0.25), y + Inches(0.6), Inches(5.35), Inches(0.55))
            para(tf, d, 11, False, SUB, first=True, after=0, line=1.3)
            y += Inches(1.4)

    tf = textbox(s, Inches(0.72), Inches(6.5), Inches(11.9), Inches(0.6))
    para(tf, "最重要的局限：评测集是合成的。真实图库的分布更杂，"
             "上线前建议先用几百张真实图确认阈值。", 12, True, AMBER,
         first=True, after=0, line=1.35)
    notes(s, "主动讲局限。特别强调「合成评测集」这个根本局限，"
             "以及换到真实图库前要先做小规模验证。")

    # ---- 16 结尾 ----
    s = prs.slides.add_slide(prs.slide_layouts[6])
    rect(s, 0, 0, W, H, DARK)
    tf = textbox(s, Inches(1.35), Inches(2.85), Inches(10.5), Inches(1.2))
    para(tf, "谢谢，欢迎提问", 40, True, WHITE, first=True, after=10)
    tf = textbox(s, Inches(1.35), Inches(4.25), Inches(10.5), Inches(1.4))
    para(tf, "开源仓库：github.com/ikunchumo/logo-hunter   （MIT License）", 13, False,
         RGBColor(0x9F, 0xB0, 0xC4), first=True, after=6)
    para(tf, "python src/app.py  →  http://127.0.0.1:8770", 12.5, False,
         RGBColor(0x8A, 0xC4, 0xFF), after=0)
    notes(s, "预留提问时间。可能被问到：\n"
             "1) 为什么不用 YOLO？—— 固定目标用特征匹配更划算，换目标只换一张图；"
             "YOLO 要标注几百张带框的图。\n"
             "2) 合成评测集可信吗？—— 可控但不真实，所以我们另外做了真实素材验证。\n"
             "3) 能不能识别多个不同 logo？—— 当前是单模板，多模板要改判定逻辑。\n"
             "4) 为什么 CNN 召回这么低？—— 训练分布与推理分布不一致（见第 11 页）。")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(f"已生成 {OUT}  共 {len(prs.slides._sldIdLst)} 页")


if __name__ == "__main__":
    build()
