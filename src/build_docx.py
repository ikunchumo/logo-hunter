"""
build_docx.py — 生成 Word 版汇报稿（含专业名词解释）。

产物：docs/汇报稿.docx
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "汇报稿.docx"

CN = "微软雅黑"
CN_BODY = "等线"
INK = RGBColor(0x1D, 0x1D, 0x1F)
SUB = RGBColor(0x6E, 0x6E, 0x73)
BLUE = RGBColor(0x00, 0x71, 0xE3)
ACCENT = RGBColor(0x0F, 0x2A, 0x44)


# ------------------------------------------------------------------ 基础工具
def set_font(run, size=10.5, bold=False, color=INK, font=CN_BODY):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = font
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    rFonts.set(qn("w:eastAsia"), font)
    rFonts.set(qn("w:ascii"), font)
    rFonts.set(qn("w:hAnsi"), font)


def para(doc, text="", size=10.5, bold=False, color=INK, align=None,
         before=0, after=6, indent=None, line=1.5, font=CN_BODY, style=None):
    p = doc.add_paragraph(style=style) if style else doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = line
    if indent is not None:
        pf.first_line_indent = Cm(indent)
    if align is not None:
        p.alignment = align
    if text:
        r = p.add_run(text)
        set_font(r, size, bold, color, font)
    return p


def heading(doc, text, level=1):
    sizes = {1: 16, 2: 13, 3: 11.5}
    space = {1: (16, 10), 2: (14, 8), 3: (10, 6)}
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(space[level][0])
    p.paragraph_format.space_after = Pt(space[level][1])
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    set_font(r, sizes[level], True, ACCENT if level == 1 else INK, CN)
    if level == 1:
        # 标题下加一条细线
        pPr = p._p.get_or_add_pPr()
        pbdr = OxmlElement("w:pBdr")
        bottom = OxmlElement("w:bottom")
        bottom.set(qn("w:val"), "single")
        bottom.set(qn("w:sz"), "8")
        bottom.set(qn("w:space"), "4")
        bottom.set(qn("w:color"), "0071E3")
        pbdr.append(bottom)
        pPr.append(pbdr)
    return p


def bullet(doc, text, size=10.5, bold_prefix=None):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.45
    if bold_prefix:
        r = p.add_run(bold_prefix)
        set_font(r, size, True, INK)
    r = p.add_run(text)
    set_font(r, size, False, INK)
    return p


def code_block(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.6)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.line_spacing = 1.3
    r = p.add_run(text)
    set_font(r, 9.5, False, RGBColor(0x2B, 0x2B, 0x2B), "Consolas")
    # 浅灰底
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), "F5F5F7")
    pPr.append(shd)
    return p


def add_page_number(section):
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    for tag, txt in (("begin", None), (None, "PAGE"), ("end", None)):
        if tag:
            fld = OxmlElement("w:fldChar")
            fld.set(qn("w:fldCharType"), tag)
            run._r.append(fld)
        else:
            instr = OxmlElement("w:instrText")
            instr.set(qn("xml:space"), "preserve")
            instr.text = " PAGE "
            run._r.append(instr)
    set_font(run, 9, False, SUB, CN)


def shade(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def make_table(doc, header, rows, widths, font_size=9, header_fill="0F2A44"):
    t = doc.add_table(rows=len(rows) + 1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for i, w in enumerate(widths):
        for row in t.rows:
            row.cells[i].width = Cm(w)
    # 表头
    for j, h in enumerate(header):
        c = t.cell(0, j)
        c.text = ""
        p = c.paragraphs[0]
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.space_before = Pt(2)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        set_font(r, font_size + 0.5, True, RGBColor(0xFF, 0xFF, 0xFF), CN)
        shade(c, header_fill)
        c.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    # 正文
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.text = ""
            p = c.paragraphs[0]
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.line_spacing = 1.25
            r = p.add_run(str(val))
            set_font(r, font_size, j == 0, INK if j == 0 else RGBColor(0x3A, 0x3A, 0x3C))
            c.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            if i % 2 == 0:
                shade(c, "F7F7F9")
    return t


def page_break(doc):
    doc.add_page_break()


# ------------------------------------------------------------------ 内容
def build():
    doc = Document()

    # 页面设置
    sec = doc.sections[0]
    sec.page_width = Cm(21)
    sec.page_height = Cm(29.7)
    sec.top_margin = Cm(2.5)
    sec.bottom_margin = Cm(2.3)
    sec.left_margin = Cm(2.6)
    sec.right_margin = Cm(2.6)
    add_page_number(sec)

    # 默认样式
    normal = doc.styles["Normal"]
    normal.font.name = CN_BODY
    normal.font.size = Pt(10.5)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), CN_BODY)

    # ============ 封面 ============
    for _ in range(5):
        para(doc, "", after=0)
    p = para(doc, "图片目标检索项目", 30, True, ACCENT, WD_ALIGN_PARAGRAPH.CENTER,
             after=6, font=CN)
    p = para(doc, "汇 报 稿", 30, True, ACCENT, WD_ALIGN_PARAGRAPH.CENTER,
             after=18, font=CN)
    para(doc, "SIFT 特征匹配 + 可视化训练页", 13, False, BLUE,
         WD_ALIGN_PARAGRAPH.CENTER, after=4, font=CN)
    para(doc, "github.com/ikunchumo/logo-hunter", 11, False, SUB,
         WD_ALIGN_PARAGRAPH.CENTER, after=40, font=CN)

    info = doc.add_table(rows=4, cols=2)
    info.alignment = WD_TABLE_ALIGNMENT.CENTER
    rows = [("汇 报 人", "郑斯元"),
            ("汇报对象", "卓工协会 培训部"),
            ("预计时长", "13 – 15 分钟（正文 12 分钟 + 提问 2~3 分钟）"),
            ("配套材料", "PPT 17 页 · 网页演示 http://127.0.0.1:8770")]
    for i, (k, v) in enumerate(rows):
        c0, c1 = info.cell(i, 0), info.cell(i, 1)
        c0.width, c1.width = Cm(3.2), Cm(9.5)
        c0.text = ""
        r = c0.paragraphs[0].add_run(k)
        set_font(r, 10.5, True, SUB, CN)
        c0.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
        c1.text = ""
        r = c1.paragraphs[0].add_run(v)
        set_font(r, 10.5, False, INK, CN)
    # 去掉封面表格边框
    tbl = info._tbl
    tblPr = tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "none")
        borders.append(el)
    tblPr.append(borders)

    page_break(doc)

    # ============ 使用说明 ============
    heading(doc, "使用说明", 1)
    para(doc, "这份文档是汇报的口播稿，按 PPT 页码组织。每节标题对应一页幻灯片，"
              "正文是可以直接念出来的话术，方括号里是动作提示。", after=8)
    para(doc, "总时长控制在 13~15 分钟：正文 12 分钟，留 2~3 分钟提问。"
              "如果时间被压缩，按下面的顺序删减页面。", after=10)

    heading(doc, "时间分配", 2)
    make_table(doc,
               ["PPT 页", "内容", "用时", "可删"],
               [["1", "封面", "0:20", "—"],
                ["2", "问题定义", "1:00", "—"],
                ["3", "两条技术路线", "1:00", "—"],
                ["4", "SIFT 五步流程", "1:40", "核心"],
                ["5", "模板制作", "0:50", "④"],
                ["6", "三组数据", "0:50", "—"],
                ["7", "主结果 100%", "1:20", "核心"],
                ["8", "参数网格搜索", "0:50", "②"],
                ["9", "真实素材验证", "1:00", "—"],
                ["10", "CNN 训练页", "1:00", "—"],
                ["11", "关键发现（失败方案）", "1:40", "★ 不能删"],
                ["12", "三个提速决定", "1:10", "—"],
                ["13", "产品形态", "0:40", "③"],
                ["14", "代码结构", "0:30", "①"],
                ["15", "开源信息", "0:30", "—"],
                ["16", "局限与下一步", "0:50", "—"],
                ["17", "Q&A", "—", "—"]],
               [2.0, 6.0, 2.6, 3.2])

    para(doc, "删减顺序：① 代码结构 → ② 网格搜索 → ③ 产品形态 → ④ 模板制作。"
              "第 11 页无论如何要保留——它是整场最有信息量的部分。",
         10, False, SUB, before=8, after=6)

    page_break(doc)

    # ============ P1 ============
    heading(doc, "P1 · 封面（20 秒）", 1)
    para(doc, "各位老师、同学好。", indent=0.74)
    para(doc, "这次汇报的项目是图片目标检索：给一个装满各种图片的文件夹，"
              "自动挑出里面含有某个特定标识的那几张，并且在图上把它的位置框出来。",
         indent=0.74)
    para(doc, "技术上做了两条路线——一条不需要训练的 SIFT 特征匹配，一条可训练的 CNN 二分类，"
              "最后对比了两者各自的能力边界。", indent=0.74)

    # ============ P2 ============
    heading(doc, "P2 · 问题定义（1 分钟）", 1)
    para(doc, "先把任务说清楚：这不是「识别这是什么」，而是「在一堆无关图片里，把它挑出来」。",
         indent=0.74, bold=True)
    para(doc, "拆成三件事：", indent=0.74)
    bullet(doc, "一个文件夹，几百到几万张图，混着风景、海报、截图、二维码。", bold_prefix="输入：")
    bullet(doc, "一个固定图案，比如公司 logo、某个标识。它可能被旋转、被压缩、"
                "被贴在海报角落只有 90 像素宽。", bold_prefix="要找的：")
    bullet(doc, "命中的图片列表，并且要框出位置。", bold_prefix="输出：")

    para(doc, "难点在哪？同一张图，可能被缩到 80 像素藏在角落，也可能被旋转 30 度、"
              "压成低质量 JPEG。目标本身是固定图案，但背景千变万化——所以不能靠"
              "「像不像照片」来判断，只能靠图案本身的局部纹理。而且我们要的是位置，"
              "不只是「有没有」。", indent=0.74, before=4)

    # ============ P3 ============
    heading(doc, "P3 · 两条技术路线（1 分钟）", 1)
    para(doc, "针对这个任务，我实现并对比了两套方案。", indent=0.74)
    para(doc, "左边是 SIFT 特征匹配：不需要训练，只要一张模板图。它能直接框出位置，"
              "对旋转缩放天然不变，换一个目标只要换一张模板图。代价是慢——纯 CPU 运算，"
              "每张 300 多毫秒。", indent=0.74)
    para(doc, "右边是 CNN 二分类：需要训练，但训练只要十几秒。它只回答「有还是没有」，"
              "不告诉你位置。速度极快，GPU 上 3 毫秒一张。换目标就要重新训练。", indent=0.74)
    para(doc, "我一开始的想法很自然：CNN 先粗筛、SIFT 再精定位——快的过滤掉大部分，"
              "慢的只处理可疑的。这个组合在纸面上非常漂亮。", indent=0.74)
    para(doc, "但第 11 页会告诉大家，这个想法在我们的场景里没有走通，以及为什么。",
         indent=0.74, bold=True, color=BLUE)

    page_break(doc)

    # ============ P4 ============
    heading(doc, "P4 · SIFT 五步流程（1 分 40 秒）", 1)
    para(doc, "SIFT 这套流程是项目的主线，分五步。", indent=0.74)

    for i, (t, d) in enumerate([
        ("第一步，模板特征。",
         "从一张干净的目标图里提取 SIFT 关键点和描述子。关键点是图像里「有辨识度」的位置"
         "——角点、边缘拐弯处；描述子是这个点周围一小块区域的纹理「指纹」，128 维向量。"
         "一张模板图能提出 2000 多个这样的指纹。"),
        ("第二步，多尺度提取。",
         "待检图在三个尺度上各提取一次：原尺寸、放大 1.7 倍、放大 2.6 倍。为什么要放大？"
         "因为目标可能只有 90 像素宽，在原图上提不出足够的关键点，放大之后细节才够。"),
        ("第三步，匹配筛选。",
         "用 FLANN 做最近邻匹配，每个模板点找两个最像的图上的点，然后只保留"
         "「第一个明显比第二个像」的——这就是 Lowe ratio，阈值取 0.72。"
         "这一步能甩掉大量似是而非的匹配。"),
        ("第四步，单应性估计。",
         "剩下的匹配点里还是有错的。用 RANSAC 求一个 3×3 的变换矩阵，它描述"
         "「模板平面怎么旋转缩放平移到图上」。RANSAC 的做法是反复随机抽几对点算矩阵，"
         "看有多少点符合——符合的点叫内点，不符合的叫外点。抽足够多次，取内点最多的那个矩阵。"),
        ("第五步，几何校验。",
         "只有内点数够多（≥12）、内点占匹配数的比例够高（≥26%）、投影出来的框长宽比合理，"
         "才判定命中。"),
    ], 1):
        para(doc, t + d, indent=0.74, after=6)

    para(doc, "为什么这套流程不需要训练？", indent=0.74, bold=True, before=6, after=4)
    para(doc, "因为 SIFT 描述子本身对旋转、尺度、光照就不变，模板和待检图可以直接比对；"
              "RANSAC 又保证了只有位置关系自洽的匹配才算数，零散的误匹配会被直接淘汰。",
         indent=0.74)
    para(doc, "代价也很明确：SIFT 是纯 CPU 的稠密运算，没有 GPU 实现，而且对极小目标敏感。",
         indent=0.74, bold=True, color=RGBColor(0xC0, 0x30, 0x25))

    # ============ P5 ============
    heading(doc, "P5 · 模板制作（50 秒）", 1)
    para(doc, "模板质量直接决定上限。", indent=0.74)
    para(doc, "流程是：白底阈值分割、取最大连通域、自动裁出图案；然后放大 2.2 倍，"
              "再做反锐化掩模提升边缘对比。", indent=0.74)
    para(doc, "为什么要锐化？因为官方给的是 JPEG，边缘很软。不锐化的话，关键点数量会少一半，"
              "小尺寸目标的召回会明显下降。", indent=0.74)
    para(doc, "左边这张图是模板和它的关键点可视化——可以看到关键点集中在轮廓、"
              "中间徽标、文字这些高对比区域，这正是我们想要的。", indent=0.74)
    para(doc, "最终模板 337×436 像素，2072 个关键点。", indent=0.74, bold=True)

    page_break(doc)

    # ============ P6 ============
    heading(doc, "P6 · 三组数据（50 秒）", 1)
    para(doc, "我准备了三组数据，各管一件事。", indent=0.74)
    bullet(doc, "40 张正样本、30 张负样本。正样本是把目标贴到不同背景上，带旋转 ±35 度、"
                "透视变形、缩放 12% 到 55%、模糊、JPEG 压缩。它能精确控制变量——"
                "我知道它是在哪一档开始失效的。", bold_prefix="合成评测集，70 张：")
    bullet(doc, "从官网抓的新闻图、海报、校园照、二维码。它不可控但真实——"
                "用来验证结论不是合成数据上的自嗨。", bold_prefix="真实素材集，58 张：")
    bullet(doc, "给 CNN 用的，贴目标的局部块 1200 张 + 随机块 1200 张。",
           bold_prefix="二分类训练池，2400 张：")
    para(doc, "这里有个细节值得说：负样本里我故意放了最容易混淆的东西——"
              "同一套视觉体系的相似标识，同样的颜色、同样的文字、同样的轮廓。"
              "就是要看它会不会误报。", indent=0.74, before=6, bold=True)

    # ============ P7 ============
    heading(doc, "P7 · 主结果（1 分 20 秒）", 1)
    para(doc, "这是合成集上的结果：召回 100%，精确率 100%，零漏检零误报。",
         indent=0.74, bold=True)
    para(doc, "上面这张网格图，绿色框是正确检出，橙色是漏检，红色是误报——这一页没有错例。",
         indent=0.74)
    para(doc, "可以看到目标在各种背景下都被准确框出来了：纯色底、纹理底、渐变底、"
         "旋转过的、斜的、藏在风景照里的。", indent=0.74)
    para(doc, "但我更想强调这个 100% 是怎么来的——不是手调的，是搜出来的。",
         indent=0.74, bold=True, color=BLUE)

    # ============ P8 ============
    heading(doc, "P8 · 参数网格搜索（50 秒）", 1)
    para(doc, "三个关键参数：ratio 阈值、内点数量下限、内点比例下限。我在三维空间里扫了 27 组。",
         indent=0.74)
    para(doc, "看这张表：ratio 放宽到 0.78，误报开始出现；收紧到 0.72，召回最稳。"
              "而内点阈值从 9 提到 11，就把唯一的那个误报消掉了。", indent=0.74)
    para(doc, "有意思的是：只提了 2 个内点，误报就没了。说明误报和正例的内点数本身是有区分度的，"
              "只是刚好卡在边界上。", indent=0.74, bold=True)
    para(doc, "最终取 ratio=0.72、内点≥12、内点比例≥26%，在留出余量的同时保持 100%。",
         indent=0.74)

    page_break(doc)

    # ============ P9 ============
    heading(doc, "P9 · 真实素材验证（1 分钟）", 1)
    para(doc, "合成集的 100% 不能说明全部问题，所以我又在真实图片上验了一遍。", indent=0.74)
    para(doc, "58 张真实图片，8 张含目标。几个典型例子：", indent=0.74)
    bullet(doc, "目标藏在一排 8 个合作 logo 的第 3 个，尺寸大约 90×115 像素。", bold_prefix="会议海报：")
    bullet(doc, "目标在左上角，和同体系的相似标识并排出现。", bold_prefix="周年活动图：")
    bullet(doc, "目标嵌在二维码中央，被严重遮挡。", bold_prefix="二维码图：")
    bullet(doc, "全部正确排除。", bold_prefix="校园风景、纯色底图、纹理背景：")
    para(doc, "另外单独做了 5 张标准素材的单张验证，5 张全部判断正确。", indent=0.74, before=4)
    para(doc, "这组数据说明：方法在真实场景是可用的，不只是合成数据上好看。",
         indent=0.74, bold=True)

    # ============ P10 ============
    heading(doc, "P10 · CNN 训练页（1 分钟）", 1)
    para(doc, "第二条路线是 CNN 二分类。我把它做成了网页上可交互的形式。", indent=0.74)
    para(doc, "左边六个参数可以拖：每类样本数、训练轮数、学习率、批大小、增强强度、"
              "输入尺寸。点「开始训练」，右边实时画出 loss 和验证准确率的曲线，"
              "跑完给出测试准确率、精确率、召回率。", indent=0.74)
    para(doc, "默认配置——每类 600 张、14 轮——在 RTX 5070 上 13 秒跑完，"
              "测试准确率 96.5%，召回 99.3%。", indent=0.74, bold=True)
    para(doc, "这个页面本身也是个教学工具：比如把每类样本数拖到 100，准确率会直接崩到 55%，"
              "召回 0——等于全判「没有」。把学习率压到 1e-4，只有 81%。", indent=0.74)

    page_break(doc)

    # ============ P11 ============
    heading(doc, "P11 · 关键发现：CNN 粗筛走不通（1 分 40 秒）", 1)
    para(doc, "这一页是整场汇报我最想讲的。", indent=0.74)
    para(doc, "原本的设想是：CNN 用 3 毫秒粗筛，通过的再交给 SIFT 精定位。"
              "这样大部分图片 3 毫秒就排除了，整体能快十倍以上。", indent=0.74)
    para(doc, "但实测数据显示，CNN 在整图上的召回只有 57%。", indent=0.74, bold=True,
         color=RGBColor(0xC0, 0x30, 0x25))
    para(doc, "看这张表：周年活动图 4 张，目标占画面比较大，模型给 0.901，判断正确。"
              "但会议海报——目标只占画面 3%——只有 0.187，漏检。校园活动图目标在角落，"
              "0.108，漏检。小尺寸目标更是只有 0.014。反过来，一张不含目标的红底周年图，"
              "模型给了 0.963，误报。", indent=0.74)
    para(doc, "为什么？", indent=0.74, bold=True, before=6, after=4)
    para(doc, "根因很清楚：训练集的正样本是「紧贴目标裁下来的局部块」，目标占画面 30% 到 60%；"
              "但推理时整图缩到 96×96，目标只占 3% 到 5%。训练分布和推理分布不一致——"
              "模型学的是「画面里有一大块这个图案」，而不是「画面某处藏了一个小图案」。",
         indent=0.74)
    para(doc, "要救它得改成滑窗推理：把图切成 96×96 的块，批量送 GPU 推理。"
              "那是另一个量级的工程——要处理几万块的调度、要重新设计正样本比例、"
              "要做后处理合并框。", indent=0.74)
    para(doc, "所以这个项目最终没有采用「CNN 粗筛」这条路。", indent=0.74, bold=True)
    para(doc, "我把这个负结果放在汇报里，是因为它比「我又实现了一个模块」更有信息量："
              "一个方案在纸面上漂亮，不代表它在真实数据分布下能工作。",
         indent=0.74, bold=True, color=BLUE)

    page_break(doc)

    # ============ P12 ============
    heading(doc, "P12 · 三个提速决定（1 分 10 秒）", 1)
    para(doc, "三个把速度做上去的决定，都是踩坑踩出来的。", indent=0.74)

    para(doc, "第一，纯 CPU 任务要用多进程，不是 GPU。", indent=0.74, bold=True, before=4, after=2)
    para(doc, "SIFT 在 OpenCV 里没有 CUDA 实现，而它本身是单线程的——你 32 个核只用了 1 个。"
              "改成 8 进程之后，118 张图从 118 秒降到 39 秒，3 倍加速，命中数完全一致。",
         indent=0.74)

    para(doc, "第二，nfeatures 必须设成 0，也就是不限。", indent=0.74, bold=True, before=4, after=2)
    para(doc, "这个坑很隐蔽：设成 3000 的时候，SIFT 会按响应值截断关键点。在并列响应值上"
              "取舍不稳定，同一张图在主进程和子进程里内点数能差 5 到 10 个，边缘样本的判定"
              "就会翻转。表现就是「单进程和多进程结果不一样」。改成不限之后，1 进程和 8 进程"
              "的结果完全对齐。", indent=0.74)

    para(doc, "第三，中间图片别过度压缩。", indent=0.74, bold=True, before=4, after=2)
    para(doc, "示例图存成 JPEG quality 88 的时候，7 张含目标的图有 4 张掉到内点阈值以下被漏检；"
              "换成 quality 97 后全部检出。压缩损失的是细节，而特征匹配吃的就是细节。",
         indent=0.74)

    para(doc, "结论：这个任务的瓶颈是特征匹配的纯 CPU 运算，不是 GPU 算力。"
              "加多进程比折腾 GPU 收益大得多。", indent=0.74, before=6, bold=True, color=BLUE)

    # ============ P13 ============
    heading(doc, "P13 · 产品形态（40 秒）", 1)
    para(doc, "最终做成一个本地网页工具，两个页面。", indent=0.74)
    bullet(doc, "填文件夹路径，逐张检测，命中的排前面、绿框标位置。", bold_prefix="扫描页：")
    bullet(doc, "就是刚才那六个滑块加实时曲线。", bold_prefix="训练页：")
    bullet(doc, "支持批量扫描、输出 CSV、画标注图、裁剪目标。", bold_prefix="命令行版本：")
    para(doc, "最大的工程价值是「可替换」：换一个目标，只需替换 assets 里的模板图，代码零改动。",
         indent=0.74, bold=True)

    page_break(doc)

    # ============ P14 ============
    heading(doc, "P14 · 代码结构（30 秒）", 1)
    para(doc, "代码分四层：", indent=0.74)
    bullet(doc, "detector.py，SIFT 加强制性验证，约 200 行。", bold_prefix="核心：")
    bullet(doc, "parallel_scan.py，多进程扫描。", bold_prefix="并行：")
    bullet(doc, "scan.py 命令行、app.py 网页。", bold_prefix="产品：")
    bullet(doc, "模板提取、评测集合成、参数搜索、真实图抓取分类、训练池生成。", bold_prefix="支撑：")
    para(doc, "每个脚本都能单独跑，串起来就是完整流水线。", indent=0.74, before=4, bold=True)

    # ============ P15 ============
    heading(doc, "P15 · 开源信息（30 秒）", 1)
    para(doc, "项目已经开源在 GitHub：github.com/ikunchumo/logo-hunter，MIT 协议。", indent=0.74)
    para(doc, "仓库里是 91 个文件：13 个脚本、58 张真实样本、完整文档。"
              "大文件不入库，但跑脚本都能再生成。", indent=0.74)
    para(doc, "三条命令就能跑起来：装依赖、命令行扫描、启动网页。欢迎 Issue 和 PR。", indent=0.74)

    # ============ P16 ============
    heading(doc, "P16 · 局限与下一步（50 秒）", 1)
    para(doc, "说几个已知的局限：", indent=0.74)
    bullet(doc, "小于 40 像素容易漏，需要加大上采样尺度。", bold_prefix="极小目标：")
    bullet(doc, "纹理丢了，需要额外做一版剪影模板。", bold_prefix="单色剪影：")
    bullet(doc, "挡掉一半以上会漏，需要降低内点阈值、接受更多误检。", bold_prefix="严重遮挡：")
    para(doc, "下一步三个方向：滑窗 CNN（重新设计正样本比例）、多模板融合、批量工具化。",
         indent=0.74, before=4)
    para(doc, "最重要的局限是：评测集是合成的。真实图库的分布更杂，上线前建议先用几百张"
              "真实图确认阈值。", indent=0.74, bold=True,
         color=RGBColor(0xC0, 0x30, 0x25))

    # ============ P17 ============
    heading(doc, "P17 · Q&A（10 秒）", 1)
    para(doc, "谢谢，欢迎提问。", indent=0.74, bold=True)

    page_break(doc)

    # ============ 附录 A：专业名词 ============
    heading(doc, "附录 A · 专业名词解释", 1)
    para(doc, "按 PPT 出现顺序排列。表格第二列是通用定义，第三列是它在本项目里的具体含义"
              "——光看定义容易懂，但不知道「到底指什么」，所以两列都给。", after=10)

    terms = [
        ("SIFT", "尺度不变特征变换。一种手工设计的图像特征提取算法，对旋转、缩放、亮度变化不敏感。",
         "整个检测器的核心。模板和目标图都提 SIFT 特征，然后比对。"),
        ("关键点", "图像里「有辨识度」的位置：角点、边缘拐弯、纹理突变处。",
         "一张模板能提出 2072 个关键点。"),
        ("描述子", "关键点周围一小块区域的纹理「指纹」，通常是一个向量。两个描述子越接近，纹理越像。",
         "SIFT 描述子是 128 维浮点向量。"),
        ("尺度空间", "同一个图案在不同放大倍率下看，会呈现不同特征。多尺度就是准备多个放大倍率的版本。",
         "待检图在 1.0×、1.7×、2.6× 三个尺度上各提一次特征。"),
        ("Lowe ratio test", "匹配时每个点找最近的两个候选，只有「第一名比第二名明显更像」才认。用来过滤模棱两可的匹配。",
         "阈值 0.72，由网格搜索选出。"),
        ("FLANN", "Fast Library for Approximate Nearest Neighbors，快速近似最近邻库。在大量向量里快速找最像的几个。",
         "用来做模板描述子和图片描述子之间的匹配。"),
        ("KNN 匹配", "K-Nearest Neighbors，找最近的 K 个。这里 K=2，每个点找两个最像的。",
         "配合 ratio test 使用。"),
        ("单应性矩阵", "一个 3×3 矩阵，描述「一个平面怎么变换成另一个平面」，含旋转、缩放、平移、透视，8 个自由度。",
         "用来把模板四角投影到待检图上，从而得到目标框。"),
        ("RANSAC", "随机抽样一致。反复随机抽最少的点算模型，统计有多少点符合，取符合最多的那个。抗噪声能力强。",
         "用来从一堆（有错的）匹配点里求出正确的单应性矩阵。"),
        ("内点 / 外点", "符合模型的数据点叫内点，不符合的叫外点（噪声、误匹配）。",
         "判定核心指标：内点数 ≥12 才算命中。"),
        ("内点比例", "内点数 ÷ 参与拟合的匹配总数，反映匹配的「纯净度」。",
         "要求 ≥26%，防止一堆乱匹配碰巧凑出矩阵。"),
        ("重投影误差", "用求出的变换矩阵把点投影回去，和实际位置的偏差。RANSAC 用它判断算不算内点。",
         "阈值 5 像素。"),
        ("IoU", "Intersection over Union，交并比。两框重叠面积 ÷ 合并面积，衡量框得准不准。通常 ≥0.5 算正确。",
         "评测时取 IoU≥0.25（目标小，标准放宽）。"),
        ("召回率", "真正有目标的图片里，被找出来的比例。衡量漏了多少。",
         "本项目 100%，一张没漏。"),
        ("精确率", "被判为「有目标」的图片里，真的有目标的比例。衡量误报多少。",
         "本项目 100%，一张没错报。"),
        ("F1", "精确率和召回率的调和平均，一个综合指标。",
         "网格搜索就是按 F1 排序选参数的。"),
        ("正样本 / 负样本", "正样本 = 含目标的图；负样本 = 不含目标的图。",
         "评测集 40 正 30 负；训练池各 1200 张。"),
        ("数据增强", "人为地对训练图做变换（旋转、缩放、调色、擦除），让模型见过更多变化，减少过拟合。",
         "训练页的「增强强度」滑块控制这个。"),
        ("连通域", "图像里连成一片的像素区域。取最大连通域 = 找到最大的那个图形块。",
         "用来从白底图里自动裁出模板。"),
        ("反锐化掩模", "一种锐化方法：原图减去模糊版得到「边缘图」，再加回原图，边缘就清晰了。",
         "提升模板边缘对比，让 SIFT 能提出更多关键点。"),
        ("多进程", "同时开多个进程分摊任务。Python 的 GIL 让多线程跑不满多核，所以计算密集型任务要用多进程。",
         "8 进程扫描，118 张从 118 秒降到 39 秒。"),
        ("GIL", "Global Interpreter Lock，全局解释器锁。同一时刻只允许一个线程执行 Python 字节码。",
         "这是为什么用多进程而不是多线程。"),
        ("CNN", "卷积神经网络，能自动从图像里学特征的网络结构。",
         "二分类方案用它判断「有没有目标」。"),
        ("BN", "批归一化。把每层输出标准化，让训练更稳更快。",
         "小数据训练时通常固定它，避免统计量被带偏。"),
        ("OneCycleLR", "一种学习率调度策略：先升后降，中途达到最大值。比固定学习率收敛更快。",
         "训练页用的就是这个。"),
        ("训练分布 / 推理分布", "训练时喂给模型的数据是什么样，推理时实际遇到的数据是什么样。两者不一致，模型就会失效。",
         "第 11 页那个失败方案的根因。"),
        ("滑窗推理", "把大图切成很多小块，逐块送模型判断，再合并结果。用来检测大图里的小目标。",
         "CNN 粗筛方案要救活就得改成这样。"),
    ]
    make_table(doc, ["名词", "通用解释", "在本项目里"], [list(t) for t in terms],
               [3.4, 7.2, 5.6], font_size=8.5)

    page_break(doc)

    # ============ 附录 B：Q&A ============
    heading(doc, "附录 B · 预判问题与答法", 1)
    qa = [
        ("为什么不用 YOLO / 目标检测网络？",
         "目标检测网络需要几百张带标注框的训练图，而且换一个 logo 就得重新标、重新训。"
         "我们的场景是「目标固定、图库很大」，用特征匹配更划算——换目标只换一张模板图。"),
        ("合成评测集 100%，真实场景也能 100% 吗？",
         "不能保证。合成集只能证明算法在这些变换下不失效；真实集 8/8 检出只能说明「可用」，"
         "不能说明「完美」。上线前应该用几百张真实图确认阈值。"),
        ("为什么内点阈值是 12 而不是 10 或 15？",
         "网格搜索的结果。11 就能消掉唯一误报，取 12 是留一点余量。再往上会开始伤召回。"),
        ("CNN 那部分是不是白做了？",
         "不是。它的价值在于证明了这条路在当前数据分布下不可行，而且给出了明确的修复方向"
         "（滑窗 + 重新设计正样本比例）。如果没做这个对比，我们可能会一直在「SIFT 太慢」"
         "这个问题上打转，而不知道有更快的方案、也不知道它为什么不行。"),
        ("能不能同时找多个不同的 logo？",
         "当前是单模板设计。多模板要改判定逻辑：每个模板各跑一遍，然后做 NMS"
         "（非极大值抑制）合并重叠框。工作量中等。"),
        ("SIFT 是专利算法吗？",
         "SIFT 的专利在 2020 年已经过期，OpenCV 4.x 把它放进了主模块。可以自由商用。"),
        ("为什么扫描不用 GPU？",
         "SIFT 在 OpenCV 里没有 CUDA 实现，CUDA 模块只有 FAST、ORB 这一档。"
         "而且真正的瓶颈是「单线程只用 1 个核」，多进程比 GPU 更直接有效。"),
        ("这个项目最难的部分是什么？",
         "不是写检测器，是搞清楚「为什么结果不稳定」。nfeatures 那个坑花了不少时间——"
         "单进程和多进程结果不一致，最后定位到是特征点数量上限导致的取舍不稳定。"),
    ]
    for i, (q, a) in enumerate(qa, 1):
        p = para(doc, f"Q{i}：{q}", 10.5, True, INK, before=8, after=3)
        para(doc, "A：" + a, 10.5, False, RGBColor(0x3A, 0x3A, 0x3C), indent=0.0, after=4)

    page_break(doc)

    # ============ 附录 C：演示建议 ============
    heading(doc, "附录 C · 现场演示建议", 1)
    para(doc, "面向卓工协会培训部，听众可能偏工程实践，建议多讲实现细节与踩坑过程，""少讲公式推导。如果现场能开电脑，强烈建议做一次真实演示——比任何截图都有说服力。", after=8)

    heading(doc, "准备", 2)
    code_block(doc, "cd D:\\project\\logo-hunter\n"
                    "D:\\project\\RVC\\.venv\\Scripts\\python.exe src\\app.py")
    para(doc, "提前把服务跑起来，浏览器打开 http://127.0.0.1:8770，"
              "确认页面正常再开始讲。", indent=0.74)

    heading(doc, "三个演示时机", 2)
    bullet(doc, "讲之前先扫一遍 data\\samples（有缓存会秒出），让观众看到真实的检测过程——"
                "58 张图里 8 张被绿框标出来。", bold_prefix="讲 P7 之前：")
    bullet(doc, "切到训练页，现场拖一下滑块、点开始，13 秒后出结果。"
                "可以故意把每类样本数拖到 100，让观众看到准确率掉到 55%。", bold_prefix="讲 P10 时：")
    bullet(doc, "跑一条 scan.py 命令，展示 CSV 和标注图输出。", bold_prefix="如果有时间：")

    heading(doc, "如果只能放 PPT", 2)
    para(doc, "按时间分配讲即可，第 7、9、10、11 页各留 2 秒让观众看图。"
              "第 7 页的网格图上，绿色框是检出、橙色是漏检、红色是误报——"
              "指出这一点观众会更容易看懂。", indent=0.74)

    # 结尾页
    para(doc, "", after=20)
    p = para(doc, "— 全文完 —", 11, False, SUB, WD_ALIGN_PARAGRAPH.CENTER, font=CN)
    p = para(doc, "github.com/ikunchumo/logo-hunter", 10, False, BLUE,
             WD_ALIGN_PARAGRAPH.CENTER, font=CN)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(f"已生成 {OUT}")
    print(f"段落数 {len(doc.paragraphs)}，表格 {len(doc.tables)} 个")


if __name__ == "__main__":
    build()
