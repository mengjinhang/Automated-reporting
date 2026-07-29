#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Markdown 报告 → PDF（基于 ReportLab，跨平台）。

用法:
    python md_to_pdf.py output/report_20260729_1641.md
    python md_to_pdf.py output/report_20260729_1641.md -o output/my_report.pdf
"""
import argparse
import os
import re
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
    HRFlowable, KeepTogether, Preformatted,
)

# ---------------------------------------------------------------------------
# 字体注册（跨平台：宋体正文 + 黑体标题）
# ---------------------------------------------------------------------------
_FONT_REGISTERED = False
_FONT_NAME = "Helvetica"
_FONT_NAME_B = "Helvetica-Bold"

_CJK_FONT_PAIRS = [
    # Windows
    ("C:/Windows/Fonts/simsun.ttc", "SimSun", 0,
     "C:/Windows/Fonts/simhei.ttf", "SimHei", None),
    ("C:/Windows/Fonts/simsun.ttc", "SimSun", 0,
     "C:/Windows/Fonts/msyh.ttc", "MSYH", 0),
    # macOS
    ("/System/Library/Fonts/Supplemental/Songti.ttc", "SongtiSC", 3,
     "/System/Library/Fonts/STHeiti Medium.ttc", "HeitiSC", 1),
    # Linux
    ("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc", "NotoSerifCJK", 0,
     "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "NotoSansCJK", 0),
    ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", "WQYZenHei", 0,
     "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", "WQYZenHeiB", 0),
]


def _try_register(path, name, subfont_index):
    if not os.path.exists(path):
        return False
    try:
        kw = {"subfontIndex": subfont_index} if subfont_index is not None else {}
        pdfmetrics.registerFont(TTFont(name, path, **kw))
        return True
    except Exception:
        return False


def _register_fonts():
    global _FONT_REGISTERED, _FONT_NAME, _FONT_NAME_B
    if _FONT_REGISTERED:
        return
    for r_path, r_name, r_idx, b_path, b_name, b_idx in _CJK_FONT_PAIRS:
        if _try_register(r_path, r_name, r_idx) and _try_register(b_path, b_name, b_idx):
            _FONT_NAME = r_name
            _FONT_NAME_B = b_name
            break
    _FONT_REGISTERED = True


# ---------------------------------------------------------------------------
# 特殊符号回退到黑体
# ---------------------------------------------------------------------------
_SYMBOL_RE = re.compile(
    r"([─━│┃┄┅┆┇┈┉┊┋╌╍╎╏═║╔╗╚╝╠╣╦╩╬▶▷◀◁◄▲△▼▽→←↑↓↔↕⇒⇐⇑⇓⇔"
    r"★☆●○◆◇■□▪▫•‣⟶⟵⟹⟸▸▹▻▾▿◂◃≥≤≠≈∞∑∏∫∂√±×÷⊂⊃⊆⊇∈∉∪∩∧∨¬∀∃∅∇∝∠℃℉‰※►✓✕]+)"
)


def _symbol_fallback(text):
    if _FONT_NAME == _FONT_NAME_B:
        return text
    return _SYMBOL_RE.sub(rf'<font face="{_FONT_NAME_B}">\1</font>', text)


# ---------------------------------------------------------------------------
# 样式
# ---------------------------------------------------------------------------
PAGE_W, PAGE_H = A4
MARGIN = 18 * mm

C_PRIMARY = colors.HexColor("#1a237e")
C_HEADER_BG = colors.HexColor("#1a237e")
C_HEADER_FG = colors.white
C_ROW_ALT = colors.HexColor("#f5f5f5")
C_BORDER = colors.HexColor("#bdbdbd")
C_LIGHT_BORDER = colors.HexColor("#e0e0e0")

SYSTEM_NAME = "传染病智能预警系统"


def _styles():
    _register_fonts()

    def _ps(name, **kw):
        kw.setdefault("fontName", _FONT_NAME)
        return ParagraphStyle(name, **kw)

    return {
        "title": _ps("T", fontSize=20, leading=26, alignment=TA_CENTER,
                      textColor=C_PRIMARY, spaceAfter=4, fontName=_FONT_NAME_B),
        "subtitle": _ps("ST", fontSize=10, leading=14, alignment=TA_CENTER,
                         textColor=colors.HexColor("#424242"), spaceAfter=10),
        "h2": _ps("H2", fontSize=14, leading=18, spaceBefore=10, spaceAfter=4,
                   textColor=C_PRIMARY, fontName=_FONT_NAME_B),
        "h3": _ps("H3", fontSize=12, leading=16, spaceBefore=6, spaceAfter=3,
                   textColor=colors.HexColor("#283593"), fontName=_FONT_NAME_B),
        "h4": _ps("H4", fontSize=10, leading=14, spaceBefore=8, spaceAfter=3,
                   textColor=colors.HexColor("#37474f"), fontName=_FONT_NAME_B),
        "body": _ps("B", fontSize=9, leading=13, alignment=TA_JUSTIFY,
                     firstLineIndent=18, wordWrap="CJK"),
        "bullet": _ps("BL", fontSize=9, leading=13, leftIndent=14, bulletIndent=4),
        "quote": _ps("Q", fontSize=8.5, leading=12, leftIndent=12,
                      textColor=colors.HexColor("#455a64"),
                      backColor=colors.HexColor("#f5f5f5")),
        "code": _ps("C", fontName=_FONT_NAME_B, fontSize=7.5, leading=10,
                     leftIndent=8, rightIndent=8, spaceBefore=4, spaceAfter=4,
                     backColor=colors.HexColor("#f5f5f5"),
                     textColor=colors.HexColor("#263238")),
        "caption": _ps("CAP", fontSize=8, leading=10, alignment=TA_CENTER,
                        textColor=colors.HexColor("#757575")),
        "cell": _ps("CELL", fontSize=8, leading=10),
        "cell_b": _ps("CELLB", fontSize=8, leading=10, fontName=_FONT_NAME_B),
        "cell_h": _ps("CELLH", fontSize=8, leading=10,
                       textColor=C_HEADER_FG, fontName=_FONT_NAME_B),
        "footer": _ps("FT", fontSize=7, leading=9, alignment=TA_CENTER,
                       textColor=colors.HexColor("#9e9e9e")),
        "timestamp": _ps("TS", fontSize=8, leading=10, alignment=TA_RIGHT,
                          textColor=colors.HexColor("#9e9e9e"), fontName=_FONT_NAME_B),
    }


# ---------------------------------------------------------------------------
# Markdown 行内标记 → ReportLab XML
# ---------------------------------------------------------------------------
def _md_inline(text):
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    codes = []

    def _stash(m):
        codes.append(m.group(1))
        return f"\x00{len(codes) - 1}\x00"

    text = re.sub(r"`([^`]+)`", _stash, text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(
        r"\x00(\d+)\x00",
        lambda m: f'<font color="#c62828">{codes[int(m.group(1))]}</font>',
        text,
    )
    return _symbol_fallback(text)


# ---------------------------------------------------------------------------
# 页眉/页脚
# ---------------------------------------------------------------------------
def _header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(_FONT_NAME, 7)
    canvas.setFillColor(colors.HexColor("#9e9e9e"))
    canvas.drawString(MARGIN, PAGE_H - 12 * mm, SYSTEM_NAME)
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 12 * mm, f"第 {doc.page} 页")
    canvas.setStrokeColor(C_LIGHT_BORDER)
    canvas.line(MARGIN, PAGE_H - 13 * mm, PAGE_W - MARGIN, PAGE_H - 13 * mm)
    canvas.setFont(_FONT_NAME, 6)
    canvas.drawCentredString(PAGE_W / 2, 10 * mm,
                             f"{SYSTEM_NAME} — 山西大学复杂系统研究所 & 西安交通大学生物数学团队")
    canvas.line(MARGIN, 14 * mm, PAGE_W - MARGIN, 14 * mm)
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Markdown → PDF
# ---------------------------------------------------------------------------
def export_md_pdf(md_path, out_path=None):
    _register_fonts()
    st = _styles()

    if out_path is None:
        out_path = os.path.splitext(md_path)[0] + ".pdf"

    md_dir = os.path.dirname(os.path.abspath(md_path))

    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=16 * mm, bottomMargin=18 * mm,
    )

    story = []
    i = 0
    n = len(lines)
    avail_w = PAGE_W - 2 * MARGIN

    while i < n:
        line = lines[i].rstrip("\n")
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # 水平线
        if stripped in ("---", "***", "___"):
            story.append(HRFlowable(width="100%", thickness=0.5, color=C_LIGHT_BORDER,
                                    spaceAfter=4, spaceBefore=4))
            i += 1
            continue

        # 标题
        if stripped.startswith("#"):
            m = re.match(r"^(#{1,4})\s+(.+)", stripped)
            if m:
                level = len(m.group(1))
                text = _md_inline(m.group(2))
                style_key = {1: "title", 2: "h2", 3: "h3", 4: "h4"}[level]
                story.append(Paragraph(text, st[style_key]))
                if level == 1:
                    story.append(HRFlowable(width="100%", thickness=0.8, color=C_PRIMARY,
                                            spaceAfter=6))
                i += 1
                continue

        # 图片 ![alt](path)
        img_match = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)", stripped)
        if img_match:
            alt_text = img_match.group(1)
            img_rel = img_match.group(2)
            img_abs = os.path.normpath(os.path.join(md_dir, img_rel))
            if os.path.exists(img_abs):
                from PIL import Image as PILImage
                with PILImage.open(img_abs) as pil_img:
                    pw, ph = pil_img.size
                max_h = (PAGE_H - 2 * MARGIN) * 0.33
                img_w = avail_w * 0.78
                img_h = img_w * (ph / pw)
                if img_h > max_h:
                    img_h = max_h
                    img_w = img_h * (pw / ph)
                story.append(Spacer(1, 1 * mm))
                story.append(Image(img_abs, width=img_w, height=img_h))
                if alt_text:
                    story.append(Paragraph(alt_text, st["caption"]))
                story.append(Spacer(1, 1 * mm))
            else:
                story.append(Paragraph(f"[图片缺失: {img_rel}]", st["caption"]))
            i += 1
            continue

        # 代码块
        if stripped.startswith("```"):
            code_lines = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i].rstrip("\n"))
                i += 1
            i += 1
            code_text = "\n".join(code_lines)
            code_text = code_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            story.append(Preformatted(code_text, st["code"]))
            continue

        # 表格
        if stripped.startswith("|"):
            table_lines = []
            while i < n and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            header_cells = [c.strip() for c in table_lines[0].split("|")[1:-1]]
            data_rows = []
            for tl in table_lines[1:]:
                if re.match(r"^\|[\s\-:|]+\|$", tl):
                    continue
                cells = [c.strip() for c in tl.split("|")[1:-1]]
                data_rows.append(cells)
            t_data = [[Paragraph(_md_inline(h), st["cell_h"]) for h in header_cells]]
            for row in data_rows:
                t_data.append([Paragraph(_md_inline(c), st["cell"]) for c in row])
            ncols = len(header_cells)
            all_cells = [header_cells] + data_rows
            col_max = [max(len(r[j]) if j < len(r) else 0
                           for r in all_cells) for j in range(ncols)]
            total_chars = sum(col_max) or 1
            col_ws = [max(c / total_chars, 0.08) for c in col_max]
            ws_sum = sum(col_ws)
            col_ws = [w / ws_sum * avail_w * 0.9 for w in col_ws]
            t = Table(t_data, colWidths=col_ws, repeatRows=1)
            t_style = [
                ("BACKGROUND", (0, 0), (-1, 0), C_HEADER_BG),
                ("TEXTCOLOR", (0, 0), (-1, 0), C_HEADER_FG),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("GRID", (0, 0), (-1, -1), 0.4, C_BORDER),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
            for ri in range(1, len(t_data)):
                if ri % 2 == 0:
                    t_style.append(("BACKGROUND", (0, ri), (-1, ri), C_ROW_ALT))
            t.setStyle(TableStyle(t_style))
            story.append(t)
            story.append(Spacer(1, 2 * mm))
            continue

        # 引用块
        if stripped.startswith(">"):
            quote_lines = []
            while i < n and lines[i].strip().startswith(">"):
                quote_lines.append(re.sub(r"^>\s*", "", lines[i].strip()))
                i += 1
            text = _md_inline(" ".join(quote_lines))
            story.append(Paragraph(text, st["quote"]))
            story.append(Spacer(1, 2 * mm))
            continue

        # 列表项
        if re.match(r"^[-*]\s", stripped) or re.match(r"^\d+\.\s", stripped):
            text = re.sub(r"^[-*]\s+", "", stripped)
            text = re.sub(r"^\d+\.\s+", "", text)
            story.append(Paragraph(f"• {_md_inline(text)}", st["bullet"]))
            i += 1
            continue

        # 报告生成时间（*报告生成时间：...*）
        ts_match = re.match(r"^\*(.+)\*$", stripped)
        if ts_match and "报告生成时间" in stripped:
            text = ts_match.group(1)
            story.append(Paragraph(_md_inline(text), st["timestamp"]))
            i += 1
            continue

        # 普通段落
        para_lines = [stripped]
        i += 1
        while i < n:
            nxt = lines[i].strip()
            if (not nxt or nxt.startswith("#") or nxt.startswith("|")
                    or nxt.startswith("```") or nxt.startswith(">")
                    or nxt.startswith("- ") or nxt.startswith("* ")
                    or re.match(r"^\d+\.\s", nxt)
                    or re.match(r"^!\[", nxt)
                    or nxt in ("---", "***", "___")):
                break
            para_lines.append(nxt)
            i += 1
        text = _md_inline(" ".join(para_lines))
        story.append(Paragraph(text, st["body"]))
        story.append(Spacer(1, 2 * mm))

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    sz = os.path.getsize(out_path)
    print(f"完成 → {out_path} ({sz:,} bytes)")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Markdown 报告转 PDF")
    ap.add_argument("input", help="Markdown 文件路径")
    ap.add_argument("-o", "--output", default=None, help="输出 PDF 路径（默认同名 .pdf）")
    args = ap.parse_args()
    export_md_pdf(args.input, args.output)
