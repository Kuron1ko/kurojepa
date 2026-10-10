"""Render the editable Markdown handout to a typeset Chinese PDF.

Run with the Codex bundled Python, which includes reportlab.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import HRFlowable, KeepTogether, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ijepa_assignment.md"
OUTPUT = ROOT / "ijepa_assignment.pdf"

pdfmetrics.registerFont(TTFont("BodyCN", r"C:\Windows\Fonts\msyh.ttc", subfontIndex=0))
pdfmetrics.registerFont(TTFont("HeadCN", r"C:\Windows\Fonts\simhei.ttf"))

NAVY = colors.HexColor("#172C42")
TEAL = colors.HexColor("#167B86")
MUTED = colors.HexColor("#5E6C7A")
PALE = colors.HexColor("#F1F6F8")

styles = {
    "title": ParagraphStyle("title", fontName="HeadCN", fontSize=19, leading=26, textColor=NAVY, spaceAfter=9, wordWrap="CJK"),
    "h2": ParagraphStyle("h2", fontName="HeadCN", fontSize=13, leading=18, textColor=NAVY, spaceBefore=13, spaceAfter=6, keepWithNext=True, wordWrap="CJK"),
    "h3": ParagraphStyle("h3", fontName="HeadCN", fontSize=10.8, leading=16, textColor=TEAL, spaceBefore=9, spaceAfter=4, keepWithNext=True, wordWrap="CJK"),
    "body": ParagraphStyle("body", fontName="BodyCN", fontSize=9.3, leading=14.2, textColor=NAVY, spaceAfter=6, wordWrap="CJK", alignment=TA_LEFT),
    "list": ParagraphStyle("list", fontName="BodyCN", fontSize=9.3, leading=14.2, textColor=NAVY, leftIndent=16, firstLineIndent=-12, spaceAfter=3, wordWrap="CJK"),
    "meta": ParagraphStyle("meta", fontName="BodyCN", fontSize=8.7, leading=14, textColor=MUTED, spaceAfter=3, wordWrap="CJK"),
    "code": ParagraphStyle("code", fontName="Courier", fontSize=8, leading=11, leftIndent=12, spaceAfter=7),
    "cell": ParagraphStyle("cell", fontName="BodyCN", fontSize=8.5, leading=13, textColor=NAVY, wordWrap="CJK"),
}


def inline_markup(raw: str) -> str:
    value = html.escape(raw.strip())
    value = re.sub(r"\*\*(.+?)\*\*", r'<font name="HeadCN">\1</font>', value)
    value = re.sub(r"\*(.+?)\*", r"\1", value)
    value = re.sub(r"`([^`]+)`", r'<font color="#167B86">\1</font>', value)
    return value


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D5E1E6"))
    canvas.line(48, 45, letter[0] - 48, 45)
    canvas.setFont("BodyCN", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(48, 30, "I-JEPA 从零实现 · 学习作业")
    canvas.drawRightString(letter[0] - 48, 30, str(doc.page))
    canvas.restoreState()


def parse(lines):
    story = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
            continue
        if line.startswith("# "):
            story.append(Paragraph(inline_markup(line[2:]), styles["title"]))
            story.append(HRFlowable(width="100%", thickness=1.2, color=TEAL, spaceAfter=12))
            i += 1
            continue
        if line.startswith("## ") or line.startswith("### "):
            level = "h2" if line.startswith("## ") else "h3"
            story.append(Paragraph(inline_markup(line.lstrip("# ")), styles[level]))
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                parts = [part.strip() for part in lines[i].strip().strip("|").split("|")]
                if not all(set(part) <= {"-", ":", " "} for part in parts):
                    rows.append([Paragraph(inline_markup(part), styles["cell"]) for part in parts])
                i += 1
            widths = [142, 366] if len(rows[0]) == 2 else [118, 282, 108]
            table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), PALE),
                ("LINEBELOW", (0, 0), (-1, 0), 0.7, TEAL),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAFCFD")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.extend([Spacer(1, 5), table, Spacer(1, 8)])
            continue
        if line.startswith("- ") or re.match(r"^\d+\. ", line):
            story.append(Paragraph(inline_markup(line), styles["list"]))
            i += 1
            continue
        paragraph = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#|\||- |\d+\. )", lines[i]):
            paragraph.append(lines[i].strip())
            i += 1
        combined = " ".join(paragraph).replace("  ", " ")
        kind = "meta" if combined.startswith("**对象**") else "body"
        story.append(Paragraph(inline_markup(combined), styles[kind]))
    return story


def main():
    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=letter, leftMargin=45, rightMargin=45,
        topMargin=44, bottomMargin=50, title="I-JEPA 从零实现：学习指导书",
        author="I-JEPA learning assignment",
    )
    story = parse(SOURCE.read_text(encoding="utf-8").splitlines())
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


if __name__ == "__main__":
    main()
