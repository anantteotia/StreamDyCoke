"""
Build docs/FINAL_REPORT.pdf from docs/FINAL_REPORT_CANVAS.txt.

Requires: pip install reportlab

Usage (from repo root):
  python scripts/export_report_pdf.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _is_heading(line: str) -> bool:
    """Section titles only (not bibliography lines like '1. Tao, ...')."""
    s = line.strip()
    if not s:
        return False
    if s in ("Abstract", "References"):
        return True
    if re.match(r"^\d+\.\d+\s", s):
        return True
    if re.match(r"^\d+\.\s+[A-Za-z]", s) and len(s) < 72:
        return True
    return False


def main() -> int:
    root = _repo_root()
    src = root / "docs" / "FINAL_REPORT_CANVAS.txt"
    out = root / "docs" / "FINAL_REPORT.pdf"
    if not src.is_file():
        print(f"Missing source file: {src}", file=sys.stderr)
        return 1

    raw = src.read_text(encoding="utf-8")
    blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        fontSize=14,
        leading=18,
        spaceAfter=14,
        textColor=colors.HexColor("#111111"),
    )
    heading_style = ParagraphStyle(
        "HeadingCustom",
        parent=styles["Heading2"],
        fontSize=12,
        leading=15,
        spaceBefore=10,
        spaceAfter=8,
        textColor=colors.HexColor("#111111"),
    )
    body_style = ParagraphStyle(
        "BodyCustom",
        parent=styles["Normal"],
        fontSize=11,
        leading=14,
        alignment=TA_JUSTIFY,
        spaceAfter=8,
    )
    meta_style = ParagraphStyle(
        "MetaCustom",
        parent=styles["Normal"],
        fontSize=11,
        leading=14,
        spaceAfter=4,
    )

    story: list = []
    first = True
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        inner = "<br/>".join(escape(ln) for ln in lines)

        if first:
            story.append(Paragraph(inner, title_style))
            first = False
            continue

        if lines[0].startswith("Course:"):
            story.append(Paragraph(inner, meta_style))
            continue

        if _is_heading(lines[0]):
            story.append(Paragraph(inner, heading_style))
        else:
            story.append(Paragraph(inner, body_style))

    doc = SimpleDocTemplate(
        str(out),
        pagesize=letter,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title="Final Report: StreamDyCoke",
        author="Anant Teotia",
    )
    doc.build(story)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
