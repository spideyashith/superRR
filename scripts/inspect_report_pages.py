"""Render every page of the final report PDF to a PNG for visual inspection.

Also reports per-page metrics that make layout defects easy to spot without
opening 16 images: text coverage, empty-page detection, and any page whose
content runs past the bottom margin.

Run:  py scripts/inspect_report_pages.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "outputs" / "Methodology_Report_Final.pdf"
OUT = ROOT / "outputs" / "report_review"

DPI = 110
PAGE_H = 29.7 / 2.54 * 72                      # A4 height in points
BOTTOM_LIMIT_PT = PAGE_H - 1.9 / 2.54 * 72     # content box bottom
FOOTER_TOP = PAGE_H - 1.0 / 2.54 * 72          # anything below this is furniture


def main() -> int:
    if not PDF.exists():
        print(f"missing {PDF}")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("page_*.png"):
        old.unlink()

    doc = pymupdf.open(PDF)
    print(f"PDF: {PDF.relative_to(ROOT)}  pages={doc.page_count}")
    print(f"{'page':>4}  {'chars':>6}  {'blocks':>6}  {'maxY':>7}  {'overflow':>8}  image")
    problems = []
    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(dpi=DPI)
        png = OUT / f"page_{i:02d}.png"
        pix.save(png)
        text = page.get_text().strip()
        blocks = page.get_text("blocks")
        content = [b for b in blocks if b[3] < FOOTER_TOP]
        max_y = max((b[3] for b in content), default=0.0)
        overflow = max_y > BOTTOM_LIMIT_PT + 1
        flag = "OVERFLOW" if overflow else ""
        if overflow:
            problems.append(i)
        if len(text) < 40:
            problems.append(i)
            flag = (flag + " NEAR-EMPTY").strip()
        print(f"{i:>4}  {len(text):>6}  {len(blocks):>6}  {max_y:>7.1f}  {flag:>8}  {png.name}")

    if problems:
        print(f"\npages needing attention: {sorted(set(problems))}")
    else:
        print("\nno page overflowed the bottom margin or was near-empty")
    return 0


if __name__ == "__main__":
    sys.exit(main())
