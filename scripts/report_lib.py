"""DOCX styling helpers for the SupeRR Selector methodology report.

Kept separate from the build script so the report content stays readable.
All colours and measurements follow docs/FINAL_REPORT_CONTENT_AND_DESIGN_BRIEF.md.
"""
from __future__ import annotations

from pathlib import Path

from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

# ---------------------------------------------------------------- palette
NAVY = RGBColor(0x12, 0x23, 0x3F)
PINK = RGBColor(0xD6, 0x33, 0x6C)
GREY = RGBColor(0x59, 0x59, 0x59)
LIGHT = RGBColor(0x8C, 0x8C, 0x8C)
NAVY_HEX = "12233F"
PINK_HEX = "D6336C"
ZEBRA_HEX = "F4F5F7"
RESERVE_HEX = "FCE9F0"

BODY_PT = 10.5
TABLE_PT = 8.0
NOTE_PT = 8.0


# ---------------------------------------------------------------- low level
# Children of w:tcPr must appear in schema order or Word silently drops them.
_AFTER_SHD = ("w:noWrap", "w:tcMar", "w:textDirection", "w:tcFitText",
              "w:vAlign", "w:hideMark")
_AFTER_BORDERS = ("w:shd",) + _AFTER_SHD


def shade(cell, hex_fill: str) -> None:
    """Fill a table cell with a solid colour."""
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tc_pr.insert_element_before(shd, *_AFTER_SHD)


def cell_borders(cell, edges=("top", "bottom"), sz=4, color=NAVY_HEX) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for edge in edges:
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), str(sz))
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), color)
        borders.append(el)
    tc_pr.insert_element_before(borders, *_AFTER_BORDERS)


def repeat_header(row) -> None:
    """Repeat a table header row on every page the table spans."""
    tr_pr = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    tr_pr.append(header)


def no_split(row) -> None:
    """Keep a row (and its wrapped name) from breaking across pages."""
    tr_pr = row._tr.get_or_add_trPr()
    cant = OxmlElement("w:cantSplit")
    cant.set(qn("w:val"), "true")
    tr_pr.append(cant)


def cell_margins(table, top=40, bottom=40, left=70, right=70) -> None:
    tbl_pr = table._tbl.tblPr
    mar = OxmlElement("w:tblCellMar")
    for edge, val in (("top", top), ("left", left), ("bottom", bottom), ("right", right)):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:w"), str(val))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    tbl_pr.append(mar)


def write_cell(cell, text: str, *, bold=False, align="left", size=TABLE_PT,
               color=None, italic=False) -> None:
    cell.text = ""
    para = cell.paragraphs[0]
    para.alignment = {"left": WD_ALIGN_PARAGRAPH.LEFT,
                      "right": WD_ALIGN_PARAGRAPH.RIGHT,
                      "center": WD_ALIGN_PARAGRAPH.CENTER}[align]
    para.paragraph_format.space_before = Pt(1)
    para.paragraph_format.space_after = Pt(1)
    para.paragraph_format.line_spacing = 1.0
    run = para.add_run(str(text))
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = "Calibri"
    if color is not None:
        run.font.color.rgb = color


def add_field(paragraph, instruction: str, size=8, color=GREY) -> None:
    """Insert a Word field (PAGE, NUMPAGES, TOC) into a paragraph."""
    run = paragraph.add_run()
    run.font.size = Pt(size)
    run.font.color.rgb = color
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = instruction
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    placeholder = OxmlElement("w:t")
    placeholder.text = "1"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    for el in (fld_begin, instr, fld_sep, placeholder, fld_end):
        run._r.append(el)


# ---------------------------------------------------------------- blocks
def h1(doc, number: str, title: str, *, page_break=True) -> None:
    if page_break:
        doc.add_page_break()
    # Real Heading 1 style so Word's table of contents picks the section up.
    para = doc.add_paragraph(style="Heading 1")
    para.paragraph_format.space_before = Pt(0)
    para.paragraph_format.space_after = Pt(2)
    para.paragraph_format.keep_with_next = True
    run = para.add_run(f"{number}")
    run.font.size = Pt(20)
    run.font.bold = True
    run.font.color.rgb = PINK
    run.font.name = "Calibri"
    run2 = para.add_run(f"   {title}")
    run2.font.size = Pt(16)
    run2.font.bold = True
    run2.font.color.rgb = NAVY
    run2.font.name = "Calibri"
    rule(doc, width=17.0, color=PINK_HEX, sz=8, space_after=10)


def h2(doc, title: str, *, space_before=12) -> None:
    para = doc.add_paragraph(style="Heading 2")
    para.paragraph_format.space_before = Pt(space_before)
    para.paragraph_format.space_after = Pt(3)
    para.paragraph_format.keep_with_next = True
    run = para.add_run(title)
    run.font.size = Pt(11.5)
    run.font.bold = True
    run.font.color.rgb = NAVY
    run.font.name = "Calibri"


def rule(doc, *, width=17.0, color=NAVY_HEX, sz=6, space_after=6) -> None:
    para = doc.add_paragraph()
    para.paragraph_format.space_before = Pt(0)
    para.paragraph_format.space_after = Pt(space_after)
    para._p.get_or_add_pPr().append(_bottom_border(color, sz, width))


def _bottom_border(color: str, sz: int, width_cm: float):
    p_pr = OxmlElement("w:pPr")
    borders = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), str(sz))
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), color)
    borders.append(bottom)
    p_pr.append(borders)
    return p_pr


def body(doc, text: str, *, size=BODY_PT, space_after=6, italic=False,
         color=None, align="justify") -> None:
    para = doc.add_paragraph()
    para.paragraph_format.space_before = Pt(0)
    para.paragraph_format.space_after = Pt(space_after)
    para.paragraph_format.line_spacing = 1.28
    para.alignment = {"justify": WD_ALIGN_PARAGRAPH.JUSTIFY,
                      "left": WD_ALIGN_PARAGRAPH.LEFT,
                      "center": WD_ALIGN_PARAGRAPH.CENTER}[align]
    run = para.add_run(text)
    run.font.size = Pt(size)
    run.font.italic = italic
    run.font.name = "Calibri"
    if color is not None:
        run.font.color.rgb = color


def bullet(doc, text: str, *, size=BODY_PT) -> None:
    para = doc.add_paragraph(style="List Bullet")
    para.paragraph_format.space_before = Pt(0)
    para.paragraph_format.space_after = Pt(3)
    para.paragraph_format.line_spacing = 1.2
    para.paragraph_format.left_indent = Cm(0.6)
    run = para.add_run(text)
    run.font.size = Pt(size)
    run.font.name = "Calibri"


def caption(doc, text: str) -> None:
    para = doc.add_paragraph()
    para.paragraph_format.space_before = Pt(4)
    para.paragraph_format.space_after = Pt(2)
    run = para.add_run(text)
    run.font.size = Pt(8.5)
    run.font.bold = True
    run.font.color.rgb = NAVY
    run.font.name = "Calibri"


def source_note(doc, text: str) -> None:
    para = doc.add_paragraph()
    para.paragraph_format.space_before = Pt(0)
    para.paragraph_format.space_after = Pt(10)
    run = para.add_run(text)
    run.font.size = Pt(NOTE_PT)
    run.font.italic = True
    run.font.color.rgb = LIGHT
    run.font.name = "Calibri"


def figure(doc, path: Path, caption_text: str, note: str, *, width=16.2) -> None:
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para.paragraph_format.space_before = Pt(6)
    para.paragraph_format.space_after = Pt(0)
    para.add_run().add_picture(str(path), width=Cm(width))
    caption(doc, caption_text)
    source_note(doc, note)


def table(doc, headers, rows, widths, aligns, *, header_fill=NAVY_HEX,
          reserve_from=None, font=TABLE_PT, bold_last_row=False) -> None:
    tbl = doc.add_table(rows=1, cols=len(headers))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    cell_margins(tbl)
    # Set the grid columns as well as the cell widths; a mismatch between the two
    # makes Word draw gaps between adjacent shaded header cells.
    for i, w in enumerate(widths):
        tbl.columns[i].width = Cm(w)

    hdr = tbl.rows[0]
    repeat_header(hdr)
    for i, (text, w, a) in enumerate(zip(headers, widths, aligns)):
        cell = hdr.cells[i]
        cell.width = Cm(w)
        shade(cell, header_fill)
        write_cell(cell, text, bold=True, align=a, size=font,
                   color=RGBColor(0xFF, 0xFF, 0xFF))
        # Shade and border every edge so no white seam appears between columns.
        cell_borders(cell, edges=("top", "bottom", "left", "right"), sz=4)

    for r, row in enumerate(rows):
        tr = tbl.add_row()
        no_split(tr)
        is_reserve = reserve_from is not None and r >= reserve_from
        for i, (value, w, a) in enumerate(zip(row, widths, aligns)):
            cell = tr.cells[i]
            cell.width = Cm(w)
            write_cell(cell, value, align=a, size=font,
                       bold=(bold_last_row and r == len(rows) - 1))
            if is_reserve:
                shade(cell, RESERVE_HEX)
            elif r % 2 == 1:
                shade(cell, ZEBRA_HEX)
    return tbl
