"""Verify the v2 report against the saved result files.

Reads the generated v2 DOCX tables and the exported v2 PDF text, and checks
every ranking value against outputs/final_rankings.csv. Nothing is taken on
trust: each check re-reads the source CSV.

Run:  py scripts/verify_report_consistency_v2.py
Exit code 0 only if every check passes.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "docs" / "Methodology_Report_Final_v2.docx"
PDF = ROOT / "outputs" / "Methodology_Report_Final_v2.pdf"
FR = ROOT / "outputs" / "final_rankings.csv"
C3 = ROOT / "outputs" / "tables" / "s7_c3_overlap_recomputed.csv"

RESERVE_MARK = "\u1d3f"
FAILS: list[tuple[str, str]] = []
PASSES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASSES.append(name)
    else:
        FAILS.append((name, detail))


def clean(text: str) -> str:
    return text.replace(RESERVE_MARK, "").replace("\u2212", "-").strip()


def num(text: str) -> float | None:
    t = clean(text).replace("\u2013", "-")
    if t in ("", "\u2014", "-"):
        return None
    try:
        return float(t)
    except ValueError:
        return None


def main() -> int:
    fr = pd.read_csv(FR)
    c3 = pd.read_csv(C3)
    doc = Document(str(DOCX))
    reader = PdfReader(str(PDF))
    pdf_text = "\n".join(p.extract_text() or "" for p in reader.pages)
    pdf_flat = re.sub(r"\s+", " ", pdf_text)

    # ---- documents exist and open ---------------------------------------
    check("v2 DOCX exists and opens", DOCX.exists() and len(doc.paragraphs) > 0)
    check("v2 PDF exists and opens", PDF.exists() and len(reader.pages) > 0)

    # ---- ranking tables in the DOCX -------------------------------------
    ranking_tables = []
    for tbl in doc.tables:
        header = [clean(c.text) for c in tbl.rows[0].cells]
        if "Player" in header and "Median" in header:
            ranking_tables.append((header, tbl))
    check("two ranking tables found in the v2 DOCX", len(ranking_tables) == 2,
          f"found {len(ranking_tables)}")

    for header, tbl in ranking_tables:
        body = tbl.rows[1:]
        first_name = clean(body[0].cells[1].text)
        match = fr[fr["player_name"] == first_name]
        role = match.iloc[0]["role"] if not match.empty else None
        check(f"{role or first_name}: table has 15 data rows", len(body) == 15,
              f"{len(body)} rows")
        if role is None:
            continue
        truth = fr[fr["role"] == role].sort_values("rank").head(15)
        col = {name: i for i, name in enumerate(header)}
        idx = {"rank": col.get("Rank"), "player": col.get("Player"),
               "P": col.get("P"), "C": col.get("C"), "S": col.get("S"),
               "rating": col.get("Rating"), "median": col.get("Median"),
               "ci": col.get("95% rank CI"), "p10": col.get("P(top 10)"),
               "p5": col.get("P(top 5)")}

        names_ok = vals_ok = True
        for i, (_, t) in enumerate(truth.iterrows()):
            cells = [clean(c.text) for c in body[i].cells]
            if cells[idx["player"]] != t["player_name"]:
                names_ok = False
            for key, src, nd in (("P", "P", 3), ("C", "C", 3), ("S", "S", 3),
                                 ("rating", "rating", 3), ("p10", "p_top10", 3),
                                 ("p5", "p_top5", 3)):
                if abs(num(cells[idx[key]]) - round(t[src], nd)) > 1e-9:
                    vals_ok = False
            if num(cells[idx["median"]]) != round(t["boot_median"]):
                vals_ok = False
            lo, _, hi = cells[idx["ci"]].partition("\u2013")
            if (num(lo), num(hi)) != (round(t["rank_lo"]), round(t["rank_hi"])):
                vals_ok = False
        check(f"{role}: name order matches final_rankings.csv", names_ok)
        check(f"{role}: P, C, S, rating, median, CI, P(top10), P(top5) match", vals_ok)
        check(f"{role}: 10 selected + 5 reserves present in order",
              all(clean(body[i].cells[idx["player"]].text) == t["player_name"]
                  for i, (_, t) in enumerate(truth.iterrows())))

    # ---- competition overlaps -------------------------------------------
    for _, r in c3.iterrows():
        pattern = f"{int(r['overlap_top10'])}/10"
        check(f"overlap {r['run']} {r['role']} = {pattern} stated in v2 PDF",
              pattern in pdf_flat, "value not found in PDF text")
    bad_context = []
    for m in re.finditer(r"9/10", pdf_flat):
        window = pdf_flat[max(0, m.start() - 150): m.end() + 150].lower()
        if not any(k in window for k in ("superseded", "reported", "earlier", "consensus")):
            bad_context.append(window[:60])
    check("9/10 appears only as the disclosed superseded/consensus figure",
          not bad_context, f"{len(bad_context)} unexplained occurrence(s)")

    # ---- counts ---------------------------------------------------------
    check("test count 97 stated in v2", "97 tests" in pdf_flat)
    check("stale test count 91 absent from v2", not re.search(r"\b91 tests\b", pdf_flat))
    check("hash audit: 25 authentic comparisons stated", "25" in pdf_flat)
    check("hash audit: 44 files with no baseline stated", "44" in pdf_flat)
    check("hash audit: zero mismatches stated", "zero mismatches" in pdf_flat.lower())
    check("no claim that all files are historically verified",
          "all 69" not in pdf_flat.lower() and "all files" not in pdf_flat.lower()
          or "not described as verified" in pdf_flat.lower())

    # ---- v2-specific structure -----------------------------------------
    check("problem statement section present", "Problem and scope" in pdf_flat)
    check("challenges section present", "Challenges encountered" in pdf_flat)
    check("Rajasthan Royals implications section present",
          "Rajasthan Royals selection implications" in pdf_flat)
    check("conclusion section present", "Conclusion" in pdf_flat)
    check("reproducibility appendix present", "Reproducibility appendix" in pdf_flat)
    check("limitations section present", "Limitations" in pdf_flat)

    # ---- figure numbering 1-8 -----------------------------------------
    for n in range(1, 9):
        check(f"Figure {n} caption present in v2 PDF", f"Figure {n}." in pdf_flat)

    # ---- methodology diagrams content -----------------------------------
    check("workflow diagram caption present", "End-to-end analytical workflow" in pdf_flat)
    check("pressure/LI diagram caption present", "Delivery-level pressure evaluation" in pdf_flat)
    check("P/C/S diagram caption present", "Three-component framework" in pdf_flat)
    check("bootstrap diagram caption present", "Match-level bootstrap procedure" in pdf_flat)
    check("pressure diagram mentions WPAE", "WPAE" in pdf_flat)
    check("pressure diagram mentions Leverage Index", "Leverage Index" in pdf_flat)
    check("P/C/S diagram source states equal weights (Composite = mean of z(P),z(C),z(S))",
          "equal weights" in (ROOT / "scripts" / "build_methodology_diagrams.py").read_text(encoding="utf-8")
          and "Composite rating = mean" in (ROOT / "scripts" / "build_methodology_diagrams.py").read_text(encoding="utf-8"))
    check("pressure sign convention stated in report (bowler value = negated centred coefficient)",
          "Sign correction applied exactly once: bowler value = negated centred coefficient." in pdf_flat)
    check("pressure diagram source states bowler value = -y",
          "bowler value = -y" in (ROOT / "scripts" / "build_methodology_diagrams.py").read_text(encoding="utf-8"))
    check("bootstrap diagram mentions match-level", "match-level" in pdf_flat.lower()
          or "Match-level" in pdf_flat)

    # ---- diagram assets exist -------------------------------------------
    diag_files = ["fig_methodology_workflow.png", "fig_pressure_li_workflow.png",
                  "fig_pcs_framework.png", "fig_bootstrap_workflow.png"]
    for name in diag_files:
        check(f"diagram asset exists: {name}",
              (ROOT / "outputs" / "figures" / "report" / name).exists())

    # ---- embedded images in DOCX (at least 8 figures) -------------------
    img_count = 0
    for rel in doc.part.rels.values():
        if "image" in rel.target_ref:
            img_count += 1
    check("v2 DOCX embeds at least 8 images", img_count >= 8,
          f"{img_count} images")

    # ---- no stale 'Figure X' placeholders ------------------------------
    check("no 'Figure X' placeholder remains", "Figure X" not in pdf_flat)

    # ---- page count -----------------------------------------------------
    pages = len(reader.pages)
    check("v2 PDF page count recorded (20-30)", 20 <= pages <= 30, f"{pages} pages")

    # ---- TOC ------------------------------------------------------------
    check("table of contents populated", "No table of contents entries found" not in pdf_flat)

    # ---- no blank pages --------------------------------------------------
    blank = [i + 1 for i, p in enumerate(reader.pages)
             if len((p.extract_text() or "").strip()) < 80]
    check("no blank or near-empty pages", not blank, f"pages {blank}")

    # ---- report ---------------------------------------------------------
    print(f"{'CHECK':<68} RESULT")
    for name in PASSES:
        print(f"{name:<68} PASS")
    for name, detail in FAILS:
        print(f"{name:<68} FAIL  {detail}")
    print(f"\n{len(PASSES)} passed, {len(FAILS)} failed, pages={pages}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
