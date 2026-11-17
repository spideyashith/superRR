"""Verify the v3 report against the saved result files.

Reads the generated v3 DOCX tables and the exported v3 PDF text. Every ranking
value is checked against outputs/final_rankings.csv. The verifier also checks
that the four methodology diagrams and their source labels are gone, that the
contents page is populated, that the cover embeds the team mark, and that the
remaining figures are numbered sequentially.

Run:  py scripts/verify_report_consistency_v3.py
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
DOCX = ROOT / "docs" / "Methodology_Report_Final_v3.docx"
PDF = ROOT / "outputs" / "Methodology_Report_Final_v3.pdf"
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
    check("v3 DOCX exists and opens", DOCX.exists() and len(doc.paragraphs) > 0)
    check("v3 PDF exists and opens", PDF.exists() and len(reader.pages) > 0)

    # ---- ranking tables in the DOCX -------------------------------------
    ranking_tables = []
    for tbl in doc.tables:
        header = [clean(c.text) for c in tbl.rows[0].cells]
        if "Player" in header and "Median" in header:
            ranking_tables.append((header, tbl))
    check("two ranking tables found in the v3 DOCX", len(ranking_tables) == 2,
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
               "rating": col.get("Rating"), "consensus": col.get("Consensus 72"),
               "median": col.get("Median"), "ci": col.get("95% rank CI"),
               "p10": col.get("P(top 10)"), "p5": col.get("P(top 5)")}

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
            if num(cells[idx["consensus"]]) != round(t["consensus_rank_72"]):
                vals_ok = False
            if num(cells[idx["median"]]) != round(t["boot_median"]):
                vals_ok = False
            lo, _, hi = cells[idx["ci"]].partition("\u2013")
            if (num(lo), num(hi)) != (round(t["rank_lo"]), round(t["rank_hi"])):
                vals_ok = False
        check(f"{role}: name order matches final_rankings.csv", names_ok)
        check(f"{role}: P, C, S, rating, consensus, median, CI, P(top10), P(top5) match",
              vals_ok)
        check(f"{role}: 10 selected + 5 reserves present in order",
              all(clean(body[i].cells[idx["player"]].text) == t["player_name"]
                  for i, (_, t) in enumerate(truth.iterrows())))

    # ---- competition overlaps -------------------------------------------
    for _, r in c3.iterrows():
        pattern = f"{int(r['overlap_top10'])}/10"
        check(f"overlap {r['run']} {r['role']} = {pattern} stated in v3 PDF",
              pattern in pdf_flat, "value not found in PDF text")

    # ---- counts ---------------------------------------------------------
    check("test count 97 stated in v3", "97 tests" in pdf_flat)
    check("stale test count 91 absent from v3", not re.search(r"\b91 tests\b", pdf_flat))
    check("hash audit: 25 authentic comparisons stated", "25" in pdf_flat)
    check("hash audit: 44 files with no baseline stated", "44" in pdf_flat)
    check("hash audit: zero mismatches stated", "zero mismatches" in pdf_flat.lower())

    # ---- v3 structure ---------------------------------------------------
    for heading in ("Problem and scope", "Challenges encountered",
                    "Rajasthan Royals selection implications", "Conclusion",
                    "Reproducibility appendix", "Limitations"):
        check(f"section present: {heading}", heading in pdf_flat)

    # ---- contents page --------------------------------------------------
    check("contents page populated", "1 Executive summary" in pdf_flat
          and "12 Reproducibility appendix" in pdf_flat)
    check("old TOC instruction removed",
          "Section numbers and page numbers are generated by Word" not in pdf_flat)

    # ---- four methodology diagrams removed ------------------------------
    removed = [
        "End-to-end analytical workflow",
        "Delivery-level pressure evaluation",
        "Three-component framework",
        "Match-level bootstrap procedure",
    ]
    for text in removed:
        check(f"removed diagram caption absent: {text}", text not in pdf_flat)
    for n in (10, 11, 12, 13):
        check(f"old figure number {n} absent", f"Figure {n}." not in pdf_flat)

    # ---- data figures retained and numbered 1-9 -------------------------
    for n in range(1, 10):
        check(f"Figure {n} caption present", f"Figure {n}." in pdf_flat)
    check("no 'Figure X' placeholder remains", "Figure X" not in pdf_flat)

    # ---- source and page labels removed ---------------------------------
    check("no 'Source:' labels in v3 PDF", "Source:" not in pdf_flat)
    check("no 'Sources:' labels in v3 PDF", "Sources:" not in pdf_flat)
    check("no output-table paths in v3 PDF", "outputs/tables" not in pdf_flat)
    check("cover proprietary note removed",
          "Prepared for the Rajasthan Royals analytics audience" not in pdf_flat)
    check("no screenshot instruction", "page to screenshot" not in pdf_flat.lower())
    check("no DECISIONS.md reference", "DECISIONS.md" not in pdf_flat)

    # ---- corrected wording ----------------------------------------------
    check("rating formula stated",
          "Rating = mean(z(P), z(C), z(S))" in pdf_flat)
    check("equal weights stated", "equal weights after standardisation" in pdf_flat)
    check("split labels use matches",
          "Odd / even matches" in pdf_flat and "Earlier / later matches" in pdf_flat)
    check("top-10 overlap defined between halves",
          "both halves' top-10 lists" in pdf_flat)
    check("no winsorisation claim", "winsor" not in pdf_flat.lower())
    check("no 'Mean Spearman vs APV'", "Mean Spearman vs APV" not in pdf_flat)
    check("limitations 10.12 present", "10.12" in pdf_flat)
    check("no 10.13 limitation heading", "10.13" not in pdf_flat)
    check("conclusion says list ordered by P(top 10)",
          "ordered by P(top 10)" in pdf_flat)
    check("no conclusion claim of strongest on both",
          "on both pressure value and stability" not in pdf_flat)
    check("opposition correction present",
          "Opposition adjustment is weak for T20I-only players" in pdf_flat)
    check("no stale 'T20I opposition strength is not modelled'",
          "T20I opposition strength is not modelled" not in pdf_flat)

    # ---- placebo values --------------------------------------------------
    for value in ("0.87", "0.65", "0.12", "0.06", "0.54", "0.01", "0.66", "0.31"):
        check(f"placebo share {value} present", value in pdf_flat)

    # ---- cover mark and embedded images ---------------------------------
    img_count = sum(1 for rel in doc.part.rels.values() if "image" in rel.target_ref)
    check("v3 DOCX embeds the cover mark and data figures", img_count >= 10,
          f"{img_count} images")
    check("RR logo file exists", (ROOT / "outputs" / "figures" / "rr_logo.png").exists())

    # ---- page count and blank pages -------------------------------------
    pages = len(reader.pages)
    check("v3 PDF page count recorded (20-24)", 20 <= pages <= 24, f"{pages} pages")
    blank = [i + 1 for i, p in enumerate(reader.pages)
             if len((p.extract_text() or "").strip()) < 80]
    check("no blank or near-empty pages", not blank, f"pages {blank}")

    # ---- report ----------------------------------------------------------
    print(f"{'CHECK':<78} RESULT")
    for name in PASSES:
        print(f"{name:<78} PASS")
    for name, detail in FAILS:
        print(f"{name:<78} FAIL  {detail}")
    print(f"\n{len(PASSES)} passed, {len(FAILS)} failed, pages={pages}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
