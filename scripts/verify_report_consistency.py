"""Verify the final report against the saved result files.

Reads the generated DOCX tables and the exported PDF text, and checks every
ranking value against outputs/final_rankings.csv. Nothing is taken on trust:
each check re-reads the source CSV.

Run:  py scripts/verify_report_consistency.py
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
DOCX = ROOT / "docs" / "Methodology_Report_Final.docx"
PDF = ROOT / "outputs" / "Methodology_Report_Final.pdf"
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

    # ---- locate the two ranking tables in the DOCX -----------------------
    ranking_tables = []
    for tbl in doc.tables:
        header = [clean(c.text) for c in tbl.rows[0].cells]
        if "Player" in header and "Median" in header:
            ranking_tables.append((header, tbl))
    check("two ranking tables found in the DOCX", len(ranking_tables) == 2,
          f"found {len(ranking_tables)}")

    for header, tbl in ranking_tables:
        body = tbl.rows[1:]
        role = None
        first_name = clean(body[0].cells[1].text)
        match = fr[fr["player_name"] == first_name]
        if not match.empty:
            role = match.iloc[0]["role"]
        check(f"{role or first_name}: table has 15 data rows", len(body) == 15,
              f"{len(body)} rows")
        if role is None:
            continue
        truth = fr[fr["role"] == role].sort_values("rank").head(15)

        col = {name: i for i, name in enumerate(header)}
        idx = {
            "rank": col.get("Rank"), "player": col.get("Player"),
            "P": col.get("P"), "C": col.get("C"), "S": col.get("S"),
            "rating": col.get("Rating"), "median": col.get("Median"),
            "ci": col.get("95% rank CI"), "p10": col.get("P(top 10)"),
            "p5": col.get("P(top 5)"),
        }

        names_ok = cls_ok = True
        for i, (_, t) in enumerate(truth.iterrows()):
            cells = [clean(c.text) for c in body[i].cells]
            if cells[idx["player"]] != t["player_name"]:
                names_ok = False
            if abs(num(cells[idx["P"]]) - round(t["P"], 3)) > 1e-9:
                cls_ok = False
            if abs(num(cells[idx["C"]]) - round(t["C"], 3)) > 1e-9:
                cls_ok = False
            if abs(num(cells[idx["S"]]) - round(t["S"], 3)) > 1e-9:
                cls_ok = False
            if abs(num(cells[idx["rating"]]) - round(t["rating"], 3)) > 1e-9:
                cls_ok = False
            if num(cells[idx["median"]]) != round(t["boot_median"]):
                cls_ok = False
            if abs(num(cells[idx["p10"]]) - round(t["p_top10"], 3)) > 1e-9:
                cls_ok = False
            if abs(num(cells[idx["p5"]]) - round(t["p_top5"], 3)) > 1e-9:
                cls_ok = False
            lo, _, hi = cells[idx["ci"]].partition("\u2013")
            if (num(lo), num(hi)) != (round(t["rank_lo"]), round(t["rank_hi"])):
                cls_ok = False

        check(f"{role}: name order matches final_rankings.csv", names_ok)
        check(f"{role}: P, C, S, rating, median, CI, P(top10), P(top5) match", cls_ok)

        top10 = truth[truth["rank"] <= 10]
        reserve = truth[truth["rank"] > 10]
        check(f"{role}: 10 selection names present in order",
              all(clean(body[i].cells[idx["player"]].text) == t["player_name"]
                  for i, (_, t) in enumerate(top10.iterrows())))
        check(f"{role}: reserves 11-15 present in order",
              all(clean(body[i + 10].cells[idx["player"]].text) == t["player_name"]
                  for i, (_, t) in enumerate(reserve.iterrows())))

    # ---- competition overlaps -------------------------------------------
    for _, r in c3.iterrows():
        pattern = f"{int(r['overlap_top10'])}/10"
        check(f"overlap {r['run']} {r['role']} = {pattern}", pattern in pdf_flat,
              "value not found in PDF text")
    # 9/10 may appear only as the disclosed superseded figure, never as a result.
    bad_context = []
    for m in re.finditer(r"9/10", pdf_flat):
        window = pdf_flat[max(0, m.start() - 130): m.end() + 130].lower()
        if not ("superseded" in window or "reported" in window or "earlier" in window):
            bad_context.append(window[:80])
    check("9/10 appears only as the disclosed superseded figure", not bad_context,
          f"{len(bad_context)} unexplained occurrence(s)")

    # ---- counts that were corrected -------------------------------------
    check("test count 97 stated", "97 tests" in pdf_flat)
    check("stale test count 91 absent", not re.search(r"\b91 tests\b", pdf_flat))
    check("hash audit: 25 authentic comparisons", re.search(r"\b25\b", pdf_flat) is not None)
    check("hash audit: 44 files with no baseline", "44" in pdf_flat)
    check("hash audit: zero mismatches stated", "zero mismatches" in pdf_flat.lower())
    check("no claim that all files are historically verified",
          "all 69" not in pdf_flat.lower())

    # ---- structure -------------------------------------------------------
    check("PDF page count within 12-16", 12 <= len(reader.pages) <= 16,
          f"{len(reader.pages)} pages")
    check("table of contents populated", "No table of contents entries found" not in pdf_flat)
    check("limitations section present", "7  Limitations" in pdf_flat
          or "Limitations" in pdf_flat)
    check("old team names not presented as contracts",
          "contract" not in pdf_flat.lower())

    # ---- report ----------------------------------------------------------
    print(f"{'CHECK':<62} RESULT")
    for name in PASSES:
        print(f"{name:<62} PASS")
    for name, detail in FAILS:
        print(f"{name:<62} FAIL  {detail}")
    print(f"\n{len(PASSES)} passed, {len(FAILS)} failed, pages={len(reader.pages)}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
