# Data placement — read this before running anything

This repository contains **code, results, tables and figures only**. The original
proprietary Rajasthan Royals data pack is **not** included and must be supplied
locally by you.

## 1. Where to put the files

Create a folder named exactly:

```
Data Pack/
```

at the repository root (sibling of `src/`, `tests/`, `outputs/`) and place the six
original CSV files directly inside it. Do **not** create sub-folders; the loader
reads these paths verbatim from `src/config.py` (`DATA_DIR = ROOT / "Data Pack"`).

```
superRR/
  Data Pack/
    ball_by_ball.csv
    matches.csv
    teams.csv
    competitions.csv
    playing_eleven.csv
    competition_team_player_mapping.csv
  src/
  tests/
  outputs/
  requirements.txt
  README.md
```

## 2. Required files and expected shape

These are the row/column counts observed in the audit run
(`outputs/tables/data_audit.md`). They are a sanity check, not a requirement —
but if your file differs substantially the pipeline will not reproduce the
published rankings.

| File | Rows | Columns | Used for |
|---|---:|---:|---|
| `ball_by_ball.csv` | 200,999 | 33 | The fact table — one row per delivery |
| `matches.csv` | 883 | 22 | Match metadata, result, revised targets |
| `teams.csv` | 353 | 3 | Team registry |
| `competitions.csv` | 141 | 4 | Competition registry |
| `playing_eleven.csv` | 28,381 | 9 | XIs, batting order, impact-player flags |
| `competition_team_player_mapping.csv` | 6,413 | 6 | Canonical player-name registry |

The Round 1 problem-statement PDF is read for context only and contributes no data
fields beyond the PRD (`PRD_Pressure_Players.md`).

## 3. Do not commit the data

`Data Pack/` is excluded by `.gitignore`. It is proprietary third-party data: do
not add it to this repository, do not attach it to releases, and do not paste its
contents into issues or pull requests.

## 4. Verifying your copy before a full run

The first stage is a read-only audit and is the fastest way to confirm the data is
correct:

```bash
py -m src.audit
```

Then compare the generated `outputs/tables/data_audit.md` against the audit
already committed in this repository. The match-universe table should read:

| In scope | Reason | Matches |
|---|---|---:|
| yes | kept | 766 |
| yes | abandoned / no result | 41 |
| yes | DLS / shortened | 28 |
| yes | incomplete innings | 11 |
| yes | tied / super over | 7 |
| no | out-of-scope competition | 30 |

If those counts match, the rest of the pipeline will reproduce.

## 5. What you can run without the data

| Works without `Data Pack/` | Needs `Data Pack/` |
|---|---|
| `py -m pytest tests/ -q` (97 tests — they read saved CSVs) | `py -m src.audit` |
| `py scripts/recompute_c3_overlaps.py` | `py -m src.features` onward (all modelling stages) |
| `py scripts/dump_report_tables.py` | `py src/s7_b1_bootstrap.py` |

## 6. Honest note on reproducibility

The published rankings depend on the exact data pack and on the seed and library
versions pinned in `requirements.txt` (Python 3.13.12, pandas 2.3.3, numpy 2.4.2,
scikit-learn 1.8.0, lightgbm 4.7.0). Because the raw data is not distributed with
this repository, **a third party cannot reproduce the rankings end-to-end from
this repository alone** — they can reproduce them only after obtaining the
original data pack and placing it at `Data Pack/`.

Everything downstream of the modelling stages — the 30-player file of record, the
bootstrap summary, the placebo/temporal/competition validation, the hash audit and
the full test suite — can be verified from the committed outputs without the raw
data.
