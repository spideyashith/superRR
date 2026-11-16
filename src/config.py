"""Project-wide configuration: paths, seed, and data-domain constants.

Every stage imports paths and constants from here so that the pipeline is
reproducible and has a single source of truth.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (pathlib everywhere; Windows-friendly)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data Pack"
OUTPUT_DIR = ROOT / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Data-domain constants (values verified in Stage 0 audit)
# ---------------------------------------------------------------------------
#: ``extras_type`` values that make a delivery illegal (no legal ball counted).
#: ``Overstep`` and ``other`` are both no-ball families: the batter faces them
#: and can score off them, but they are not legal balls.
NO_BALL_TYPES = ("Overstep", "other")
WIDE = "wide"
PENALTY = "Penalty"
ILLEGAL_TYPES = (WIDE, *NO_BALL_TYPES, PENALTY)

#: Dismissals credited to the bowler.
BOWLER_WICKET_TYPES = ("Bowled", "Caught", "LBW", "Stump", "Hit Wicket")

#: Regulation T20 innings length (balls). Revised matches use their own overs.
REGULATION_BALLS = 120

#: Competition scope: IPL + T20I only (PRD section 2). The data pack also
#: contains two non-IPL T20 competitions (Emerging Teams Asia Cup, Asia Cup
#: Rising Stars); they are flagged and excluded from the modelling universe.
IPL_NAME_PATTERN = r"^IPL \d{4}$"

#: Matches whose result string indicates a tie / super over / eliminator.
TIE_PATTERN = r"Tied|Super Over|super over|Eliminator"

#: Matches whose result string indicates a DLS-adjusted outcome. A few DLS
#: matches carry no revised-overs / revised-target fields, so the text is a
#: necessary second signal.
DLS_PATTERN = r"D/L|DLS|Duckworth"
