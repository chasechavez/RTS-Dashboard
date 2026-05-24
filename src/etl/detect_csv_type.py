"""
Heuristic CSV-type detector.

Returns one of: "force" | "bodyweight" | "roster" | "unknown"

Detection priority:
  1. Force   — has at least one hip ABD/ADD force column
  2. Bodyweight — has a weight/mass column but NO hip ABD/ADD columns
  3. Roster  — has a position column and NO hip ABD/ADD + NO weight column
  4. Unknown — nothing matched
"""
import re
from typing import Literal

import pandas as pd

CsvType = Literal["force", "bodyweight", "roster", "kangatech", "unknown"]

# ── Slug helper (mirrors load_data._slug) ─────────────────────────────────────
def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


# ── Column-presence tests ─────────────────────────────────────────────────────
_FORCE_RE    = re.compile(r"abd|abduct|add(?:_|$)|adduct")
_WEIGHT_RE   = re.compile(r"weight|mass|^bw$|bw_|_bw$|^wt$|^wt_")
_POSITION_RE = re.compile(r"position|^pos$|role|^group$|pos_group|unit|depth")
_NAME_RE     = re.compile(r"name|athlete|player|subject|client")
_DATE_RE     = re.compile(r"date|session|timestamp|day$")
_ID_RE       = re.compile(r"_id$|^id$|jersey|number|roster")
# KangaTech long-format markers: separate side + movement columns
_SIDE_RE     = re.compile(r"^side$|^limb$|^laterality$|^direction$")
_MOV_LONG_RE = re.compile(r"^movement$|^test_name$|^test_type$|^exercise$|^assessment$")


def _has(slugs: list, pattern: re.Pattern) -> bool:
    return any(pattern.search(s) for s in slugs)


def detect_csv_type(df: pd.DataFrame) -> CsvType:
    """
    Inspect column names of *df* and return the most likely CSV type.

    Does NOT look at data values — purely structural heuristic.
    """
    slugs = [_slug(c) for c in df.columns]

    has_force    = _has(slugs, _FORCE_RE)
    has_weight   = _has(slugs, _WEIGHT_RE)
    has_position = _has(slugs, _POSITION_RE)
    has_name     = _has(slugs, _NAME_RE)
    has_date     = _has(slugs, _DATE_RE)
    has_id       = _has(slugs, _ID_RE)

    if has_force:
        return "force"

    # KangaTech long format: separate side + movement columns, no wide force cols
    has_side_col = _has(slugs, _SIDE_RE)
    has_mov_col  = _has(slugs, _MOV_LONG_RE)
    if has_side_col and has_mov_col and has_name:
        return "kangatech"

    if has_weight and not has_force:
        return "bodyweight"

    if has_position and (has_name or has_id) and not has_weight and not has_force:
        return "roster"

    return "unknown"


def detect_from_file(path: str) -> tuple[CsvType, pd.DataFrame]:
    """
    Read CSV at *path*, detect type, return (type, dataframe).
    Raises on parse error.
    """
    df = pd.read_csv(path)
    return detect_csv_type(df), df
