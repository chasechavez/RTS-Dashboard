"""
Flexible CSV loader — auto-maps column names from any naming convention.

Handles:
  - Any case / separator (spaces, dashes, underscores, camelCase)
  - Abbreviations: abd/abduction, add/adduction, l/lt/left, r/rt/right
  - Units in column names: (N), (lbs), (Nm), kgf  → auto-converts to N / kg
  - Missing athlete_id   → generated from name
  - Any date format      → parsed automatically
  - Body weight in lbs   → auto-detected and converted to kg
  - Extra / unknown columns → silently ignored

Multi-source helpers:
  - load_bodyweight_csv()  — weight CSV with date + name + weight
  - load_roster_csv()      — roster CSV with name + position + jersey#
  - merge_all_sources()    — fuse force + BW + roster via fuzzy name matching
"""
import glob
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── Column normalisation ───────────────────────────────────────────────────────
def _slug(text: str) -> str:
    """Lowercase, collapse non-alphanumeric runs to single underscore."""
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


# ── Identity field aliases ─────────────────────────────────────────────────────
_ID_ALIASES: Dict[str, List[str]] = {
    "date": [
        "date", "test_date", "session_date", "testing_date",
        "assessment_date", "test_day", "session_day", "day", "timestamp",
    ],
    "athlete_id": [
        "athlete_id", "player_id", "subject_id", "id", "athlete_number",
        "player_number", "jersey", "number", "uid", "roster_id",
    ],
    "athlete_name": [
        "athlete_name", "player_name", "name", "athlete", "player",
        "subject", "full_name", "first_last", "last_first", "participant",
    ],
    "position": [
        "position", "pos", "role", "group", "position_group",
        "pos_group", "unit", "depth_chart_pos",
    ],
    "bodyweight_kg": [
        "bodyweight_kg", "bw_kg", "body_weight_kg", "weight_kg", "mass_kg",
        "bodyweight", "body_weight", "bw", "weight", "mass", "body_mass",
        "bodyweight_lbs", "bw_lbs", "weight_lbs", "body_weight_lbs",
    ],
}


def _match_identity(slugs: Dict[str, str]) -> Dict[str, str]:
    """Return {canonical: original_col} for identity fields."""
    mapping: Dict[str, str] = {}
    for canonical, aliases in _ID_ALIASES.items():
        # Exact alias match first
        for alias in aliases:
            if alias in slugs:
                mapping[canonical] = slugs[alias]
                break
        # Fallback: partial keyword match
        if canonical not in mapping:
            keywords = {
                "date":          r"date|session|test_day",
                "athlete_id":    r"_id$|^id_|^id$|jersey|number",
                "athlete_name":  r"name|athlete|player|subject",
                "position":      r"position|^pos$|role|group",
                "bodyweight_kg": r"weight|mass|^bw$",
            }.get(canonical, "")
            if keywords:
                for slug, orig in slugs.items():
                    if re.search(keywords, slug) and canonical not in mapping:
                        mapping[canonical] = orig
    return mapping


# ── Force column pattern detection ────────────────────────────────────────────
# Matches patterns like:
#   "Hip Abduction Left (N)"  →  hip_abd_left_n
#   "ADD_RT_FORCE_LBS"        →  hip_add_right_n  (+ conversion flag)
#   "left_abduction_nm"       →  hip_abd_left_nm
# NOTE: slugs use underscores, so \b word-boundaries don't work (_ is \w).
# Use plain substrings or underscore-anchored patterns instead.
_MOV_RE   = re.compile(r"abd|abduct")
_ADD_RE   = re.compile(r"(?:^|_)add(?:_|$)|adduct")
_LEFT_RE  = re.compile(r"left|_lt_|_lt$|^lt_|_l_|_l$|^l_")
_RIGHT_RE = re.compile(r"right|_rt_|_rt$|^rt_|_r_|_r$|^r_")
_NM_RE    = re.compile(r"_nm_|_nm$|^nm_|^nm$|torque|moment")
_LBS_RE   = re.compile(r"lbs|pound")
_KGF_RE   = re.compile(r"kgf|kg_f")
_SKIP_RE  = re.compile(r"per_kg|_n_kg|normalized|per_bw|relative|percent|pct|ratio")


def _match_force(slugs: Dict[str, str]) -> Dict[str, str]:
    """Return {canonical_force_col: original_col} for all force columns."""
    mapping: Dict[str, str] = {}
    for slug, orig in slugs.items():
        # Skip normalised/per-BW columns (we compute these ourselves)
        if _SKIP_RE.search(slug):
            continue
        # Movement
        if _MOV_RE.search(slug):
            mov = "abd"
        elif _ADD_RE.search(slug):
            mov = "add"
        else:
            continue
        # Side
        if _LEFT_RE.search(slug):
            side = "left"
        elif _RIGHT_RE.search(slug):
            side = "right"
        else:
            continue
        # Unit (nm/torque takes priority over plain N)
        unit = "nm" if _NM_RE.search(slug) else "n"
        canonical = f"hip_{mov}_{side}_{unit}"
        if canonical not in mapping:
            mapping[canonical] = orig
    return mapping


def _detect_columns(cols: List[str]) -> Dict[str, str]:
    slugs = {_slug(c): c for c in cols}
    mapping = _match_identity(slugs)
    mapping.update(_match_force(slugs))
    return mapping


# ── Unit conversion ───────────────────────────────────────────────────────────
def _force_multiplier(orig_col: str, values: pd.Series) -> float:
    """Return factor to multiply column to get Newtons."""
    s = _slug(orig_col)
    if _LBS_RE.search(s):
        return 4.44822          # lbs → N
    if _KGF_RE.search(s):
        return 9.80665          # kgf → N
    # Heuristic: median force < 50 is almost certainly kgf (30–50 kgf = 300–500 N)
    med = values.dropna().median()
    if pd.notna(med) and 10 < med < 80:
        return 9.80665
    return 1.0


def _bw_multiplier(orig_col: str, values: pd.Series) -> float:
    """Return factor to convert body weight column to kg."""
    s = _slug(orig_col)
    if _LBS_RE.search(s):
        return 0.453592         # lbs → kg
    # Heuristic: median > 150 → likely lbs (American football players)
    med = values.dropna().median()
    if pd.notna(med) and med > 150:
        return 0.453592
    return 1.0


# ── Standardise one dataframe ─────────────────────────────────────────────────
def _standardize(raw: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    rename = {orig: canon for canon, orig in mapping.items()}
    df = raw.rename(columns=rename)

    # ── Date ─────────────────────────────────────────────────────────────────
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=False)

    # ── Body weight conversion ────────────────────────────────────────────────
    if "bodyweight_kg" in df.columns:
        orig_bw = mapping.get("bodyweight_kg", "bodyweight_kg")
        mult = _bw_multiplier(orig_bw, df["bodyweight_kg"])
        if mult != 1.0:
            df["bodyweight_kg"] = df["bodyweight_kg"] * mult
        df["bodyweight_kg"] = df["bodyweight_kg"].round(1)

    # ── Force / torque conversion ─────────────────────────────────────────────
    for canon, orig in mapping.items():
        if re.match(r"hip_(abd|add)_(left|right)_(n|nm)$", canon) and canon in df.columns:
            mult = _force_multiplier(orig, df[canon])
            if mult != 1.0:
                df[canon] = df[canon] * mult
            df[canon] = df[canon].round(1)

    # ── Athlete ID fallback ───────────────────────────────────────────────────
    if "athlete_id" not in df.columns and "athlete_name" in df.columns:
        df["athlete_id"] = (
            df["athlete_name"]
            .astype(str)
            .apply(lambda n: re.sub(r"[^A-Z0-9]", "", n.upper())[:10])
        )

    # ── Position fallback ─────────────────────────────────────────────────────
    if "position" not in df.columns:
        df["position"] = "UNK"

    return df


# ── Minimum viable columns ────────────────────────────────────────────────────
_REQUIRED = {
    "date",
    "athlete_name",
    "bodyweight_kg",
    "hip_abd_left_n",
    "hip_abd_right_n",
    "hip_add_left_n",
    "hip_add_right_n",
}


def load_isometric_data(raw_dir: str) -> Tuple[pd.DataFrame, List[str]]:
    files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    errors: List[str] = []

    if not files:
        return pd.DataFrame(), errors

    dfs: List[pd.DataFrame] = []

    for path in files:
        fname = os.path.basename(path)
        try:
            raw = pd.read_csv(path)
            if raw.empty:
                continue

            mapping = _detect_columns(raw.columns.tolist())
            missing  = _REQUIRED - set(mapping.keys())

            if len(missing) > 2:
                mapped_str  = ", ".join(sorted(mapping.keys()))
                missing_str = ", ".join(sorted(missing))
                errors.append(
                    f"{fname}: missing {missing_str}. "
                    f"Detected: {mapped_str or 'nothing'}."
                )
                continue

            if missing:
                errors.append(f"{fname}: partial load — missing {', '.join(sorted(missing))}")

            df = _standardize(raw, mapping)

            # Drop rows missing all force data
            force_cols = [c for c in df.columns
                          if re.match(r"hip_(abd|add)_(left|right)_n$", c)]
            if force_cols:
                df = df.dropna(subset=force_cols, how="all")

            dfs.append(df)

        except Exception as exc:
            errors.append(f"{fname}: {exc}")

    if not dfs:
        return pd.DataFrame(), errors

    combined = pd.concat(dfs, ignore_index=True)

    dedup_key = ["date", "athlete_id"] if "athlete_id" in combined.columns else ["date", "athlete_name"]
    combined = (
        combined
        .drop_duplicates(subset=dedup_key)
        .sort_values(["athlete_name", "date"])
        .reset_index(drop=True)
    )

    if "athlete_id" not in combined.columns:
        combined["athlete_id"] = (
            combined["athlete_name"]
            .astype(str)
            .apply(lambda n: re.sub(r"[^A-Z0-9]", "", n.upper())[:10])
        )

    return combined, errors


# ── Body-weight CSV loader ────────────────────────────────────────────────────
_BW_NAME_ALIASES = [
    "athlete_name", "player_name", "name", "athlete", "player",
    "subject", "full_name", "first_last", "last_first", "participant",
]
_BW_DATE_ALIASES = [
    "date", "test_date", "session_date", "testing_date",
    "assessment_date", "test_day", "session_day", "day", "timestamp",
]
_BW_WEIGHT_ALIASES = [
    "bodyweight_kg", "bw_kg", "body_weight_kg", "weight_kg", "mass_kg",
    "bodyweight", "body_weight", "bw", "weight", "mass", "body_mass",
    "bodyweight_lbs", "bw_lbs", "weight_lbs", "body_weight_lbs",
]


def load_bodyweight_csv(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Parse a bodyweight-only CSV.

    Returns a DataFrame with columns: date, athlete_name, bodyweight_kg
    Handles lbs auto-detection and conversion.
    """
    errors: List[str] = []
    slugs = {_slug(c): c for c in df.columns}

    # Map required columns
    name_col   = next((slugs[a] for a in _BW_NAME_ALIASES if a in slugs), None)
    date_col   = next((slugs[a] for a in _BW_DATE_ALIASES if a in slugs), None)
    weight_col = next((slugs[a] for a in _BW_WEIGHT_ALIASES if a in slugs), None)

    if not name_col:
        errors.append("bodyweight CSV: cannot find athlete name column")
        return pd.DataFrame(), errors
    if not weight_col:
        errors.append("bodyweight CSV: cannot find weight column")
        return pd.DataFrame(), errors

    out = pd.DataFrame()
    out["athlete_name"] = df[name_col].astype(str)

    if date_col:
        out["date"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=False)
    else:
        out["date"] = pd.NaT

    out["bodyweight_kg"] = pd.to_numeric(df[weight_col], errors="coerce")
    mult = _bw_multiplier(weight_col, out["bodyweight_kg"])
    if mult != 1.0:
        out["bodyweight_kg"] = (out["bodyweight_kg"] * mult).round(1)
    else:
        out["bodyweight_kg"] = out["bodyweight_kg"].round(1)

    out = out.dropna(subset=["athlete_name", "bodyweight_kg"])
    return out.reset_index(drop=True), errors


# ── Roster CSV loader ─────────────────────────────────────────────────────────
_ROSTER_POS_ALIASES = [
    "position", "pos", "role", "group", "position_group",
    "pos_group", "unit", "depth_chart_pos",
]
_ROSTER_JERSEY_ALIASES = [
    "jersey", "jersey_number", "number", "jersey_num", "num",
    "player_number", "athlete_number", "no", "no_",
]
_ROSTER_HEIGHT_ALIASES = [
    "ht", "height", "height_ft", "ht_ft", "ht_in",
]
_ROSTER_WEIGHT_ALIASES = [
    "wt", "wt_lbs", "weight_lbs", "wt_lb", "weight_lb",
]
_ROSTER_ELIG_ALIASES = [
    "elig", "eligibility", "yr", "class_year", "year",
]
_ROSTER_HOMETOWN_ALIASES = [
    "hometown_high_school", "hometown", "city_high_school",
    "high_school", "origin", "location",
]


def _parse_height_cm(val) -> Optional[float]:
    """Parse '6\\'4\"' or '6-4' height string to centimetres."""
    try:
        m = re.match(r"""(\d+)['’\-](\d+)""", str(val).strip())
        if m:
            return round((int(m.group(1)) * 12 + int(m.group(2))) * 2.54, 1)
    except Exception:
        pass
    return None


def load_roster_csv(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Parse a roster CSV (flexible — handles NW-style rosters and generic formats).

    Extracts: athlete_name, athlete_id, position, jersey_number,
              height_raw, height_cm, bodyweight_kg (from WT lbs),
              eligibility, hometown.
    """
    errors: List[str] = []
    slugs = {_slug(c): c for c in df.columns}

    name_col     = next((slugs[a] for a in _BW_NAME_ALIASES          if a in slugs), None)
    id_col       = next((slugs[a] for a in _ID_ALIASES["athlete_id"] if a in slugs), None)
    pos_col      = next((slugs[a] for a in _ROSTER_POS_ALIASES        if a in slugs), None)
    jersey_col   = next((slugs[a] for a in _ROSTER_JERSEY_ALIASES     if a in slugs), None)
    height_col   = next((slugs[a] for a in _ROSTER_HEIGHT_ALIASES     if a in slugs), None)
    weight_col   = next((slugs[a] for a in _ROSTER_WEIGHT_ALIASES     if a in slugs), None)
    elig_col     = next((slugs[a] for a in _ROSTER_ELIG_ALIASES       if a in slugs), None)
    hometown_col = next((slugs[a] for a in _ROSTER_HOMETOWN_ALIASES   if a in slugs), None)

    if not name_col and not id_col:
        errors.append("roster CSV: cannot find athlete name or ID column")
        return pd.DataFrame(), errors

    out = pd.DataFrame()

    if name_col:
        out["athlete_name"] = df[name_col].astype(str).str.strip()
    if id_col:
        out["athlete_id"] = df[id_col].astype(str).str.strip()
    if pos_col:
        out["position"] = df[pos_col].astype(str).str.strip().str.upper()
    if jersey_col:
        out["jersey_number"] = pd.to_numeric(df[jersey_col], errors="coerce")
    if height_col:
        raw_h = df[height_col].astype(str)
        out["height_raw"] = raw_h
        out["height_cm"]  = raw_h.apply(_parse_height_cm)
    if weight_col:
        wt_vals = pd.to_numeric(df[weight_col], errors="coerce")
        mult = _bw_multiplier(weight_col, wt_vals)
        out["bodyweight_kg"] = (wt_vals * mult).round(1)
    if elig_col:
        out["eligibility"] = df[elig_col].astype(str).str.strip()
    if hometown_col:
        out["hometown"] = df[hometown_col].astype(str).str.strip()

    if "athlete_name" not in out.columns and "athlete_id" in out.columns:
        out["athlete_name"] = out["athlete_id"]

    out = out.dropna(subset=["athlete_name"])
    out = out[out["athlete_name"].str.lower() != "nan"]
    return out.drop_duplicates(subset=["athlete_name"]).reset_index(drop=True), errors


# ── KangaTech 360 long-format loader ─────────────────────────────────────────
# Maps substrings found in KangaTech's "Movement" / "Test Name" column
# to our canonical movement key (hip_abd / hip_add).
_KANGATECH_MOV_MAP: Dict[str, str] = {
    "hip abduction":  "hip_abd",
    "hip abd":        "hip_abd",
    "abduction":      "hip_abd",
    "hip adduction":  "hip_add",
    "hip add":        "hip_add",
    "adduction":      "hip_add",
}

_KT_NAME_ALIASES = [
    "subject", "client", "subject_name", "client_name",
    "athlete_name", "name", "athlete", "player", "participant",
]
_KT_DATE_ALIASES = [
    "test_date", "date", "assessment_date", "session_date", "timestamp",
]
_KT_MOV_ALIASES = [
    "movement", "test_name", "test_type", "exercise", "test", "assessment",
]
_KT_SIDE_ALIASES = [
    "side", "limb", "laterality", "direction",
]
_KT_FORCE_ALIASES = [
    "peak_force_n", "peak_force", "maximum_force_n", "maximum_force",
    "best_result_n", "best_result", "force_n", "force", "peak",
    "max_force", "peak_value", "result",
]
_KT_BW_ALIASES = [
    "body_mass_kg", "body_weight_kg", "body_mass", "body_weight",
    "mass_kg", "weight_kg", "bodyweight_kg", "bw", "weight", "mass",
]


def load_kangatech_csv(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Reshape KangaTech 360 long-format exports (one row per movement/side test)
    into the wide format expected by the rest of the pipeline.

    Expected input columns (flexible naming):
      Subject/Name, Test Date, Movement/Test Name, Side, Peak Force (N),
      Body Mass (kg)

    Returns (wide_df, errors) where wide_df has canonical column names.
    """
    errors: List[str] = []
    slugs = {_slug(c): c for c in df.columns}

    name_col  = next((slugs[a] for a in _KT_NAME_ALIASES  if a in slugs), None)
    date_col  = next((slugs[a] for a in _KT_DATE_ALIASES  if a in slugs), None)
    mov_col   = next((slugs[a] for a in _KT_MOV_ALIASES   if a in slugs), None)
    side_col  = next((slugs[a] for a in _KT_SIDE_ALIASES  if a in slugs), None)
    force_col = next((slugs[a] for a in _KT_FORCE_ALIASES if a in slugs), None)
    bw_col    = next((slugs[a] for a in _KT_BW_ALIASES    if a in slugs), None)

    missing = []
    if not name_col:  missing.append("subject/name")
    if not mov_col:   missing.append("movement/test_name")
    if not side_col:  missing.append("side/limb")
    if not force_col: missing.append("force/peak_force")
    if missing:
        avail = ", ".join(list(df.columns)[:12])
        return pd.DataFrame(), [
            f"KangaTech CSV: missing columns: {', '.join(missing)}. "
            f"Available: {avail}"
        ]

    group_keys = [name_col] + ([date_col] if date_col else [])
    rows: List[dict] = []

    for group_vals, group in df.groupby(group_keys, sort=False):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)

        row: dict = {"athlete_name": str(group_vals[0]).strip()}
        if date_col and len(group_vals) > 1:
            row["date"] = group_vals[1]

        if bw_col:
            bw_vals = pd.to_numeric(group[bw_col], errors="coerce").dropna()
            if not bw_vals.empty:
                row["bodyweight_kg"] = float(bw_vals.iloc[0])

        for _, test_row in group.iterrows():
            mov_raw  = str(test_row[mov_col]).lower().strip()
            side_raw = str(test_row[side_col]).lower().strip()

            canon_mov = next(
                (v for k, v in _KANGATECH_MOV_MAP.items() if k in mov_raw), None
            )
            if not canon_mov:
                continue

            if "left"  in side_raw or side_raw in ("l", "lt", "l."):
                side = "left"
            elif "right" in side_raw or side_raw in ("r", "rt", "r."):
                side = "right"
            else:
                continue

            try:
                val = float(test_row[force_col])
                if not np.isnan(val):
                    row[f"{canon_mov}_{side}_n"] = val
            except (TypeError, ValueError):
                pass

        rows.append(row)

    if not rows:
        return pd.DataFrame(), ["KangaTech CSV: no rows could be parsed"]

    result = pd.DataFrame(rows)

    if "date" in result.columns:
        result["date"] = pd.to_datetime(result["date"], errors="coerce")

    if "bodyweight_kg" in result.columns and bw_col:
        mult = _bw_multiplier(bw_col, result["bodyweight_kg"])
        if mult != 1.0:
            result["bodyweight_kg"] = (result["bodyweight_kg"] * mult).round(1)
        else:
            result["bodyweight_kg"] = result["bodyweight_kg"].round(1)

    if "athlete_id" not in result.columns and "athlete_name" in result.columns:
        result["athlete_id"] = (
            result["athlete_name"]
            .astype(str)
            .apply(lambda n: re.sub(r"[^A-Z0-9]", "", n.upper())[:10])
        )

    if "position" not in result.columns:
        result["position"] = "UNK"

    return result.reset_index(drop=True), errors


# ── Multi-source merger ───────────────────────────────────────────────────────
def merge_all_sources(
    force_df: pd.DataFrame,
    bw_df: Optional[pd.DataFrame] = None,
    roster_df: Optional[pd.DataFrame] = None,
    bw_date_tolerance_days: int = 3,
    name_match_threshold: float = 0.72,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Fuse force data with optional bodyweight and roster DataFrames.

    BW matching:
      - Exact date first; fallback nearest within *bw_date_tolerance_days*.
      - Existing bodyweight_kg in force_df is preserved when no BW row matches.

    Roster matching:
      - Fuzzy name match to bring in position / jersey_number.
      - Existing position in force_df is preserved when no match found.

    Returns (merged_df, warnings).
    """
    from src.etl.name_matching import build_name_map  # local import avoids circulars

    warnings: List[str] = []
    out = force_df.copy()

    # ── Merge bodyweight ──────────────────────────────────────────────────────
    if bw_df is not None and not bw_df.empty and "athlete_name" in bw_df.columns:
        bw = bw_df.copy()
        force_names  = out["athlete_name"].dropna().unique().tolist()
        bw_names     = bw["athlete_name"].dropna().unique().tolist()
        name_map_bw  = build_name_map(force_names, bw_names, name_match_threshold)

        # Reverse map: bw_canonical → force_name
        force_to_bw: Dict[str, str] = {}
        for fn, (bn, score) in name_map_bw.items():
            if bn is not None:
                force_to_bw[fn] = bn
            else:
                warnings.append(
                    f"BW: no match for '{fn}' (best score {score:.2f})"
                )

        has_bw_date = "date" in bw.columns and bw["date"].notna().any()

        if has_bw_date:
            # Date-matched merge
            rows_updated = 0
            for idx, row in out.iterrows():
                fn = row.get("athlete_name")
                fd = row.get("date")
                if fn not in force_to_bw or pd.isna(fd):
                    continue
                bn   = force_to_bw[fn]
                sub  = bw[bw["athlete_name"] == bn].copy()
                sub  = sub.dropna(subset=["date"])
                if sub.empty:
                    continue
                # Exact match
                exact = sub[sub["date"] == fd]
                if not exact.empty:
                    out.at[idx, "bodyweight_kg"] = exact.iloc[0]["bodyweight_kg"]
                    rows_updated += 1
                else:
                    # Nearest within tolerance
                    sub["_diff"] = (sub["date"] - fd).abs()
                    nearest = sub.sort_values("_diff").iloc[0]
                    tol = pd.Timedelta(days=bw_date_tolerance_days)
                    if nearest["_diff"] <= tol:
                        out.at[idx, "bodyweight_kg"] = nearest["bodyweight_kg"]
                        rows_updated += 1
            if rows_updated == 0:
                warnings.append(
                    "BW merge: no date-matched rows updated. "
                    "Check date formats match between CSVs."
                )
        else:
            # No date column — use single static BW per athlete
            static_bw: Dict[str, float] = {}
            for fn, bn in force_to_bw.items():
                sub = bw[bw["athlete_name"] == bn]["bodyweight_kg"].dropna()
                if not sub.empty:
                    static_bw[fn] = sub.iloc[-1]  # take last value

            def _fill_bw(row):
                if pd.isna(row.get("bodyweight_kg")):
                    return static_bw.get(row["athlete_name"], row.get("bodyweight_kg"))
                return row.get("bodyweight_kg")

            out["bodyweight_kg"] = out.apply(_fill_bw, axis=1)
            if static_bw:
                warnings.append(
                    "BW CSV has no date column — using static weight per athlete."
                )

    # ── Merge roster ──────────────────────────────────────────────────────────
    if roster_df is not None and not roster_df.empty and "athlete_name" in roster_df.columns:
        roster = roster_df.copy()
        force_names   = out["athlete_name"].dropna().unique().tolist()
        roster_names  = roster["athlete_name"].dropna().unique().tolist()
        name_map_ros  = build_name_map(force_names, roster_names, name_match_threshold)

        for idx, row in out.iterrows():
            fn = row.get("athlete_name")
            matched, score = name_map_ros.get(fn, (None, 0.0))
            if matched is None:
                continue
            r_row = roster[roster["athlete_name"] == matched].iloc[0]

            # Position — only fill if missing
            if "position" not in out.columns or pd.isna(row.get("position")) or str(row.get("position")) in ("UNK", "nan", ""):
                if "position" in r_row.index and pd.notna(r_row["position"]):
                    out.at[idx, "position"] = r_row["position"]

            # Jersey number
            if "jersey_number" not in out.columns or pd.isna(row.get("jersey_number")):
                if "jersey_number" in r_row.index and pd.notna(r_row["jersey_number"]):
                    out.at[idx, "jersey_number"] = r_row["jersey_number"]

            # Athlete ID from roster
            if "athlete_id" not in out.columns or pd.isna(row.get("athlete_id")):
                if "athlete_id" in r_row.index and pd.notna(r_row["athlete_id"]):
                    out.at[idx, "athlete_id"] = r_row["athlete_id"]

        # Report unmatched
        unmatched = [fn for fn, (bn, _) in name_map_ros.items() if bn is None]
        if unmatched:
            warnings.append(
                f"Roster: {len(unmatched)} athlete(s) unmatched: "
                + ", ".join(f"'{n}'" for n in unmatched[:8])
                + (" …" if len(unmatched) > 8 else "")
            )

    # ── Ensure position fallback ──────────────────────────────────────────────
    if "position" not in out.columns:
        out["position"] = "UNK"
    else:
        out["position"] = out["position"].fillna("UNK").replace("", "UNK")

    return out, warnings
