import numpy as np
import pandas as pd

from src.config import load_asym_thresholds

POSITION_TIERS = {
    "QB": "Skill", "WR": "Skill", "TE": "Skill",
    "DB": "Skill", "CB": "Skill", "S":  "Skill",
    "K":  "Skill", "P":  "Skill", "LS": "Skill",
    "RB": "Mid",   "FB": "Mid",   "LB": "Mid",
    "OL": "Big",   "DL": "Big",   "NT": "Big",
}

_THRESH = load_asym_thresholds()
DEFAULT_FLAG_PCT    = _THRESH["flag"]
DEFAULT_WARNING_PCT = _THRESH["warning"]


def process_metrics(df: pd.DataFrame, flag_pct: float = DEFAULT_FLAG_PCT) -> pd.DataFrame:
    out = df.copy()

    if "bodyweight_kg" not in out.columns:
        out["bodyweight_kg"] = np.nan
    bw = out["bodyweight_kg"].replace(0, np.nan)

    # Position tier
    if "position" in out.columns:
        out["tier"] = out["position"].map(POSITION_TIERS).fillna("Other")

    for mov in ("abd", "add"):
        for side in ("left", "right"):
            n_col  = f"hip_{mov}_{side}_n"
            nm_col = f"hip_{mov}_{side}_nm"

            if n_col in out.columns:
                out[f"hip_{mov}_{side}_n_per_kg"] = (out[n_col] / bw).round(2)

            if nm_col in out.columns:
                out[f"hip_{mov}_{side}_nm_per_kg"] = (out[nm_col] / bw).round(2)

        l_col = f"hip_{mov}_left_n"
        r_col = f"hip_{mov}_right_n"
        if l_col in out.columns and r_col in out.columns:
            stronger = out[[l_col, r_col]].max(axis=1)
            weaker   = out[[l_col, r_col]].min(axis=1)
            out[f"hip_{mov}_asym_pct"]  = (
                (stronger - weaker) / stronger.replace(0, np.nan) * 100
            ).round(1)
            out[f"hip_{mov}_asym_flag"] = out[f"hip_{mov}_asym_pct"] > flag_pct
            out[f"hip_{mov}_dominant"]  = np.where(
                out[l_col] >= out[r_col], "left", "right"
            )

    # ABD:ADD ratio per side (N/kg — always available when BW present)
    for side in ("left", "right"):
        abd_col = f"hip_abd_{side}_n_per_kg"
        add_col = f"hip_add_{side}_n_per_kg"
        if abd_col in out.columns and add_col in out.columns:
            out[f"hip_abd_add_ratio_{side}"] = (
                out[abd_col] / out[add_col].replace(0, np.nan)
            ).round(2)

    # Round raw force columns
    force_cols = [c for c in out.columns if c.endswith("_n") or c.endswith("_nm")]
    for c in force_cols:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(1)

    return out
