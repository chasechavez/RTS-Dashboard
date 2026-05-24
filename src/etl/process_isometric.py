import numpy as np
import pandas as pd

POSITION_TIERS = {
    "QB": "Skill", "WR": "Skill", "TE": "Skill",
    "DB": "Skill", "CB": "Skill", "S":  "Skill",
    "K":  "Skill", "P":  "Skill", "LS": "Skill",
    "RB": "Mid",   "FB": "Mid",   "LB": "Mid",
    "OL": "Big",   "DL": "Big",   "NT": "Big",
}


def process_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    bw = out["bodyweight_kg"].replace(0, np.nan)

    # Position tier
    out["tier"] = out["position"].map(POSITION_TIERS).fillna("Other")

    for mov in ("abd", "add"):
        for side in ("left", "right"):
            n_col  = f"hip_{mov}_{side}_n"
            nm_col = f"hip_{mov}_{side}_nm"

            if n_col in out.columns:
                out[f"hip_{mov}_{side}_n_per_kg"] = (out[n_col] / bw).round(1)

            if nm_col in out.columns:
                out[f"hip_{mov}_{side}_nm_per_kg"] = (out[nm_col] / bw).round(1)

        l_col = f"hip_{mov}_left_n"
        r_col = f"hip_{mov}_right_n"
        if l_col in out.columns and r_col in out.columns:
            stronger = out[[l_col, r_col]].max(axis=1)
            weaker   = out[[l_col, r_col]].min(axis=1)
            out[f"hip_{mov}_asym_pct"]  = ((stronger - weaker) / stronger * 100).round(1)
            out[f"hip_{mov}_asym_flag"] = out[f"hip_{mov}_asym_pct"] > 15
            out[f"hip_{mov}_dominant"]  = np.where(
                out[l_col] >= out[r_col], "left", "right"
            )

    # Round raw force columns to 1 decimal
    force_cols = [c for c in out.columns if c.endswith("_n") or c.endswith("_nm")]
    for c in force_cols:
        if out[c].dtype in (float, np.float64):
            out[c] = out[c].round(1)

    return out
