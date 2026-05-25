"""Athlete profile tab — render function + helpers."""
import numpy as np
import pandas as pd
import streamlit as st

from src.viz.isometric_plots import (
    asymmetry_chart,
    percentile_strip_chart,
    zscore_benchmark_chart,
    torque_bw_chart,
)

_COL_LABELS = {
    "date":                    "Date",
    "athlete_id":              "Athlete ID",
    "athlete_name":            "Athlete",
    "position":                "Position",
    "tier":                    "Tier",
    "jersey_number":           "Jersey #",
    "bodyweight_kg":           "Bodyweight (kg)",
    "hip_abd_left_n":          "Hip ABD Left (N)",
    "hip_abd_right_n":         "Hip ABD Right (N)",
    "hip_abd_left_n_per_kg":   "ABD Left (N/kg)",
    "hip_abd_right_n_per_kg":  "ABD Right (N/kg)",
    "hip_abd_left_nm_per_kg":  "ABD Left (Nm/kg)",
    "hip_abd_right_nm_per_kg": "ABD Right (Nm/kg)",
    "hip_abd_asym_pct":        "ABD Asym (%)",
    "hip_abd_asym_flag":       "ABD Flag",
    "hip_add_left_n":          "Hip ADD Left (N)",
    "hip_add_right_n":         "Hip ADD Right (N)",
    "hip_add_left_n_per_kg":   "ADD Left (N/kg)",
    "hip_add_right_n_per_kg":  "ADD Right (N/kg)",
    "hip_add_left_nm_per_kg":  "ADD Left (Nm/kg)",
    "hip_add_right_nm_per_kg": "ADD Right (Nm/kg)",
    "hip_add_asym_pct":        "ADD Asym (%)",
    "hip_add_asym_flag":       "ADD Flag",
    "hip_abd_add_ratio_left":  "ABD:ADD Ratio L",
    "hip_abd_add_ratio_right": "ABD:ADD Ratio R",
}


# ── Shared helpers ─────────────────────────────────────────────────────────────
def _fmt(val, spec=".1f", fallback="-"):
    try:
        return fallback if pd.isna(val) else f"{val:{spec}}"
    except Exception:
        return fallback


def _delta(latest: pd.Series, prev, col: str, spec="+.1f"):
    if prev is None or col not in latest.index:
        return None
    try:
        d = float(latest[col]) - float(prev[col])
        return f"{d:{spec}}"
    except Exception:
        return None


def _jersey_str(latest: pd.Series) -> str:
    """Return '#NN&nbsp;&nbsp;·&nbsp;&nbsp;' prefix if jersey present, else ''."""
    raw = latest.get("jersey_number", latest.get("jersey", None))
    if raw is None:
        return ""
    try:
        return f"#{int(float(raw))}&nbsp;&nbsp;&middot;&nbsp;&nbsp;"
    except (TypeError, ValueError):
        return ""


# ── Movement section ───────────────────────────────────────────────────────────
def _movement_section(
    mov: str,
    label: str,
    latest: pd.Series,
    prev,
    adf: pd.DataFrame,
    flag_pct: float = 15.0,
    warn_pct: float = 10.0,
) -> None:
    st.subheader(label)

    l_n   = latest.get(f"hip_{mov}_left_n",          np.nan)
    r_n   = latest.get(f"hip_{mov}_right_n",         np.nan)
    l_rel = latest.get(f"hip_{mov}_left_n_per_kg",   np.nan)
    r_rel = latest.get(f"hip_{mov}_right_n_per_kg",  np.nan)
    l_nm  = latest.get(f"hip_{mov}_left_nm_per_kg",  np.nan)
    r_nm  = latest.get(f"hip_{mov}_right_nm_per_kg", np.nan)
    asym  = latest.get(f"hip_{mov}_asym_pct",        np.nan)

    # Dominant side
    dom_raw = latest.get(f"hip_{mov}_dominant", None)
    dom_str = (f" · {'L>R' if dom_raw == 'left' else 'R>L'}"
               if dom_raw in ("left", "right") else "")

    try:
        asym_f = float(asym)
    except (TypeError, ValueError):
        asym_f = None

    # Compute flag live from current thresholds (not stored boolean)
    flag = asym_f is not None and asym_f > flag_pct

    def _lb(n_val):
        try:
            return _fmt(float(n_val) * 0.224809, ".1f")
        except Exception:
            return "-"

    def _dchip(col, inverse=False):
        d = _delta(latest, prev, col)
        if d is None:
            return ""
        try:
            fval = float(d)
        except (TypeError, ValueError):
            return ""
        better = fval <= 0 if inverse else fval >= 0
        color  = "#22C55E" if better else "#EF4444"
        sign   = "+" if fval >= 0 else ""
        return (f'<span style="font-size:0.62rem; color:{color}; '
                f'margin-left:5px; opacity:0.85;">{sign}{fval:.1f}</span>')

    if flag:
        ac, abg, abdr = "#EF4444", "rgba(239,68,68,0.15)",  "#EF444450"
    elif asym_f is not None and asym_f > warn_pct:
        ac, abg, abdr = "#FFB400", "rgba(255,180,0,0.15)",  "#FFB40050"
    else:
        ac, abg, abdr = "#22C55E", "rgba(34,197,94,0.12)",  "#22C55E50"

    def _row(metric, l_val, r_val, l_sub="", r_sub="",
             col_l=None, col_r=None, inverse=False):
        ld = _dchip(col_l, inverse) if col_l else ""
        rd = _dchip(col_r, inverse) if col_r else ""
        ls = (f'<br><span style="font-size:0.68rem; color:var(--cc-sub-text);">'
              f'{l_sub}</span>') if l_sub else ""
        rs = (f'<br><span style="font-size:0.68rem; color:var(--cc-sub-text);">'
              f'{r_sub}</span>') if r_sub else ""
        return f"""
<tr style="border-top:1px dashed var(--cc-row-sep);">
  <td style="color:var(--cc-label-dim); font-size:0.75rem; padding:8px 0;
             font-family:Arial,sans-serif; white-space:nowrap;">{metric}</td>
  <td style="color:#00A3A3; font-size:0.92rem; font-weight:700;
             text-align:right; padding:8px 16px 8px 0;
             font-family:'Courier New',monospace; line-height:1.4;"
  >{l_val}{ls}{ld}</td>
  <td style="color:var(--cc-right-val); font-size:0.92rem; font-weight:700;
             text-align:right; padding:8px 0;
             font-family:'Courier New',monospace; line-height:1.4;"
  >{r_val}{rs}{rd}</td>
</tr>"""

    rows = _row(
        "Force",
        f"{_fmt(l_n)} N", f"{_fmt(r_n)} N",
        l_sub=f"{_lb(l_n)} lb", r_sub=f"{_lb(r_n)} lb",
        col_l=f"hip_{mov}_left_n", col_r=f"hip_{mov}_right_n",
    )
    rows += _row(
        "Force / BW",
        f"{_fmt(l_rel, '.2f')} N/kg", f"{_fmt(r_rel, '.2f')} N/kg",
        col_l=f"hip_{mov}_left_n_per_kg", col_r=f"hip_{mov}_right_n_per_kg",
    )
    has_torque = f"hip_{mov}_left_nm_per_kg" in adf.columns
    if has_torque:
        rows += _row(
            "Torque / BW",
            f"{_fmt(l_nm, '.2f')} Nm/kg", f"{_fmt(r_nm, '.2f')} Nm/kg",
            col_l=f"hip_{mov}_left_nm_per_kg", col_r=f"hip_{mov}_right_nm_per_kg",
        )

    asym_dchip = _dchip(f"hip_{mov}_asym_pct", inverse=True)
    lbl_upper  = "ABDUCTION" if mov == "abd" else "ADDUCTION"

    st.markdown(f"""
<div style="background:var(--cc-card-bg); border-radius:10px;
            padding:14px 18px 12px 18px; margin-bottom:0.7rem;
            border:1px solid var(--cc-card-border);">
  <div style="display:flex; justify-content:space-between; align-items:center;
              margin-bottom:10px; padding-bottom:8px;
              border-bottom:1px dashed var(--cc-row-sep);">
    <span style="color:var(--cc-label-dim); font-size:0.65rem; letter-spacing:1.8px;
                 text-transform:uppercase; font-family:Arial,sans-serif;">
      HIP {lbl_upper}
    </span>
    <span style="background:{abg}; color:{ac};
                 font-family:'Courier New',monospace; font-size:0.8rem;
                 font-weight:700; padding:3px 10px; border-radius:4px;
                 border:1px solid {abdr};">
      ASYM {_fmt(asym)}%{dom_str}{asym_dchip}
    </span>
  </div>
  <table style="width:100%; border-collapse:collapse;">
    <thead>
      <tr>
        <th style="color:var(--cc-th-metric); font-size:0.62rem; font-weight:500;
                   text-align:left; padding:0 0 5px 0; letter-spacing:1px;
                   font-family:Arial,sans-serif; width:26%;">METRIC</th>
        <th style="color:var(--cc-th-left,#00A3A3); font-size:0.62rem; font-weight:500;
                   text-align:right; padding:0 16px 5px 0; letter-spacing:1px;
                   font-family:Arial,sans-serif;">LEFT</th>
        <th style="color:var(--cc-th-right); font-size:0.62rem; font-weight:500;
                   text-align:right; padding:0 0 5px 0; letter-spacing:1px;
                   font-family:Arial,sans-serif;">RIGHT</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</div>
""", unsafe_allow_html=True)

    if flag:
        st.error(
            f"ASYMMETRY FLAG — {_fmt(asym)}% exceeds {flag_pct:.0f}% clinical threshold. "
            "Review recommended."
        )
    elif asym_f is not None and asym_f > warn_pct:
        st.warning(
            f"MONITOR — {_fmt(asym)}% asymmetry above {warn_pct:.0f}% warning level. "
            "Track closely."
        )
    else:
        st.success(f"Symmetry within normal range ({_fmt(asym)}%).")


# ── Public render function ─────────────────────────────────────────────────────
def render_athlete_profile(
    df: pd.DataFrame,
    adf: pd.DataFrame,
    sel_id: str,
    sel_name: str,
    latest: pd.Series,
    prev,
    dark_mode: bool = True,
    flag_pct: float = 15.0,
    warn_pct: float = 10.0,
) -> None:
    # ── Branded header ────────────────────────────────────────────────────────
    pos_str   = str(latest.get("position", "-"))
    tier_str  = str(latest.get("tier", "-")) if "tier" in latest.index else "-"
    bw_kg_val = latest.get("bodyweight_kg", np.nan)
    bw_str    = _fmt(bw_kg_val, ".1f")
    try:
        bw_lbs_str = _fmt(float(bw_kg_val) * 2.20462, ".1f")
    except Exception:
        bw_lbs_str = "-"
    bw_display = f"{bw_str} kg / {bw_lbs_str} lbs" if bw_str != "-" else "-"
    date_str   = pd.to_datetime(latest["date"]).strftime("%d %b %Y")
    ns         = len(adf)
    jsy        = _jersey_str(latest)

    st.markdown(f"""
<div style="background:#005F87; border-radius:10px;
            padding:20px 24px 16px 24px; margin-bottom:1.4rem;">
  <div style="font-size:1.75rem; font-weight:700; color:#FFFFFF;
              font-family:Arial,Helvetica,sans-serif; letter-spacing:-0.3px;
              line-height:1.2; margin-bottom:6px;">{sel_name}</div>
  <div style="font-size:1.08rem; font-weight:600; color:#00A3A3;
              font-family:'Courier New',Courier,monospace;
              letter-spacing:0.05em; margin-bottom:2px;">
    {jsy}{pos_str}&nbsp;&nbsp;&middot;&nbsp;&nbsp;{tier_str}&nbsp;&nbsp;&middot;&nbsp;&nbsp;{bw_display}
  </div>
  <div style="border-top:1px solid rgba(255,255,255,0.18); margin-top:14px;
              padding-top:11px; display:flex; gap:2.8rem; flex-wrap:wrap;">
    <span style="color:#C5DBE8; font-size:0.95rem; font-family:Arial,sans-serif;">
      Latest Test:&nbsp;<b style="color:#00A3A3; font-size:1rem;">{date_str}</b>
    </span>
    <span style="color:#C5DBE8; font-size:0.95rem; font-family:Arial,sans-serif;">
      Total Sessions:&nbsp;<b style="color:#00A3A3; font-size:1rem;">{ns}</b>
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Body weight staleness warning ─────────────────────────────────────────
    bw_vals = adf["bodyweight_kg"].dropna()
    if len(bw_vals) > 1 and bw_vals.nunique() == 1:
        st.warning(
            "Body weight identical across all sessions — "
            "verify athletes are weighed at each test. "
            "Static BW produces inaccurate torque normalisation."
        )

    # ── Movement columns ──────────────────────────────────────────────────────
    _mc1, _mc2 = st.columns(2)
    with _mc1:
        _movement_section("abd", "Hip Abduction", latest, prev, adf,
                          flag_pct=flag_pct, warn_pct=warn_pct)
    with _mc2:
        _movement_section("add", "Hip Adduction", latest, prev, adf,
                          flag_pct=flag_pct, warn_pct=warn_pct)

    # ── ABD:ADD ratio (norm 0.75–1.25) ────────────────────────────────────────
    ratio_l = latest.get("hip_abd_add_ratio_left",  np.nan)
    ratio_r = latest.get("hip_abd_add_ratio_right", np.nan)
    if not (pd.isna(ratio_l) and pd.isna(ratio_r)):
        def _ratio_color(v):
            try:
                fv = float(v)
                return "#22C55E" if 0.75 <= fv <= 1.25 else "#FFB400"
            except Exception:
                return "#6B7280"

        rc_l = _ratio_color(ratio_l)
        rc_r = _ratio_color(ratio_r)
        st.markdown(f"""
<div style="background:var(--cc-card-bg); border-radius:8px; padding:10px 18px;
            border:1px solid var(--cc-card-border); margin-bottom:0.7rem;
            display:flex; gap:2rem; align-items:center; flex-wrap:wrap;">
  <span style="color:var(--cc-label-dim); font-size:0.70rem; letter-spacing:1.5px;
               text-transform:uppercase; font-family:Arial,sans-serif;">
    ABD:ADD Ratio
  </span>
  <span style="font-family:'Courier New',monospace; font-size:0.88rem; font-weight:700;
               color:{rc_l};">Left {_fmt(ratio_l, '.2f')}</span>
  <span style="font-family:'Courier New',monospace; font-size:0.88rem; font-weight:700;
               color:{rc_r};">Right {_fmt(ratio_r, '.2f')}</span>
  <span style="color:var(--cc-label-dimmer); font-size:0.68rem;
               font-family:Arial,sans-serif; font-style:italic;">norm 0.75–1.25</span>
</div>
""", unsafe_allow_html=True)

    st.divider()
    st.subheader("Force / Torque Trends")
    _tc1, _tc2 = st.columns(2)
    with _tc1:
        _fig_abd = torque_bw_chart(adf, "abd", dark_mode=dark_mode)
        if _fig_abd.data:
            st.plotly_chart(_fig_abd, use_container_width=True)
        else:
            st.caption("ABD trend unavailable — check bodyweight recorded.")
    with _tc2:
        _fig_add = torque_bw_chart(adf, "add", dark_mode=dark_mode)
        if _fig_add.data:
            st.plotly_chart(_fig_add, use_container_width=True)
        else:
            st.caption("ADD trend unavailable — check bodyweight recorded.")

    st.divider()
    st.subheader("Asymmetry Over Time")
    st.plotly_chart(
        asymmetry_chart(adf, dark_mode=dark_mode,
                        flag_pct=flag_pct, warn_pct=warn_pct),
        use_container_width=True,
    )

    st.divider()
    st.subheader("Position Benchmarks")
    zfig = zscore_benchmark_chart(df, sel_id, dark_mode=dark_mode)
    if zfig.data:
        st.plotly_chart(zfig, use_container_width=True)
    else:
        st.caption("Z-score chart requires 3+ athletes in same position.")

    pc1, pc2 = st.columns(2)
    with pc1:
        pfig_abd = percentile_strip_chart(df, sel_id, "abd", dark_mode=dark_mode)
        if pfig_abd.data:
            st.plotly_chart(pfig_abd, use_container_width=True)
        else:
            st.caption("Percentile strip requires 3+ athletes in position (ABD).")
    with pc2:
        pfig_add = percentile_strip_chart(df, sel_id, "add", dark_mode=dark_mode)
        if pfig_add.data:
            st.plotly_chart(pfig_add, use_container_width=True)
        else:
            st.caption("Percentile strip requires 3+ athletes in position (ADD).")

    with st.expander("Raw Data"):
        display_df = (
            adf.sort_values("date", ascending=False)
            .reset_index(drop=True)
            .rename(columns={c: _COL_LABELS.get(c, c) for c in adf.columns})
        )
        st.dataframe(display_df, use_container_width=True)
