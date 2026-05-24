import numpy as np
import pandas as pd
import streamlit as st

from src.etl.load_data import (
    load_isometric_data,
    load_bodyweight_csv,
    load_roster_csv,
    load_kangatech_csv,
    merge_all_sources,
)
from src.etl.detect_csv_type import detect_csv_type
from src.etl.process_isometric import process_metrics, POSITION_TIERS
import src.viz.isometric_plots as _iso_mod
from src.viz.isometric_plots import (
    asymmetry_chart,
    athlete_vs_team_chart,
    percentile_strip_chart,
    zscore_benchmark_chart,
    team_torque_by_position,
    team_asymmetry_rank,
    team_torque_distribution,
    team_risk_matrix,
    torque_bw_chart,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chase Chavez | Hip Strength",
    page_icon="assets/CC_Node_Primary_Blue.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Chase Chavez brand — global ─────────────────────── */
html, body, [class*="css"] {
    font-family: Arial, Helvetica, sans-serif;
}

/* ── Theme tokens — light (default) ─────────────────── */
:root {
    --cc-card-bg:       #FFFFFF;
    --cc-card-border:   #E2E8F0;
    --cc-row-sep:       rgba(0,0,0,0.09);
    --cc-label-dim:     #6B7280;
    --cc-label-dimmer:  #9CA3AF;
    --cc-right-val:     #1F2937;
    --cc-sub-text:      #6B7280;
    --cc-th-metric:     #374151;
    --cc-th-left:       #005F87;
    --cc-th-right:      #374151;
}

/* Dark — system preference */
@media (prefers-color-scheme: dark) {
    :root {
        --cc-card-bg:       #0D1B24;
        --cc-card-border:   #1a2d3a;
        --cc-row-sep:       #1a2d3a;
        --cc-label-dim:     #4A6475;
        --cc-label-dimmer:  #3D5A6B;
        --cc-right-val:     #C5D8E5;
        --cc-sub-text:      #4A6475;
        --cc-th-metric:     #3D5A6B;
        --cc-th-left:       #00A3A3;
        --cc-th-right:      #7A95A8;
    }
}

/* Dark — Streamlit theme toggle */
[data-theme="dark"] {
    --cc-card-bg:       #0D1B24;
    --cc-card-border:   #1a2d3a;
    --cc-row-sep:       #1a2d3a;
    --cc-label-dim:     #4A6475;
    --cc-label-dimmer:  #3D5A6B;
    --cc-right-val:     #C5D8E5;
    --cc-sub-text:      #4A6475;
    --cc-th-metric:     #3D5A6B;
    --cc-th-left:       #00A3A3;
    --cc-th-right:      #7A95A8;
}

/* ── Metric tiles ────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--secondary-background-color, #F5F4F2);
    border-radius: 8px;
    padding: 12px 14px;
    border: 1px solid var(--cc-card-border);
    border-top: 3px solid #005F87;
}
[data-testid="stMetric"] label {
    font-size: 0.70rem !important;
    color: var(--cc-label-dim, #565A5C) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
    color: #005F87 !important;
    font-weight: 700 !important;
}
[data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

/* ── Layout ──────────────────────────────────────────── */
.block-container { padding-top: 1rem; }

/* ── Section headings ────────────────────────────────── */
h3 {
    color: #005F87 !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

/* ── Sidebar — always dark regardless of Streamlit theme ─ */
[data-testid="stSidebar"] {
    background: #141414 !important;
}
/* Text elements */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] .stMarkdown {
    color: #F5F4F2 !important;
}
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] .stMarkdown p {
    color: #8CA3B5 !important;
}
[data-testid="stSidebar"] hr {
    border-color: #1a2d3a !important;
}
/* Selectbox inputs */
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="select"] > div > div,
[data-testid="stSidebar"] [data-baseweb="popover"] {
    background-color: #1a2d3a !important;
    border-color: #253B4A !important;
    color: #F5F4F2 !important;
}
/* File uploader */
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"],
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] > div {
    background-color: #1a2d3a !important;
    border-color: #253B4A !important;
}
/* Expander */
[data-testid="stSidebar"] details,
[data-testid="stSidebar"] details > div,
[data-testid="stSidebar"] [data-testid="stExpander"],
[data-testid="stSidebar"] [data-testid="stExpander"] > div {
    background-color: rgba(26,45,58,0.6) !important;
    border-color: #253B4A !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    color: #F5F4F2 !important;
    font-weight: 600 !important;
}
/* Buttons */
[data-testid="stSidebar"] button {
    background-color: #1a2d3a !important;
    border-color: #253B4A !important;
    color: #F5F4F2 !important;
}
[data-testid="stSidebar"] button:hover {
    background-color: #253B4A !important;
    border-color: #00A3A3 !important;
}
/* Toggle */
[data-testid="stSidebar"] [data-testid="stToggle"] label {
    color: #8CA3B5 !important;
}
/* Metric tiles in sidebar */
[data-testid="stSidebar"] [data-testid="stMetric"] {
    background: #0D1B24 !important;
    border-top-color: #00A3A3 !important;
    border-color: #1a2d3a !important;
}
[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    color: #00A3A3 !important;
}

/* ── Tab bar — inherit page bg so it works in both modes */
[data-baseweb="tab-list"],
div[data-testid="stTabs"] > div:first-child {
    position: sticky !important;
    top: 3.75rem !important;
    z-index: 999 !important;
    background-color: var(--background-color, #FFFFFF) !important;
    padding-top: 6px !important;
    padding-bottom: 2px !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.10) !important;
}
[data-baseweb="tab"] {
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
}
[aria-selected="true"] [data-baseweb="tab"] {
    color: #005F87 !important;
}

/* ── Dividers ────────────────────────────────────────── */
hr { border-color: var(--cc-card-border, #E5E7EB) !important; }

/* ── Expander headers ────────────────────────────────── */
[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    color: #005F87 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Column display labels ─────────────────────────────────────────────────────
_COL_LABELS = {
    "date":                   "Date",
    "athlete_id":             "Athlete ID",
    "athlete_name":           "Athlete",
    "position":               "Position",
    "tier":                   "Tier",
    "jersey_number":          "Jersey #",
    "bodyweight_kg":          "Bodyweight (kg)",
    "hip_abd_left_n":         "Hip ABD Left (N)",
    "hip_abd_right_n":        "Hip ABD Right (N)",
    "hip_abd_left_n_per_kg":  "ABD Left (N/kg)",
    "hip_abd_right_n_per_kg": "ABD Right (N/kg)",
    "hip_abd_left_nm_per_kg": "ABD Left (Nm/kg)",
    "hip_abd_right_nm_per_kg":"ABD Right (Nm/kg)",
    "hip_abd_asym_pct":       "ABD Asym (%)",
    "hip_abd_asym_flag":      "ABD Flag",
    "hip_add_left_n":         "Hip ADD Left (N)",
    "hip_add_right_n":        "Hip ADD Right (N)",
    "hip_add_left_n_per_kg":  "ADD Left (N/kg)",
    "hip_add_right_n_per_kg": "ADD Right (N/kg)",
    "hip_add_left_nm_per_kg": "ADD Left (Nm/kg)",
    "hip_add_right_nm_per_kg":"ADD Right (Nm/kg)",
    "hip_add_asym_pct":       "ADD Asym (%)",
    "hip_add_asym_flag":      "ADD Flag",
    "dominant":               "Dominant Side",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
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
    """Return '#NN  ·  ' prefix if jersey number present, else empty string."""
    raw = latest.get("jersey_number", latest.get("jersey", None))
    if raw is None:
        return ""
    try:
        return f"#{int(float(raw))}&nbsp;&nbsp;&middot;&nbsp;&nbsp;"
    except (TypeError, ValueError):
        return ""


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def get_file_data():
    raw, errors = load_isometric_data("data/raw/")
    if raw.empty:
        return raw, errors
    return process_metrics(raw), errors


file_df, load_errors = get_file_data()


def _build_df() -> pd.DataFrame:
    """Combine file-based + uploaded force CSVs, then fuse BW and roster."""
    # Gather all force DataFrames
    force_dfs = [d for d in ([file_df] + st.session_state.get("ss_force_dfs", []))
                 if d is not None and not d.empty]

    if not force_dfs:
        return pd.DataFrame()

    combined = pd.concat(force_dfs, ignore_index=True)
    dedup_key = (["date", "athlete_id"] if "athlete_id" in combined.columns
                 else ["date", "athlete_name"])
    combined = (combined
                .drop_duplicates(subset=dedup_key)
                .sort_values(["athlete_name", "date"])
                .reset_index(drop=True))

    # Fuse bodyweight + roster
    bw_df     = st.session_state.get("ss_bw_df")
    roster_df = st.session_state.get("ss_roster_df")

    merged, warn = merge_all_sources(combined, bw_df, roster_df)
    st.session_state["ss_merge_warnings"] = warn

    result = process_metrics(merged)

    # Apply manual position → tier overrides
    overrides: dict = st.session_state.get("ss_pos_tier_override", {})
    if overrides and "position" in result.columns:
        def _tier(pos):
            return overrides.get(str(pos), POSITION_TIERS.get(str(pos), "Other"))
        result["tier"] = result["position"].apply(_tier)

    return result


df = _build_df()


# ── SIDEBAR — Part 1: selection controls ─────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style="padding: 8px 0 18px 0; text-align: center;">

  <!-- Node Mark — reversed (white arms, teal terminals) on Near Black -->
  <img src="data:image/png;base64,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" width="72"
       style="display:block; margin: 0 auto 14px;" alt="Chase Chavez Node Mark"/>

  <!-- Wordmark — Lockup A Reversed per brand book -->
  <div style="font-size:1.05rem; font-weight:700; color:#FFFFFF;
              letter-spacing:0.5px; font-family:Arial,Helvetica,sans-serif;
              line-height:1.05;">Chase Chavez</div>
  <div style="font-size:0.68rem; font-weight:700; color:#00A3A3;
              letter-spacing:3px; font-family:Arial,Helvetica,sans-serif;
              margin-top:4px;">PT, DPT, SCS</div>
  <div style="width:36px; height:1.5px; background:#00A3A3; margin:10px auto;"></div>
  <div style="font-size:0.60rem; color:rgba(255,255,255,0.45); letter-spacing:1.5px;
              text-transform:uppercase; font-family:Arial,Helvetica,sans-serif;
              line-height:1.5;">Board-Certified<br>Sports Physical Therapist</div>
  <div style="font-size:0.60rem; color:rgba(0,163,163,0.85); letter-spacing:1.5px;
              font-style:italic; font-family:Arial,Helvetica,sans-serif;
              margin-top:6px;">Build what stays.</div>

  <div style="margin-top:14px; padding-top:10px; border-top:1px solid #1a2d3a;
              font-size:0.62rem; color:#565A5C; letter-spacing:0.1em;
              text-transform:uppercase; font-family:Arial,Helvetica,sans-serif;">
    Hip Strength Monitor &nbsp;·&nbsp; Football
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Theme toggle ─────────────────────────────────────────────────────────
    _dark = st.toggle(
        "🌙 Dark charts",
        value=st.session_state.get("dark_mode", True),
        key="dark_mode_toggle",
        help="Switch chart backgrounds between dark and light",
    )
    st.session_state["dark_mode"] = _dark

    st.divider()

    # ── Smart multi-source uploader ───────────────────────────────────────────
    with st.expander("📤 Upload Data Sources", expanded=True):
        st.caption(
            "Drop any CSV — force output, body weight, or roster. "
            "System auto-detects type and merges automatically."
        )

        uploaded_files = st.file_uploader(
            "Drop CSV file(s)",
            type=["csv"],
            accept_multiple_files=True,
            key="smart_uploader",
        )

        if uploaded_files:
            new_force: list   = []
            new_bw_dfs: list  = []
            new_roster_dfs: list = []
            log: list         = []
            had_error         = False

            for f in uploaded_files:
                try:
                    raw = pd.read_csv(f)
                    csv_type = detect_csv_type(raw)

                    if csv_type == "force":
                        from src.etl.load_data import _detect_columns, _standardize
                        mapping = _detect_columns(raw.columns.tolist())
                        std = _standardize(raw, mapping)
                        new_force.append(std)
                        log.append((f.name, "🟢 Force", len(std), ""))

                    elif csv_type == "bodyweight":
                        bw, errs = load_bodyweight_csv(raw)
                        new_bw_dfs.append(bw)
                        log.append((f.name, "🔵 Body Weight", len(bw),
                                    "; ".join(errs) if errs else ""))

                    elif csv_type == "roster":
                        ros, errs = load_roster_csv(raw)
                        new_roster_dfs.append(ros)
                        log.append((f.name, "🟡 Roster", len(ros),
                                    "; ".join(errs) if errs else ""))

                    elif csv_type == "kangatech":
                        kt_df, kt_errs = load_kangatech_csv(raw)
                        if not kt_df.empty:
                            new_force.append(kt_df)
                            log.append((f.name, "🟣 KangaTech 360", len(kt_df),
                                        "; ".join(kt_errs) if kt_errs else ""))
                        else:
                            log.append((f.name, "❌ KangaTech parse failed", 0,
                                        "; ".join(kt_errs)))
                            had_error = True

                    else:
                        # Unknown — try force loader as best guess
                        from src.etl.load_data import _detect_columns, _standardize
                        mapping = _detect_columns(raw.columns.tolist())
                        if mapping:
                            std = _standardize(raw, mapping)
                            new_force.append(std)
                            log.append((f.name, "⚪ Unknown → Force?", len(std),
                                        "Type unclear; treated as force data."))
                        else:
                            log.append((f.name, "❌ Unknown", 0,
                                        "Cannot detect column types."))
                            had_error = True

                except Exception as exc:
                    log.append((f.name, "❌ Error", 0, str(exc)))
                    had_error = True

            # Persist to session state
            if new_force:
                st.session_state["ss_force_dfs"] = new_force
            if new_bw_dfs:
                combined_bw = pd.concat(new_bw_dfs, ignore_index=True)
                st.session_state["ss_bw_df"] = combined_bw
            if new_roster_dfs:
                combined_ros = pd.concat(new_roster_dfs, ignore_index=True)
                st.session_state["ss_roster_df"] = combined_ros

            # Upload log
            for fname, ftype, nrows, note in log:
                color = "#16A34A" if "Force" in ftype or "Body" in ftype or "Roster" in ftype else "#EF4444"
                st.markdown(
                    f'<div style="font-size:0.78rem; margin:2px 0;">'
                    f'<b style="color:{color};">{ftype}</b>&nbsp;&nbsp;'
                    f'<code>{fname}</code>&nbsp;'
                    f'<span style="color:var(--cc-label-dim);">{nrows} rows</span>'
                    + (f'<br><span style="color:#FFB400; font-size:0.70rem;">⚠ {note}</span>' if note else "")
                    + "</div>",
                    unsafe_allow_html=True,
                )

            if new_force or new_bw_dfs or new_roster_dfs:
                st.rerun()

        # Status badges for what's currently loaded
        loaded_parts = []
        if st.session_state.get("ss_force_dfs"):
            n = sum(len(d) for d in st.session_state["ss_force_dfs"])
            loaded_parts.append(f"🟢 Force ({n} rows)")
        if st.session_state.get("ss_bw_df") is not None:
            n = len(st.session_state["ss_bw_df"])
            loaded_parts.append(f"🔵 BW ({n} rows)")
        if st.session_state.get("ss_roster_df") is not None:
            n = len(st.session_state["ss_roster_df"])
            loaded_parts.append(f"🟡 Roster ({n} rows)")
        if loaded_parts:
            st.caption("  ·  ".join(loaded_parts))

        # Clear uploads button
        if any(k in st.session_state for k in
               ("ss_force_dfs", "ss_bw_df", "ss_roster_df")):
            if st.button("🗑 Clear all uploads", use_container_width=True):
                for k in ("ss_force_dfs", "ss_bw_df", "ss_roster_df",
                          "ss_merge_warnings"):
                    st.session_state.pop(k, None)
                st.rerun()

    # ── Merge warnings ────────────────────────────────────────────────────────
    merge_warns = st.session_state.get("ss_merge_warnings", [])
    if merge_warns:
        with st.expander(f"🔗 {len(merge_warns)} merge note(s)"):
            for w in merge_warns:
                st.caption(w)

    if load_errors:
        with st.expander(f"⚠️ {len(load_errors)} load warning(s)"):
            for e in load_errors:
                st.warning(e)

    if df.empty:
        st.info("No data loaded yet.")
        st.markdown("**Drop any of these CSV types above:**")
        st.markdown("""
- **Force CSV** — athlete name/ID + date + `hip_abd/add_left/right_n`
- **Body Weight CSV** — athlete name + date + weight (kg or lbs)
- **Roster CSV** — athlete name + position + jersey #
        """)
        st.stop()

    # ── Position → Tier override ──────────────────────────────────────────────
    known_positions = sorted(df["position"].dropna().unique()) if not df.empty else []
    unknown_pos = [p for p in known_positions
                   if str(p) not in POSITION_TIERS and str(p) not in ("UNK", "Other", "nan")]
    if unknown_pos:
        with st.expander(f"⚙️ Assign tiers ({len(unknown_pos)} unknown)"):
            st.caption("Positions not in default POSITION_TIERS — assign manually.")
            overrides = dict(st.session_state.get("ss_pos_tier_override", {}))
            for pos in unknown_pos:
                current = overrides.get(pos, "Other")
                chosen = st.selectbox(
                    f"`{pos}` tier",
                    ["Skill", "Mid", "Big", "Other"],
                    index=["Skill", "Mid", "Big", "Other"].index(current)
                          if current in ["Skill", "Mid", "Big", "Other"] else 3,
                    key=f"tier_override_{pos}",
                )
                overrides[pos] = chosen
            st.session_state["ss_pos_tier_override"] = overrides

    # Tier → Position → Athlete cascade
    tiers    = ["All", "Skill", "Mid", "Big"]
    sel_tier = st.selectbox("Position tier", tiers)

    if sel_tier == "All":
        tier_pool = df
    else:
        tier_pool = df[df["tier"] == sel_tier] if "tier" in df.columns else df

    positions = ["All"] + sorted(tier_pool["position"].dropna().unique())
    sel_pos   = st.selectbox("Position", positions)
    pos_pool  = tier_pool if sel_pos == "All" else tier_pool[tier_pool["position"] == sel_pos]

    names    = sorted(pos_pool["athlete_name"].unique())
    sel_name = st.selectbox("Athlete", names)
    sel_id   = df[df["athlete_name"] == sel_name]["athlete_id"].iloc[0]


# ── Athlete-scoped data (defined BEFORE second sidebar block) ─────────────────
adf    = df[df["athlete_id"] == sel_id].sort_values("date").reset_index(drop=True)
latest = adf.iloc[-1]
prev   = adf.iloc[-2] if len(adf) >= 2 else None


# ── SIDEBAR — Part 2: actions (adf now defined) ───────────────────────────────
with st.sidebar:
    st.divider()
    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("Auto-refreshes every 30 s")
    st.divider()
    st.caption(
        "**Body weight must be recorded each session** for accurate "
        "torque normalisation (Nm/kg). Ensure athletes weigh in before "
        "every test — do not carry forward prior values."
    )

    st.divider()
    if st.button("📋 Generate Team Report", use_container_width=True, key="gen_team_pdf"):
        with st.spinner("Building team PDF..."):
            from src.export.pdf_report import generate_team_pdf
            team_pdf_bytes = generate_team_pdf(df)
        st.session_state["team_pdf"] = team_pdf_bytes
        st.rerun()

    if "team_pdf" in st.session_state:
        st.download_button(
            label="⬇️ Download Team PDF",
            data=st.session_state["team_pdf"],
            file_name=f"team_hip_strength_{pd.Timestamp.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="dl_team_pdf",
        )

    if st.button("📄 Generate PDF Report", use_container_width=True, key="gen_pdf"):
        with st.spinner("Building PDF..."):
            from src.export.pdf_report import generate_athlete_pdf
            pdf_bytes = generate_athlete_pdf(sel_name, adf, df, sel_id, latest, prev)
        st.session_state[f"pdf_{sel_id}"] = pdf_bytes
        st.rerun()

    pdf_key = f"pdf_{sel_id}"
    if pdf_key in st.session_state:
        fname = (
            f"{sel_name.replace(' ', '_')}_hip_strength"
            f"_{pd.Timestamp.now().strftime('%Y%m%d')}.pdf"
        )
        st.download_button(
            label="⬇️ Download PDF",
            data=st.session_state[pdf_key],
            file_name=fname,
            mime="application/pdf",
            use_container_width=True,
        )
        st.caption("Re-generate after switching athlete")


# ── Movement section helper ───────────────────────────────────────────────────
def _movement_section(mov: str, label: str):
    st.subheader(label)

    l_n   = latest.get(f"hip_{mov}_left_n",          np.nan)
    r_n   = latest.get(f"hip_{mov}_right_n",         np.nan)
    l_rel = latest.get(f"hip_{mov}_left_n_per_kg",   np.nan)
    r_rel = latest.get(f"hip_{mov}_right_n_per_kg",  np.nan)
    l_nm  = latest.get(f"hip_{mov}_left_nm_per_kg",  np.nan)
    r_nm  = latest.get(f"hip_{mov}_right_nm_per_kg", np.nan)
    asym  = latest.get(f"hip_{mov}_asym_pct",        np.nan)
    flag  = latest.get(f"hip_{mov}_asym_flag",       False)

    try:
        asym_f = float(asym)
    except (TypeError, ValueError):
        asym_f = None

    # ── N → lb conversion ────────────────────────────────────────────────────
    def _lb(n_val):
        try:
            return _fmt(float(n_val) * 0.224809, ".1f")
        except Exception:
            return "-"

    # ── Inline delta chip (for HTML card) ────────────────────────────────────
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

    # ── Asymmetry badge colours ───────────────────────────────────────────────
    if flag or (asym_f is not None and asym_f > 15):
        ac, abg, abdr = "#EF4444", "rgba(239,68,68,0.15)",  "#EF444450"
    elif asym_f is not None and asym_f > 10:
        ac, abg, abdr = "#FFB400", "rgba(255,180,0,0.15)",  "#FFB40050"
    else:
        ac, abg, abdr = "#22C55E", "rgba(34,197,94,0.12)",  "#22C55E50"

    # ── Table row builder ─────────────────────────────────────────────────────
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
      ASYM {_fmt(asym)}%{asym_dchip}
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

    # ── Asymmetry status banner ───────────────────────────────────────────────
    if flag:
        st.error(
            f"ASYMMETRY FLAG — {_fmt(asym)}% exceeds 15% clinical threshold. "
            "Review recommended."
        )
    elif asym_f is not None and asym_f > 10:
        st.warning(
            f"MONITOR — {_fmt(asym)}% asymmetry above 10% warning level. "
            "Track closely."
        )
    else:
        st.success(f"Symmetry within normal range ({_fmt(asym)}%).")



# ── Apply chart theme ────────────────────────────────────────────────────────
_iso_mod.DARK_MODE = st.session_state.get("dark_mode", True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_profile, tab_roster = st.tabs(["📊 Athlete Profile", "👥 Roster Overview"])

# ══════════════════════════════════════════════════════════════════════════════
# ATHLETE PROFILE
# ══════════════════════════════════════════════════════════════════════════════
with tab_profile:

    # ── Branded athlete header ───────────────────────────────────────────────
    pos_str  = str(latest.get("position", "-"))
    tier_str = str(latest.get("tier", "-")) if "tier" in latest.index else "-"
    bw_kg_val = latest.get("bodyweight_kg", np.nan)
    bw_str   = _fmt(bw_kg_val, ".1f")
    try:
        bw_lbs_str = _fmt(float(bw_kg_val) * 2.20462, ".1f")
    except Exception:
        bw_lbs_str = "-"
    bw_display = f"{bw_str} kg / {bw_lbs_str} lbs" if bw_str != "-" else "-"
    date_str = pd.to_datetime(latest["date"]).strftime("%d %b %Y")
    ns       = len(adf)
    jsy      = _jersey_str(latest)

    st.markdown(f"""
<div style="
    background: #005F87;
    border-radius: 10px;
    padding: 20px 24px 16px 24px;
    margin-bottom: 1.4rem;
">
  <div style="
      font-size: 1.75rem;
      font-weight: 700;
      color: #FFFFFF;
      font-family: Arial, Helvetica, sans-serif;
      letter-spacing: -0.3px;
      line-height: 1.2;
      margin-bottom: 6px;
  ">{sel_name}</div>
  <div style="
      font-size: 1.08rem;
      font-weight: 600;
      color: #00A3A3;
      font-family: 'Courier New', Courier, monospace;
      letter-spacing: 0.05em;
      margin-bottom: 2px;
  ">{jsy}{pos_str}&nbsp;&nbsp;&middot;&nbsp;&nbsp;{tier_str}&nbsp;&nbsp;&middot;&nbsp;&nbsp;{bw_display}</div>
  <div style="
      border-top: 1px solid rgba(255,255,255,0.18);
      margin-top: 14px;
      padding-top: 11px;
      display: flex;
      gap: 2.8rem;
      flex-wrap: wrap;
  ">
    <span style="color:#C5DBE8; font-size:0.95rem; font-family:Arial,sans-serif;">
      Latest Test:&nbsp;<b style="color:#00A3A3; font-size:1rem;">{date_str}</b>
    </span>
    <span style="color:#C5DBE8; font-size:0.95rem; font-family:Arial,sans-serif;">
      Total Sessions:&nbsp;<b style="color:#00A3A3; font-size:1rem;">{ns}</b>
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Body weight staleness warning ────────────────────────────────────────
    bw_vals = adf["bodyweight_kg"].dropna()
    if len(bw_vals) > 1 and bw_vals.nunique() == 1:
        st.warning(
            "Body weight is identical across all sessions. "
            "Verify athletes are weighed at each test — static BW produces "
            "inaccurate torque normalisation."
        )

    _mov_col1, _mov_col2 = st.columns(2)
    with _mov_col1:
        _movement_section("abd", "Hip Abduction")
    with _mov_col2:
        _movement_section("add", "Hip Adduction")

    st.divider()
    st.subheader("Force / Torque Trends")
    _tc1, _tc2 = st.columns(2)
    with _tc1:
        _fig_abd = torque_bw_chart(adf, "abd")
        if _fig_abd.data:
            st.plotly_chart(_fig_abd, use_container_width=True)
        else:
            st.caption("ABD trend unavailable — check bodyweight recorded.")
    with _tc2:
        _fig_add = torque_bw_chart(adf, "add")
        if _fig_add.data:
            st.plotly_chart(_fig_add, use_container_width=True)
        else:
            st.caption("ADD trend unavailable — check bodyweight recorded.")

    st.divider()
    st.subheader("Asymmetry Over Time")
    st.plotly_chart(asymmetry_chart(adf), use_container_width=True)

    st.divider()
    st.subheader("Position Benchmarks")

    # Z-score: all metrics vs positional cohort (full width)
    zfig = zscore_benchmark_chart(df, sel_id)
    if zfig.data:
        st.plotly_chart(zfig, use_container_width=True)
    else:
        st.caption("Z-score chart requires 3+ athletes in the same position.")

    # Percentile strips: one per movement (side by side)
    pc1, pc2 = st.columns(2)
    with pc1:
        pfig_abd = percentile_strip_chart(df, sel_id, "abd")
        if pfig_abd.data:
            st.plotly_chart(pfig_abd, use_container_width=True)
        else:
            st.caption("Percentile strip requires 3+ athletes in position (ABD).")
    with pc2:
        pfig_add = percentile_strip_chart(df, sel_id, "add")
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


# ══════════════════════════════════════════════════════════════════════════════
# ROSTER OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_roster:

    # ── Latest snapshot per athlete ───────────────────────────────────────────
    roster     = df.sort_values("date").groupby("athlete_id").last().reset_index()
    now        = pd.Timestamp.now()
    tested_14d = int((roster["date"] >= now - pd.Timedelta(days=14)).sum())
    abd_flags  = int(roster.get("hip_abd_asym_flag", pd.Series(dtype=bool)).sum())
    add_flags  = int(roster.get("hip_add_asym_flag", pd.Series(dtype=bool)).sum())

    abd_warn = int(
        ((roster.get("hip_abd_asym_pct", pd.Series(dtype=float)) > 10) &
         (roster.get("hip_abd_asym_pct", pd.Series(dtype=float)) <= 15)).sum()
    )
    add_warn = int(
        ((roster.get("hip_add_asym_pct", pd.Series(dtype=float)) > 10) &
         (roster.get("hip_add_asym_pct", pd.Series(dtype=float)) <= 15)).sum()
    )

    # ── Athlete vs Team comparison ────────────────────────────────────────────
    st.subheader("Athlete vs Team")
    cmp_names = sorted(roster["athlete_name"].dropna().unique())
    cmp_sel   = st.selectbox("Search athlete", cmp_names, key="roster_compare",
                              help="Compare one athlete's metrics against full team distribution")
    cmp_id    = df[df["athlete_name"] == cmp_sel]["athlete_id"].iloc[0]
    fig_cmp   = athlete_vs_team_chart(df, cmp_id)
    if fig_cmp.data:
        st.caption(
            "Bar = team min–max · Dark box = IQR · Tick = team median · "
            "Diamond = selected athlete. Asym diamonds: teal ≤10 % · amber 10–15 % · red >15 %"
        )
        st.plotly_chart(fig_cmp, use_container_width=True)
    else:
        st.caption("Need N/kg data to compare. Verify bodyweight is recorded.")

    st.divider()

    # ── Tier / position filter (applies to charts + table) ───────────────────
    rf1, rf2 = st.columns([1, 5])
    with rf1:
        roster_tier = st.selectbox("Filter tier",
                                   ["All", "Skill", "Mid", "Big"],
                                   key="roster_tier")
    df_team  = df if roster_tier == "All" else (
        df[df["tier"] == roster_tier] if "tier" in df.columns else df
    )
    roster_v = roster if roster_tier == "All" else (
        roster[roster["tier"] == roster_tier] if "tier" in roster.columns else roster
    )

    # ── Summary KPIs ─────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Athletes",         len(roster_v))
    k2.metric("Tested (14 d)",    tested_14d)
    k3.metric("ABD Flag (>15%)",  abd_flags,  delta=None)
    k4.metric("ABD Warn (10-15%)",abd_warn)
    k5.metric("ADD Flag (>15%)",  add_flags)
    k6.metric("ADD Warn (10-15%)",add_warn)

    st.divider()

    # ── SECTION 1: Torque output by position tier ─────────────────────────────
    st.subheader("Torque Output by Position Group")
    tc1, tc2 = st.columns(2)
    with tc1:
        fig_t_abd = team_torque_by_position(df_team, "abd")
        if fig_t_abd.data:
            st.plotly_chart(fig_t_abd, use_container_width=True)
        else:
            st.caption("Need tier data + Nm/kg for this chart.")
    with tc2:
        fig_t_add = team_torque_by_position(df_team, "add")
        if fig_t_add.data:
            st.plotly_chart(fig_t_add, use_container_width=True)
        else:
            st.caption("Need tier data + Nm/kg for this chart.")

    # ── SECTION 2: Torque distribution (strip + box per tier) ─────────────────
    st.subheader("Torque Distribution by Tier")
    td1, td2 = st.columns(2)
    with td1:
        fig_d_abd = team_torque_distribution(df_team, "abd")
        if fig_d_abd.data:
            st.plotly_chart(fig_d_abd, use_container_width=True)
    with td2:
        fig_d_add = team_torque_distribution(df_team, "add")
        if fig_d_add.data:
            st.plotly_chart(fig_d_add, use_container_width=True)

    # ── SECTION 3: Risk matrix ────────────────────────────────────────────────
    st.subheader("Risk Matrix — Strength vs Asymmetry")
    st.caption(
        "X = avg torque across all movements (Nm/kg). "
        "Y = highest asymmetry across ABD + ADD. "
        "Bubble size = body weight. "
        "Top-left quadrant (low strength + high asym) = highest intervention priority."
    )
    fig_risk = team_risk_matrix(df_team)
    if fig_risk.data:
        st.plotly_chart(fig_risk, use_container_width=True)
    else:
        st.caption("Need Nm/kg + asymmetry data for risk matrix.")

    st.divider()

    # ── SECTION 4: Asymmetry ranking ─────────────────────────────────────────
    st.subheader("Asymmetry Index — All Athletes")
    ac1, ac2 = st.columns(2)
    with ac1:
        fig_a_abd = team_asymmetry_rank(df_team, "abd")
        if fig_a_abd.data:
            st.plotly_chart(fig_a_abd, use_container_width=True)
        else:
            st.caption("No ABD asymmetry data.")
    with ac2:
        fig_a_add = team_asymmetry_rank(df_team, "add")
        if fig_a_add.data:
            st.plotly_chart(fig_a_add, use_container_width=True)
        else:
            st.caption("No ADD asymmetry data.")

    st.divider()

    # ── SECTION 5: Full roster table ─────────────────────────────────────────
    st.subheader("Full Roster — Latest Values")

    # Build a display copy with BW as "X.X kg / Y.Y lbs" string
    roster_v_display = roster_v.copy()
    if "bodyweight_kg" in roster_v_display.columns:
        roster_v_display["bodyweight_display"] = roster_v_display["bodyweight_kg"].apply(
            lambda v: f"{v:.1f} kg / {v * 2.20462:.1f} lbs"
            if pd.notna(v) else "-"
        )

    show_cols = [c for c in (
        "athlete_name", "position", "tier", "date",
        "bodyweight_display",
        "hip_abd_left_nm_per_kg", "hip_abd_right_nm_per_kg", "hip_abd_asym_pct",
        "hip_abd_asym_flag",
        "hip_add_left_nm_per_kg", "hip_add_right_nm_per_kg", "hip_add_asym_pct",
        "hip_add_asym_flag",
    ) if c in roster_v_display.columns]

    flag_cols_orig = [c for c in (
        "hip_abd_asym_pct", "hip_add_asym_pct",
        "hip_abd_asym_flag", "hip_add_asym_flag",
    ) if c in roster_v_display.columns]

    _COL_LABELS_DISPLAY = {**_COL_LABELS, "bodyweight_display": "Bodyweight"}
    col_map           = {c: _COL_LABELS_DISPLAY.get(c, c) for c in show_cols}
    _sort_col = "hip_abd_asym_pct" if "hip_abd_asym_pct" in roster_v_display.columns else show_cols[0]
    roster_display    = (
        roster_v_display[show_cols]
        .sort_values(_sort_col, ascending=False)
        .rename(columns=col_map)
    )
    flag_cols_display = [col_map.get(c, c) for c in flag_cols_orig]

    # Numeric columns to format to 1 decimal (exclude text/date/bool/pre-formatted)
    _skip = {"athlete_name", "position", "tier", "date",
             "hip_abd_asym_flag", "hip_add_asym_flag", "bodyweight_display"}
    num_display_cols = [col_map[c] for c in show_cols
                        if c not in _skip and col_map[c] in roster_display.columns]

    def _color_cell(val):
        if isinstance(val, (bool, np.bool_)) and val:
            return "background-color: #fee2e2; color: #b91c1c"
        if isinstance(val, float):
            if val > 15:
                return "background-color: #fee2e2; color: #b91c1c"
            if val > 10:
                return "background-color: #fff3c3; color: #b46e00"
        return ""

    styled = (
        roster_display.style
        .map(_color_cell, subset=flag_cols_display)
        .format("{:.1f}", subset=num_display_cols, na_rep="-")
    )
    st.dataframe(styled, use_container_width=True, height=520)

# ── Branded footer ────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem; padding:20px 28px; background:#141414; border-radius:8px;
            display:flex; justify-content:space-between; align-items:center;
            flex-wrap:wrap; gap:10px;">

  <!-- Left: mark + wordmark horizontal lockup -->
  <div style="display:flex; align-items:center; gap:14px;">
    <img src="data:image/png;base64,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" width="42"
         style="flex-shrink:0;" alt="Chase Chavez Node Mark"/>
    <div style="border-left:1.5px solid rgba(0,163,163,0.5); padding-left:14px;">
      <div style="font-size:0.82rem; font-weight:700; color:#FFFFFF; letter-spacing:0.5px;
                  font-family:Arial,Helvetica,sans-serif; line-height:1.1;">Chase Chavez</div>
      <div style="font-size:0.65rem; font-weight:700; color:#00A3A3; letter-spacing:3px;
                  font-family:Arial,Helvetica,sans-serif; margin-top:2px;">PT, DPT, SCS</div>
    </div>
  </div>

  <!-- Right: context label -->
  <div style="font-size:0.68rem; color:#565A5C; letter-spacing:0.05em;
              font-family:Arial,Helvetica,sans-serif;">
    Hip Isometric Strength &nbsp;·&nbsp; Football Performance
  </div>
</div>
""", unsafe_allow_html=True)
