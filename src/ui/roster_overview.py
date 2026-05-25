"""Roster overview tab — render function."""
import numpy as np
import pandas as pd
import streamlit as st

from src.viz.isometric_plots import (
    athlete_vs_team_chart,
    team_asymmetry_rank,
    team_risk_matrix,
    team_torque_by_position,
    team_torque_distribution,
)

_COL_LABELS = {
    "athlete_name":            "Athlete",
    "position":                "Position",
    "tier":                    "Tier",
    "date":                    "Date",
    "bodyweight_display":      "Bodyweight",
    "hip_abd_left_nm_per_kg":  "ABD Left (Nm/kg)",
    "hip_abd_right_nm_per_kg": "ABD Right (Nm/kg)",
    "hip_abd_asym_pct":        "ABD Asym (%)",
    "hip_abd_asym_flag":       "ABD Flag",
    "hip_add_left_nm_per_kg":  "ADD Left (Nm/kg)",
    "hip_add_right_nm_per_kg": "ADD Right (Nm/kg)",
    "hip_add_asym_pct":        "ADD Asym (%)",
    "hip_add_asym_flag":       "ADD Flag",
    "hip_abd_add_ratio_left":  "ABD:ADD L",
    "hip_abd_add_ratio_right": "ABD:ADD R",
}


def render_roster_overview(
    df: pd.DataFrame,
    dark_mode: bool = True,
    flag_pct: float = 15.0,
    warn_pct: float = 10.0,
) -> None:
    # ── Latest snapshot per athlete ───────────────────────────────────────────
    roster     = df.sort_values("date").groupby("athlete_id").last().reset_index()
    now        = pd.Timestamp.now()
    tested_14d = int((roster["date"] >= now - pd.Timedelta(days=14)).sum())

    # Live KPI counts — use current thresholds, not stored boolean flags
    if "hip_abd_asym_pct" in roster.columns:
        abd_flags = int((roster["hip_abd_asym_pct"] > flag_pct).sum())
        abd_warn  = int(((roster["hip_abd_asym_pct"] > warn_pct) &
                         (roster["hip_abd_asym_pct"] <= flag_pct)).sum())
    else:
        abd_flags = abd_warn = 0

    if "hip_add_asym_pct" in roster.columns:
        add_flags = int((roster["hip_add_asym_pct"] > flag_pct).sum())
        add_warn  = int(((roster["hip_add_asym_pct"] > warn_pct) &
                         (roster["hip_add_asym_pct"] <= flag_pct)).sum())
    else:
        add_flags = add_warn = 0

    # ── Athlete vs Team comparison ────────────────────────────────────────────
    st.subheader("Athlete vs Team")
    cmp_names = sorted(roster["athlete_name"].dropna().unique())
    cmp_sel   = st.selectbox(
        "Search athlete", cmp_names, key="roster_compare",
        help="Compare one athlete's metrics against full team distribution",
    )
    _cmp_id_s = df[df["athlete_name"] == cmp_sel]["athlete_id"]
    if _cmp_id_s.empty:
        st.error(f"No data found for '{cmp_sel}'.")
        return
    cmp_id    = _cmp_id_s.iloc[0]
    fig_cmp   = athlete_vs_team_chart(df, cmp_id, dark_mode=dark_mode,
                                       flag_pct=flag_pct, warn_pct=warn_pct)
    if fig_cmp.data:
        st.caption(
            "Bar = team min–max · Dark box = IQR · Tick = team median · "
            "Diamond = selected athlete. "
            f"Teal ≤{warn_pct:.0f}% · Amber {warn_pct:.0f}–{flag_pct:.0f}% · Red >{flag_pct:.0f}%"
        )
        st.plotly_chart(fig_cmp, use_container_width=True)
    else:
        st.caption("Need N/kg data to compare. Verify bodyweight is recorded.")

    st.divider()

    # ── Tier / position filter ────────────────────────────────────────────────
    rf1, _ = st.columns([1, 5])
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
    k1.metric("Athletes",              len(roster_v))
    k2.metric("Tested (14 d)",         tested_14d)
    k3.metric(f"ABD Flag (>{flag_pct:.0f}%)",  abd_flags)
    k4.metric(f"ABD Warn ({warn_pct:.0f}–{flag_pct:.0f}%)", abd_warn)
    k5.metric(f"ADD Flag (>{flag_pct:.0f}%)",  add_flags)
    k6.metric(f"ADD Warn ({warn_pct:.0f}–{flag_pct:.0f}%)", add_warn)

    st.divider()

    # ── Torque output by position tier ───────────────────────────────────────
    st.subheader("Torque Output by Position Group")
    tc1, tc2 = st.columns(2)
    with tc1:
        fig_t_abd = team_torque_by_position(df_team, "abd", dark_mode=dark_mode)
        if fig_t_abd.data:
            st.plotly_chart(fig_t_abd, use_container_width=True)
        else:
            st.caption("Need tier data + Nm/kg for this chart.")
    with tc2:
        fig_t_add = team_torque_by_position(df_team, "add", dark_mode=dark_mode)
        if fig_t_add.data:
            st.plotly_chart(fig_t_add, use_container_width=True)
        else:
            st.caption("Need tier data + Nm/kg for this chart.")

    # ── Torque distribution ───────────────────────────────────────────────────
    st.subheader("Torque Distribution by Tier")
    td1, td2 = st.columns(2)
    with td1:
        fig_d_abd = team_torque_distribution(df_team, "abd", dark_mode=dark_mode)
        if fig_d_abd.data:
            st.plotly_chart(fig_d_abd, use_container_width=True)
    with td2:
        fig_d_add = team_torque_distribution(df_team, "add", dark_mode=dark_mode)
        if fig_d_add.data:
            st.plotly_chart(fig_d_add, use_container_width=True)

    # ── Risk matrix ───────────────────────────────────────────────────────────
    st.subheader("Risk Matrix — Strength vs Asymmetry")
    st.caption(
        "X = avg torque (Nm/kg). Y = highest asymmetry (ABD or ADD). "
        "Bubble = body weight. Top-left = highest intervention priority."
    )
    fig_risk = team_risk_matrix(df_team, dark_mode=dark_mode,
                                 flag_pct=flag_pct, warn_pct=warn_pct)
    if fig_risk.data:
        st.plotly_chart(fig_risk, use_container_width=True)
    else:
        st.caption("Need Nm/kg + asymmetry data for risk matrix.")

    st.divider()

    # ── Asymmetry ranking ─────────────────────────────────────────────────────
    st.subheader("Asymmetry Index — All Athletes")
    ac1, ac2 = st.columns(2)
    with ac1:
        fig_a_abd = team_asymmetry_rank(df_team, "abd", dark_mode=dark_mode,
                                         flag_pct=flag_pct, warn_pct=warn_pct)
        if fig_a_abd.data:
            st.plotly_chart(fig_a_abd, use_container_width=True)
        else:
            st.caption("No ABD asymmetry data.")
    with ac2:
        fig_a_add = team_asymmetry_rank(df_team, "add", dark_mode=dark_mode,
                                         flag_pct=flag_pct, warn_pct=warn_pct)
        if fig_a_add.data:
            st.plotly_chart(fig_a_add, use_container_width=True)
        else:
            st.caption("No ADD asymmetry data.")

    st.divider()

    # ── Full roster table ─────────────────────────────────────────────────────
    st.subheader("Full Roster — Latest Values")

    roster_v_display = roster_v.copy()
    if "bodyweight_kg" in roster_v_display.columns:
        roster_v_display["bodyweight_display"] = roster_v_display["bodyweight_kg"].apply(
            lambda v: f"{v:.1f} kg / {v * 2.20462:.1f} lbs" if pd.notna(v) else "-"
        )

    show_cols = [c for c in (
        "athlete_name", "position", "tier", "date",
        "bodyweight_display",
        "hip_abd_left_nm_per_kg", "hip_abd_right_nm_per_kg", "hip_abd_asym_pct",
        "hip_abd_asym_flag",
        "hip_add_left_nm_per_kg", "hip_add_right_nm_per_kg", "hip_add_asym_pct",
        "hip_add_asym_flag",
        "hip_abd_add_ratio_left", "hip_abd_add_ratio_right",
    ) if c in roster_v_display.columns]

    flag_cols_orig = [c for c in (
        "hip_abd_asym_pct", "hip_add_asym_pct",
        "hip_abd_asym_flag", "hip_add_asym_flag",
    ) if c in roster_v_display.columns]

    col_map = {c: _COL_LABELS.get(c, c) for c in show_cols}

    _sort_col = next(
        (c for c in ("hip_abd_asym_pct", show_cols[0]) if c in roster_v_display.columns),
        show_cols[0] if show_cols else "athlete_name",
    )
    roster_display = (
        roster_v_display[show_cols]
        .sort_values(_sort_col, ascending=False)
        .rename(columns=col_map)
    )
    flag_cols_display = [col_map.get(c, c) for c in flag_cols_orig]

    _skip = {"athlete_name", "position", "tier", "date",
             "hip_abd_asym_flag", "hip_add_asym_flag", "bodyweight_display"}
    num_display_cols = [col_map[c] for c in show_cols
                        if c not in _skip and col_map[c] in roster_display.columns]

    def _color_cell(val):
        if isinstance(val, (bool, np.bool_)) and val:
            return "background-color: #fee2e2; color: #b91c1c"
        if isinstance(val, float):
            if val > flag_pct:
                return "background-color: #fee2e2; color: #b91c1c"
            if val > warn_pct:
                return "background-color: #fff3c3; color: #b46e00"
        return ""

    styled = (
        roster_display.style
        .map(_color_cell, subset=flag_cols_display)
        .format("{:.1f}", subset=num_display_cols, na_rep="-")
    )
    st.dataframe(styled, use_container_width=True, height=520)
