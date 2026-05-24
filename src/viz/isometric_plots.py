import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.utils.metrics import fit_trend_line

# ── Theme flag — toggled by app.py before each render ─────────────────────────
DARK_MODE: bool = True

# Static semantic colours (same in both themes)
FLAG_CLR  = "#EF4444"   # Clinical red  (>15 % asymmetry)
WARN_CLR  = "#FFB400"   # Brand amber   (10–15 % zone)
OK_CLR    = "#22C55E"   # Green         (symmetry OK)

# Kept for external compatibility — use _pal() inside functions
LEFT_CLR  = "#00A3A3"
RIGHT_CLR = "#8DC4DB"


def _pal() -> dict:
    """Return the active colour palette based on DARK_MODE flag."""
    if DARK_MODE:
        return dict(
            bg         = "#0D1B24",
            grid       = "#1a2d3a",
            ax_line    = "#253B4A",
            font       = "#8CA3B5",
            title      = "#00A3A3",
            left       = "#00A3A3",
            right      = "#8DC4DB",
            peer       = "#2D4A5A",
            range_line = "#2D4A5A",
            iqr_fill   = "rgba(45,74,90,0.65)",
        )
    return dict(
        bg         = "#FFFFFF",
        grid       = "#E5E7EB",
        ax_line    = "#D1D5DB",
        font       = "#374151",
        title      = "#005F87",
        left       = "#00A3A3",
        right      = "#005F87",
        peer       = "#9CA3AF",
        range_line = "#9CA3AF",
        iqr_fill   = "rgba(150,170,185,0.35)",
    )


def _base() -> dict:
    p = _pal()
    return dict(
        template="none",
        font=dict(family="Arial, Helvetica, sans-serif", size=12, color=p["font"]),
        margin=dict(t=52, b=60, l=60, r=24),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5,
            font=dict(color=p["font"], size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor=p["bg"],
        plot_bgcolor=p["bg"],
    )


def _xaxis() -> dict:
    p = _pal()
    return dict(
        showgrid=False, linecolor=p["ax_line"], showline=True, zeroline=False,
        tickfont=dict(color=p["font"]), title_font=dict(color=p["font"]),
    )


def _yaxis() -> dict:
    p = _pal()
    return dict(
        gridcolor=p["grid"], linecolor=p["ax_line"], showline=True, zeroline=False,
        tickfont=dict(color=p["font"]), title_font=dict(color=p["font"]),
    )


def _layout(title: str, ytitle: str) -> dict:
    p = _pal()
    return {
        **_base(),
        "title": dict(
            text=title,
            font=dict(size=14, color=p["title"], family="Arial, Helvetica, sans-serif"),
        ),
        "xaxis": _xaxis(),
        "yaxis": {**_yaxis(), "title": ytitle},
    }


def _best_torque_col(df: pd.DataFrame, mov: str, side: str) -> tuple:
    """Return (col_name, unit_label) — prefer Nm/kg, fall back to N/kg."""
    nm = f"hip_{mov}_{side}_nm_per_kg"
    n  = f"hip_{mov}_{side}_n_per_kg"
    if nm in df.columns and df[nm].notna().any():
        return nm, "Nm/kg"
    if n in df.columns and df[n].notna().any():
        return n, "N/kg"
    return None, None


def _add_series(fig: go.Figure, adf: pd.DataFrame, col: str, label: str, color: str):
    p = _pal()
    if col not in adf.columns:
        return
    valid = adf[["date", col]].dropna()
    if valid.empty:
        return
    fig.add_trace(go.Scatter(
        x=valid["date"], y=valid[col], name=label,
        mode="lines+markers",
        line=dict(color=color, width=2.5),
        marker=dict(size=7, color=color, line=dict(width=1.5, color=p["bg"])),
    ))
    if len(valid) >= 3:
        trend, _ = fit_trend_line(valid["date"], valid[col])
        fig.add_trace(go.Scatter(
            x=valid["date"], y=trend.values, name=f"{label} trend",
            mode="lines",
            line=dict(color=color, width=1.5, dash="dot"),
            showlegend=False, opacity=0.50,
        ))


def force_trend_chart(adf: pd.DataFrame, mov: str) -> go.Figure:
    p = _pal()
    lbl = "Hip Abduction" if mov == "abd" else "Hip Adduction"
    fig = go.Figure()
    _add_series(fig, adf, f"hip_{mov}_left_n",  "Left",  p["left"])
    _add_series(fig, adf, f"hip_{mov}_right_n", "Right", p["right"])
    fig.update_layout(**_layout(f"{lbl} - Raw Force", "Force (N)"))
    return fig


def relative_force_chart(adf: pd.DataFrame, mov: str) -> go.Figure:
    p = _pal()
    lbl = "Hip Abduction" if mov == "abd" else "Hip Adduction"
    fig = go.Figure()
    _add_series(fig, adf, f"hip_{mov}_left_n_per_kg",  "Left",  p["left"])
    _add_series(fig, adf, f"hip_{mov}_right_n_per_kg", "Right", p["right"])
    fig.update_layout(**_layout(f"{lbl} - Relative Force", "N / kg BW"))
    return fig


def torque_bw_chart(adf: pd.DataFrame, mov: str) -> go.Figure:
    p = _pal()
    lbl = "Hip Abduction" if mov == "abd" else "Hip Adduction"

    col_l, unit = _best_torque_col(adf, mov, "left")
    col_r, _    = _best_torque_col(adf, mov, "right")
    if not col_l:
        return go.Figure()

    fig = go.Figure()
    _add_series(fig, adf, col_l, "Left",  p["left"])
    if col_r:
        _add_series(fig, adf, col_r, "Right", p["right"])
    fig.update_layout(**_layout(f"{lbl} — Torque / BW", f"{unit} BW"))
    return fig


def asymmetry_chart(adf: pd.DataFrame) -> go.Figure:
    p = _pal()
    fig = go.Figure()

    palette = {"abd": p["left"], "add": p["right"]}
    labels  = {"abd": "Abduction", "add": "Adduction"}

    for mov in ("abd", "add"):
        col = f"hip_{mov}_asym_pct"
        if col not in adf.columns:
            continue
        valid = adf[["date", col]].dropna()
        bar_colors = [
            FLAG_CLR if v > 15 else WARN_CLR if v > 10 else palette[mov]
            for v in valid[col]
        ]
        fig.add_trace(go.Bar(
            x=valid["date"], y=valid[col], name=labels[mov],
            marker_color=bar_colors, opacity=0.85,
        ))

    fig.add_hline(y=15, line_dash="dash", line_color=FLAG_CLR, line_width=1.5,
                  annotation_text="15% flag", annotation_font_color=FLAG_CLR,
                  annotation_position="top right")
    fig.add_hline(y=10, line_dash="dot", line_color=WARN_CLR, line_width=1,
                  annotation_text="10% warn", annotation_font_color=WARN_CLR,
                  annotation_position="top right")

    asym_cols = [c for c in ["hip_abd_asym_pct", "hip_add_asym_pct"] if c in adf.columns]
    y_max = max(22.0, float(adf[asym_cols].max().max()) * 1.25) if asym_cols else 22.0

    fig.update_layout(
        **_layout("Asymmetry Index Over Time", "Asymmetry (%)"),
        barmode="group",
        yaxis_range=[0, y_max],
    )
    return fig


def combined_torque_trend(adf: pd.DataFrame) -> go.Figure:
    """
    All 4 movement lines (ABD L/R + ADD L/R) on one chart.
    Prefers Nm/kg; falls back to N/kg when limb-length data absent.
    """
    p = _pal()
    series_def = [
        ("abd", "left",  p["left"],   "ABD Left"),
        ("abd", "right", p["right"],  "ABD Right"),
        ("add", "left",  WARN_CLR,    "ADD Left"),
        ("add", "right", OK_CLR,      "ADD Right"),
    ]
    fig = go.Figure()
    unit_label = "Nm/kg BW"
    has_data   = False

    for mov, side, color, label in series_def:
        col, unit = _best_torque_col(adf, mov, side)
        if col:
            _add_series(fig, adf, col, label, color)
            unit_label = f"{unit} BW"
            has_data   = True

    if not has_data:
        return go.Figure()

    fig.update_layout(**_layout("ABD + ADD — Torque Trend", unit_label))
    return fig


# ── Benchmark helpers ─────────────────────────────────────────────────────────
def _pct_rank(arr: np.ndarray, val: float) -> float:
    """Percentile rank (0–100). No scipy dependency."""
    n = len(arr)
    if n == 0:
        return 50.0
    return float(np.sum(arr < val) + 0.5 * np.sum(arr == val)) / n * 100.0


def _peer_avg_nm(df: pd.DataFrame, mov: str) -> tuple:
    """Return (col_l, col_r) using Nm/kg for the movement."""
    col_l = f"hip_{mov}_left_nm_per_kg"
    col_r = f"hip_{mov}_right_nm_per_kg"
    return col_l, col_r


def percentile_strip_chart(df: pd.DataFrame, athlete_id: str, mov: str) -> go.Figure:
    """
    Horizontal percentile strip using torque (Nm/kg BW, avg L+R).
    Fixed-position annotation avoids label crowding.
    """
    p = _pal()
    col_l, col_r = _peer_avg_nm(df, mov)
    lbl = "Abduction" if mov == "abd" else "Adduction"

    ath_df = df[df["athlete_id"] == athlete_id].sort_values("date")
    if ath_df.empty or col_l not in df.columns:
        return go.Figure()

    position = ath_df["position"].iloc[-1]
    peers    = df[df["position"] == position].groupby("athlete_id").last().reset_index()

    if col_l in peers.columns and col_r in peers.columns:
        peers["_avg"] = (peers[col_l] + peers[col_r]) / 2
    elif col_l in peers.columns:
        peers["_avg"] = peers[col_l]
    else:
        return go.Figure()

    peers = peers.dropna(subset=["_avg"])
    if len(peers) < 3:
        return go.Figure()

    try:
        if col_l in ath_df.columns and col_r in ath_df.columns:
            ath_val = float(
                (ath_df[col_l].dropna().iloc[-1] + ath_df[col_r].dropna().iloc[-1]) / 2
            )
        else:
            ath_val = float(ath_df[col_l].dropna().iloc[-1])
    except (IndexError, ValueError):
        return go.Figure()

    peer_vals = peers["_avg"].values
    peer_pcts = np.array([_pct_rank(peer_vals, v) for v in peer_vals])
    ath_pct   = _pct_rank(peer_vals, ath_val)

    fig = go.Figure()

    # Quartile zone shading
    for x0, x1, fill, label_txt in [
        (0,  25,  "rgba(239,68,68,0.18)",   "Bottom 25%"),
        (25, 50,  "rgba(234,179,8,0.14)",   "25–50%"),
        (50, 75,  "rgba(34,197,94,0.14)",   "50–75%"),
        (75, 100, "rgba(0,95,135,0.20)",    "Top 25%"),
    ]:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=-0.45, y1=0.45,
                      fillcolor=fill, line_width=0, layer="below")
        fig.add_annotation(
            x=(x0 + x1) / 2, y=0.52,
            text=label_txt,
            showarrow=False,
            font=dict(size=7.5, color=p["font"]),
            yanchor="bottom", xanchor="center",
        )

    # Peer dots
    fig.add_trace(go.Scatter(
        x=peer_pcts, y=[0] * len(peer_pcts),
        mode="markers",
        marker=dict(color=p["peer"], size=10, opacity=0.85,
                    line=dict(color=p["grid"], width=1.5)),
        name=f"{position} cohort",
        hovertemplate="Peer: %{x:.0f}th %ile<extra></extra>",
    ))

    # Athlete diamond
    fig.add_trace(go.Scatter(
        x=[ath_pct], y=[0],
        mode="markers",
        marker=dict(color=p["left"], size=18, symbol="diamond",
                    line=dict(color="white", width=2)),
        name="Selected athlete",
        hovertemplate=f"{ath_pct:.1f}th percentile<extra></extra>",
    ))

    # Athlete label
    fig.add_annotation(
        x=ath_pct, y=1.1,
        text=f"<b>{ath_pct:.0f}th %ile</b>",
        showarrow=True,
        arrowhead=2, arrowcolor=p["left"], arrowwidth=1.5,
        ax=0, ay=30,
        font=dict(size=12, color=p["left"], family="Arial"),
        bgcolor=p["bg"],
        bordercolor=p["left"],
        borderwidth=1,
        borderpad=4,
        yanchor="bottom",
        xanchor="center",
    )

    fig.add_vline(x=ath_pct, line_color=p["left"], line_width=1.2,
                  line_dash="dot", opacity=0.35)

    lay = _layout(f"Hip {lbl} — Percentile Rank vs {position}", "")
    lay.update({
        "yaxis": dict(visible=False, range=[-1.2, 2.0]),
        "xaxis": dict(
            title="Percentile Rank  (avg Nm/kg BW, L+R)",
            range=[-3, 103],
            showgrid=False,
            linecolor=p["ax_line"],
            showline=True,
            zeroline=False,
            tickvals=[0, 25, 50, 75, 100],
            tickfont=dict(color=p["font"]),
            title_font=dict(color=p["font"]),
        ),
        "height": 260,
        "showlegend": True,
        "legend": dict(
            orientation="h", y=-0.28, x=0.5, xanchor="center",
            font=dict(color=p["font"], size=10),
            bgcolor="rgba(0,0,0,0)",
        ),
    })
    fig.update_layout(**lay)
    return fig


def zscore_benchmark_chart(df: pd.DataFrame, athlete_id: str) -> go.Figure:
    """
    Horizontal z-score bars for Nm/kg metrics only vs positional cohort.
    """
    p = _pal()
    ath_df = df[df["athlete_id"] == athlete_id].sort_values("date")
    if ath_df.empty:
        return go.Figure()

    position = ath_df["position"].iloc[-1]
    ath      = ath_df.iloc[-1]
    peers    = df[df["position"] == position].groupby("athlete_id").last().reset_index()

    metric_defs = [
        ("ABD Left  (Nm/kg)",  "hip_abd_left_nm_per_kg"),
        ("ABD Right (Nm/kg)",  "hip_abd_right_nm_per_kg"),
        ("ADD Left  (Nm/kg)",  "hip_add_left_nm_per_kg"),
        ("ADD Right (Nm/kg)",  "hip_add_right_nm_per_kg"),
    ]

    labels, zs, colors, texts = [], [], [], []

    for label, col in metric_defs:
        if col not in peers.columns or col not in ath.index:
            continue
        peer_vals = peers[col].dropna().values
        if len(peer_vals) < 3:
            continue
        mean = float(peer_vals.mean())
        std  = float(peer_vals.std(ddof=1))
        if std < 1e-9:
            continue
        try:
            ath_val = float(ath[col])
            if np.isnan(ath_val):
                continue
        except (TypeError, ValueError):
            continue

        z = (ath_val - mean) / std
        labels.append(label)
        zs.append(round(z, 2))
        colors.append(p["left"] if z >= 0 else FLAG_CLR)
        texts.append(f"{z:+.2f} SD")

    if not labels:
        return go.Figure()

    max_abs = max(abs(z) for z in zs)
    x_range = max(2.5, max_abs + 0.9)

    fig = go.Figure()

    # ±1 SD shaded zone
    fig.add_vrect(
        x0=-1, x1=1,
        fillcolor="rgba(128,128,128,0.08)",
        opacity=1, line_width=0,
        annotation_text="Avg ±1 SD",
        annotation_position="top left",
        annotation_font=dict(size=8, color=p["font"]),
    )

    fig.add_vline(x=0, line_color=p["ax_line"], line_width=1.5)

    fig.add_trace(go.Bar(
        x=zs, y=labels,
        orientation="h",
        marker=dict(
            color=colors, opacity=0.85,
            line=dict(color=p["bg"], width=0.5),
        ),
        text=texts,
        textposition="outside",
        textfont=dict(size=10, color=p["font"]),
        cliponaxis=False,
        hovertemplate="%{y}: %{x:+.2f} SD<extra></extra>",
    ))

    lay = _layout(f"Z-Score vs {position} Cohort — Torque (Nm/kg)", "")
    lay.update({
        "yaxis": dict(
            autorange="reversed",
            gridcolor=p["grid"],
            linecolor=p["ax_line"],
            showline=True,
            tickfont=dict(color=p["font"]),
        ),
        "xaxis": dict(
            title="Standard Deviations from Cohort Mean",
            range=[-x_range, x_range],
            zeroline=False,
            gridcolor=p["grid"],
            linecolor=p["ax_line"],
            showline=True,
            tickfont=dict(color=p["font"]),
            title_font=dict(color=p["font"]),
        ),
        "showlegend": False,
        "height": max(300, len(labels) * 60 + 110),
        "bargap": 0.38,
        "margin": dict(t=52, b=60, l=160, r=100),
    })
    fig.update_layout(**lay)
    return fig


# ── Team / roster charts ──────────────────────────────────────────────────────
def team_torque_by_position(df: pd.DataFrame, mov: str) -> go.Figure:
    """Grouped bar: avg torque Left + Right by position tier. Prefers Nm/kg, falls back to N/kg."""
    p = _pal()
    lbl    = "Abduction" if mov == "abd" else "Adduction"
    latest = df.sort_values("date").groupby("athlete_id").last().reset_index()

    if "tier" not in latest.columns:
        return go.Figure()

    col_l, unit = _best_torque_col(latest, mov, "left")
    col_r, _    = _best_torque_col(latest, mov, "right")
    if not col_l:
        return go.Figure()

    tier_order = ["Skill", "Mid", "Big"]
    fig = go.Figure()

    for col, name, color in [(col_l, "Left", p["left"]), (col_r, "Right", p["right"])]:
        if col not in latest.columns:
            continue
        means, labels = [], []
        for t in tier_order:
            sub = latest[latest["tier"] == t][col].dropna()
            if len(sub) >= 1:
                means.append(float(sub.mean()))
                labels.append(f"{t}\nn={len(sub)}")
        if not means:
            continue
        fig.add_trace(go.Bar(
            x=labels, y=means, name=name,
            marker=dict(color=color, opacity=0.85,
                        line=dict(color=p["bg"], width=0.5)),
            text=[f"{v:.2f}" for v in means],
            textposition="outside",
            textfont=dict(size=10, color=p["font"]),
            cliponaxis=False,
        ))

    lay = _layout(f"Hip {lbl} — Avg Torque by Tier", f"{unit} BW")
    lay.update({"barmode": "group", "bargap": 0.28})
    fig.update_layout(**lay)
    return fig


def team_asymmetry_rank(df: pd.DataFrame, mov: str) -> go.Figure:
    """Horizontal bar: each athlete's asym %, sorted ascending, coloured by severity."""
    p = _pal()
    col_asym = f"hip_{mov}_asym_pct"
    lbl      = "Abduction" if mov == "abd" else "Adduction"

    latest = df.sort_values("date").groupby("athlete_id").last().reset_index()

    if col_asym not in latest.columns:
        return go.Figure()

    data = (
        latest[["athlete_name", col_asym, "position"]]
        .dropna(subset=[col_asym])
        .sort_values(col_asym, ascending=True)
        .reset_index(drop=True)
    )
    if data.empty:
        return go.Figure()

    bar_colors = [
        FLAG_CLR if v > 15 else WARN_CLR if v > 10 else p["left"]
        for v in data[col_asym]
    ]
    names = [n[:20] + "…" if len(n) > 20 else n for n in data["athlete_name"]]
    max_x = max(20.0, float(data[col_asym].max()) * 1.3)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=data[col_asym], y=names,
        orientation="h",
        marker=dict(color=bar_colors, opacity=0.85,
                    line=dict(color=p["bg"], width=0.5)),
        text=[f"{v:.1f}%" for v in data[col_asym]],
        textposition="outside",
        textfont=dict(size=9, color=p["font"]),
        cliponaxis=False,
        customdata=data["position"],
        hovertemplate="%{y} (%{customdata}): %{x:.1f}%<extra></extra>",
    ))

    fig.add_vline(x=15, line_dash="dash", line_color=FLAG_CLR, line_width=1.5,
                  annotation_text="15% flag", annotation_font_color=FLAG_CLR,
                  annotation_position="top right")
    fig.add_vline(x=10, line_dash="dot",  line_color=WARN_CLR, line_width=1,
                  annotation_text="10% warn", annotation_font_color=WARN_CLR,
                  annotation_position="top right")

    lay = _layout(f"Hip {lbl} — Asymmetry Ranking", "")
    lay.update({
        "xaxis": dict(
            title="Asymmetry Index (%)", range=[0, max_x],
            showgrid=True, gridcolor=p["grid"],
            linecolor=p["ax_line"], showline=True, zeroline=False,
            tickfont=dict(color=p["font"]), title_font=dict(color=p["font"]),
        ),
        "showlegend": False,
        "height": max(320, len(data) * 30 + 100),
        "bargap": 0.22,
        "margin": dict(t=52, b=60, l=130, r=80),
    })
    fig.update_layout(**lay)
    return fig


def team_torque_distribution(df: pd.DataFrame, mov: str) -> go.Figure:
    """Strip plot: individual torque values by tier, L+R overlaid."""
    p = _pal()
    lbl    = "Abduction" if mov == "abd" else "Adduction"
    latest = df.sort_values("date").groupby("athlete_id").last().reset_index()

    if "tier" not in latest.columns:
        return go.Figure()

    col_l, unit = _best_torque_col(latest, mov, "left")
    col_r, _    = _best_torque_col(latest, mov, "right")
    if not col_l:
        return go.Figure()

    fig = go.Figure()
    tier_order = ["Skill", "Mid", "Big"]

    for col, name, color in [(col_l, "Left", p["left"]), (col_r, "Right", p["right"])]:
        if col not in latest.columns:
            continue
        for t in tier_order:
            sub = latest[latest["tier"] == t][col].dropna()
            if sub.empty:
                continue
            fig.add_trace(go.Box(
                y=sub, x=[t] * len(sub),
                name=f"{name}",
                legendgroup=name,
                showlegend=(t == tier_order[0]),
                boxpoints="all", jitter=0.35, pointpos=0,
                marker=dict(color=color, size=6, opacity=0.75,
                            line=dict(color=p["bg"], width=1)),
                line=dict(color=color, width=1.5),
                fillcolor="rgba(0,0,0,0)",
            ))

    lay = _layout(f"Hip {lbl} — Torque Distribution by Tier", f"{unit} BW")
    lay.update({"boxmode": "group", "showlegend": True})
    fig.update_layout(**lay)
    return fig


def team_lr_scatter(df: pd.DataFrame, mov: str) -> go.Figure:
    """Scatter Left vs Right torque per athlete — symmetry bands overlaid."""
    p = _pal()
    lbl    = "Abduction" if mov == "abd" else "Adduction"
    latest = df.sort_values("date").groupby("athlete_id").last().reset_index()

    col_l, unit = _best_torque_col(latest, mov, "left")
    col_r, _    = _best_torque_col(latest, mov, "right")
    if not col_l or not col_r:
        return go.Figure()

    data = (
        latest[["athlete_name", "position", "tier", col_l, col_r]]
        .dropna(subset=[col_l, col_r])
    )
    if data.empty:
        return go.Figure()

    max_val = float(max(data[col_l].max(), data[col_r].max())) * 1.15
    xs = np.linspace(0, max_val, 100)
    tier_colors = {"Skill": p["left"], "Mid": WARN_CLR, "Big": p["right"]}

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(xs) + list(xs[::-1]),
        y=list(xs * 0.85) + list((xs / 0.85)[::-1]),
        fill="toself", fillcolor="rgba(239,68,68,0.08)",
        line_color="rgba(0,0,0,0)", showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=list(xs) + list(xs[::-1]),
        y=list(xs * 0.90) + list((xs / 0.90)[::-1]),
        fill="toself", fillcolor="rgba(255,180,0,0.07)",
        line_color="rgba(0,0,0,0)", showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines",
        line=dict(color=p["ax_line"], width=1.5, dash="dot"),
        name="Perfect symmetry", hoverinfo="skip",
    ))

    for tier in ["Skill", "Mid", "Big", "Other"]:
        sub = data[data["tier"] == tier]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub[col_l], y=sub[col_r],
            mode="markers",
            name=tier,
            marker=dict(color=tier_colors.get(tier, p["font"]), size=10, opacity=0.85,
                        line=dict(color=p["bg"], width=1.5)),
            customdata=sub[["athlete_name", "position"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                "Left: %{x:.2f} Nm/kg<br>Right: %{y:.2f} Nm/kg<extra></extra>"
            ),
        ))

    lay = _layout(f"Hip {lbl} — Left vs Right ({unit})", f"Right ({unit} BW)")
    lay.update({
        "xaxis": dict(
            title=f"Left ({unit} BW)", range=[0, max_val],
            showgrid=True, gridcolor=p["grid"],
            linecolor=p["ax_line"], showline=True, zeroline=False,
            tickfont=dict(color=p["font"]), title_font=dict(color=p["font"]),
        ),
        "yaxis": dict(
            title=f"Right ({unit} BW)", range=[0, max_val],
            showgrid=True, gridcolor=p["grid"],
            linecolor=p["ax_line"], showline=True, zeroline=False,
            tickfont=dict(color=p["font"]), title_font=dict(color=p["font"]),
        ),
        "showlegend": True,
        "height": 420,
    })
    fig.update_layout(**lay)
    return fig


def team_lr_scatter_combined(df: pd.DataFrame) -> go.Figure:
    """Single L vs R scatter with ABD and ADD overlaid."""
    p = _pal()
    latest = df.sort_values("date").groupby("athlete_id").last().reset_index()

    mov_def = [
        ("abd", "circle",  p["left"],  "ABD"),
        ("add", "diamond", WARN_CLR,   "ADD"),
    ]

    fig      = go.Figure()
    max_vals = []
    unit_label = "Nm/kg BW"

    for mov, symbol, color, label in mov_def:
        col_l, unit = _best_torque_col(latest, mov, "left")
        col_r, _    = _best_torque_col(latest, mov, "right")
        if not col_l or not col_r:
            continue
        unit_label = f"{unit} BW"

        data = (
            latest[["athlete_name", "position", "tier", col_l, col_r]]
            .dropna(subset=[col_l, col_r])
        )
        if data.empty:
            continue

        max_vals.append(float(max(data[col_l].max(), data[col_r].max())))

        fig.add_trace(go.Scatter(
            x=data[col_l], y=data[col_r],
            mode="markers",
            name=label,
            marker=dict(color=color, size=10, symbol=symbol, opacity=0.85,
                        line=dict(color=p["bg"], width=1.5)),
            customdata=data[["athlete_name", "position", "tier"]].values,
            hovertemplate=(
                f"<b>%{{customdata[0]}}</b> (%{{customdata[1]}} · %{{customdata[2]}})<br>"
                f"{label} Left: %{{x:.2f}} {unit}<br>"
                f"{label} Right: %{{y:.2f}} {unit}<extra></extra>"
            ),
        ))

    if not max_vals:
        return go.Figure()

    max_val = max(max_vals) * 1.15
    xs = np.linspace(0, max_val, 100)

    fig.add_trace(go.Scatter(
        x=list(xs) + list(xs[::-1]),
        y=list(xs * 0.85) + list((xs / 0.85)[::-1]),
        fill="toself", fillcolor="rgba(239,68,68,0.08)",
        line_color="rgba(0,0,0,0)", showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=list(xs) + list(xs[::-1]),
        y=list(xs * 0.90) + list((xs / 0.90)[::-1]),
        fill="toself", fillcolor="rgba(255,180,0,0.07)",
        line_color="rgba(0,0,0,0)", showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines",
        line=dict(color=p["ax_line"], width=1.5, dash="dot"),
        name="Perfect symmetry", hoverinfo="skip",
    ))

    lay = _layout("Bilateral Symmetry — ABD + ADD (Left vs Right)", f"Right ({unit_label})")
    lay.update({
        "xaxis": dict(
            title=f"Left ({unit_label})", range=[0, max_val],
            showgrid=True, gridcolor=p["grid"],
            linecolor=p["ax_line"], showline=True, zeroline=False,
            tickfont=dict(color=p["font"]), title_font=dict(color=p["font"]),
        ),
        "yaxis": dict(
            title=f"Right ({unit_label})", range=[0, max_val],
            showgrid=True, gridcolor=p["grid"],
            linecolor=p["ax_line"], showline=True, zeroline=False,
            tickfont=dict(color=p["font"]), title_font=dict(color=p["font"]),
        ),
        "showlegend": True,
        "height": 460,
    })
    fig.update_layout(**lay)
    return fig


def team_risk_matrix(df: pd.DataFrame) -> go.Figure:
    """
    Bubble chart: avg torque (x) vs max asymmetry % (y).
    """
    p = _pal()
    latest = df.sort_values("date").groupby("athlete_id").last().reset_index()

    nm_cols = []
    unit = "N/kg"
    for _mov in ("abd", "add"):
        for _side in ("left", "right"):
            _col, _unit = _best_torque_col(latest, _mov, _side)
            if _col:
                nm_cols.append(_col)
                unit = _unit
    asym_cols = [c for c in ["hip_abd_asym_pct", "hip_add_asym_pct"] if c in latest.columns]

    if not nm_cols or not asym_cols:
        return go.Figure()

    latest = latest.copy()
    latest["_avg_strength"] = latest[nm_cols].mean(axis=1)
    latest["_max_asym"]     = latest[asym_cols].max(axis=1)

    data = (
        latest[["athlete_name", "position", "tier", "_avg_strength",
                "_max_asym", "bodyweight_kg"]]
        .dropna(subset=["_avg_strength", "_max_asym"])
    )
    if data.empty:
        return go.Figure()

    x_min_v = float(data["_avg_strength"].min())
    x_max_v = float(data["_avg_strength"].max())
    x_pad   = max((x_max_v - x_min_v) * 0.15, 0.1)
    x_start = max(0.0, x_min_v - x_pad)
    x_end   = x_max_v + x_pad
    x_mid   = (x_start + x_end) / 2
    y_max   = max(25.0, float(data["_max_asym"].max()) * 1.2)
    y_split = 15.0

    tier_colors = {"Skill": p["left"], "Mid": WARN_CLR, "Big": p["right"]}

    fig = go.Figure()

    for x0, x1, y0, y1, fill in [
        (x_start, x_mid, y_split, y_max, "rgba(239,68,68,0.11)"),
        (x_mid,   x_end, y_split, y_max, "rgba(255,140,0,0.07)"),
        (x_start, x_mid, 0,       y_split,"rgba(255,180,0,0.05)"),
        (x_mid,   x_end, 0,       y_split,"rgba(34,197,94,0.05)"),
    ]:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                      fillcolor=fill, line_width=0, layer="below")

    for qx, qy, qtxt, qclr in [
        ((x_start + x_mid) / 2, y_max * 0.88, "HIGH PRIORITY", FLAG_CLR),
        ((x_mid + x_end)  / 2,  y_max * 0.88, "ASYMMETRIC",    WARN_CLR),
        ((x_start + x_mid) / 2, y_split * 0.18, "DEVELOPING",  p["font"]),
        ((x_mid + x_end)  / 2,  y_split * 0.18, "OPTIMAL",     OK_CLR),
    ]:
        fig.add_annotation(x=qx, y=qy, text=qtxt, showarrow=False,
                           font=dict(size=9, color=qclr, family="Arial"),
                           xanchor="center", opacity=0.75)

    fig.add_hline(y=y_split, line_dash="dash", line_color=FLAG_CLR, line_width=1.5,
                  annotation_text="15 % flag", annotation_font_color=FLAG_CLR,
                  annotation_position="top right")
    fig.add_hline(y=10, line_dash="dot", line_color=WARN_CLR, line_width=1,
                  annotation_text="10 % warn", annotation_font_color=WARN_CLR,
                  annotation_position="top right")
    fig.add_vline(x=x_mid, line_dash="dot", line_color=p["ax_line"], line_width=1.2)

    for tier in ["Skill", "Mid", "Big", "Other"]:
        sub = data[data["tier"] == tier].copy()
        if sub.empty:
            continue
        sizes = ((sub["bodyweight_kg"].fillna(85).clip(60, 145) - 60) / 85 * 14 + 8)
        fig.add_trace(go.Scatter(
            x=sub["_avg_strength"], y=sub["_max_asym"],
            mode="markers",
            name=tier,
            marker=dict(color=tier_colors.get(tier, p["font"]), size=sizes.tolist(),
                        opacity=0.85, line=dict(color=p["bg"], width=1.5)),
            customdata=sub[["athlete_name", "position", "bodyweight_kg"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                f"Avg Torque: %{{x:.2f}} {unit}<br>"
                "Max Asym: %{y:.1f}%<br>"
                "BW: %{customdata[2]:.1f} kg<extra></extra>"
            ),
        ))

    hp = data[(data["_avg_strength"] <= x_mid) & (data["_max_asym"] >= y_split)]
    for _, row in hp.iterrows():
        last = str(row["athlete_name"]).split()[-1]
        fig.add_annotation(
            x=float(row["_avg_strength"]), y=float(row["_max_asym"]),
            text=last, showarrow=False, yshift=12,
            font=dict(size=8, color=FLAG_CLR, family="Arial"),
            xanchor="center",
        )

    lay = _layout("Team Risk Matrix — Strength vs Asymmetry", "Max Asymmetry (%)")
    lay.update({
        "xaxis": dict(
            title=f"Avg Torque ({unit} BW)", range=[x_start, x_end],
            showgrid=True, gridcolor=p["grid"],
            linecolor=p["ax_line"], showline=True, zeroline=False,
            tickfont=dict(color=p["font"]), title_font=dict(color=p["font"]),
        ),
        "yaxis": dict(
            title="Max Asymmetry (%)", range=[0, y_max],
            showgrid=True, gridcolor=p["grid"],
            linecolor=p["ax_line"], showline=True, zeroline=False,
            tickfont=dict(color=p["font"]), title_font=dict(color=p["font"]),
        ),
        "showlegend": True,
        "height": 480,
    })
    fig.update_layout(**lay)
    return fig


def athlete_vs_team_chart(df: pd.DataFrame, athlete_id: str) -> go.Figure:
    """
    Dot-plot comparing one athlete to the full team (latest session per athlete).
    """
    p = _pal()
    latest = df.sort_values("date").groupby("athlete_id").last().reset_index()
    ath_rows = latest[latest["athlete_id"] == athlete_id]
    if ath_rows.empty:
        return go.Figure()
    ath_row  = ath_rows.iloc[0]
    ath_name = str(ath_row.get("athlete_name", "Selected"))

    metrics: list = []
    for mov in ("abd", "add"):
        for side in ("left", "right"):
            col, unit = _best_torque_col(latest, mov, side)
            if col:
                metrics.append((f"{mov.upper()} {side.title()} ({unit})", col, False))
    for mov, lbl in [("abd", "ABD Asym (%)"), ("add", "ADD Asym (%)")]:
        col = f"hip_{mov}_asym_pct"
        if col in latest.columns:
            metrics.append((lbl, col, True))

    if not metrics:
        return go.Figure()

    fig    = go.Figure()
    labels = []
    _seen  = set()

    for i, (label, col, lower_better) in enumerate(metrics):
        team_vals = latest[col].dropna()
        if len(team_vals) < 2:
            continue
        try:
            ath_val = float(ath_row[col])
            if np.isnan(ath_val):
                continue
        except (KeyError, TypeError, ValueError):
            continue

        t_min = float(team_vals.min())
        t_max = float(team_vals.max())
        t_q1  = float(team_vals.quantile(0.25))
        t_q3  = float(team_vals.quantile(0.75))
        t_med = float(team_vals.median())
        y     = i
        labels.append((y, label))

        fig.add_trace(go.Scatter(
            x=[t_min, t_max], y=[y, y],
            mode="lines", line=dict(color=p["range_line"], width=3),
            legendgroup="range",
            showlegend="range" not in _seen, name="Team range",
            hoverinfo="skip",
        ))
        _seen.add("range")

        fig.add_shape(type="rect",
            x0=t_q1, x1=t_q3, y0=y - 0.18, y1=y + 0.18,
            fillcolor=p["iqr_fill"], line_width=0,
        )

        fig.add_trace(go.Scatter(
            x=[t_med], y=[y],
            mode="markers",
            marker=dict(symbol="line-ns-open", color=p["font"], size=13,
                        line=dict(width=2.5, color=p["font"])),
            legendgroup="median",
            showlegend="median" not in _seen, name="Team median",
            hovertemplate=f"Median: {t_med:.2f}<extra></extra>",
        ))
        _seen.add("median")

        if lower_better:
            dot_color = FLAG_CLR if ath_val > 15 else WARN_CLR if ath_val > 10 else p["left"]
        else:
            dot_color = p["left"]

        fig.add_trace(go.Scatter(
            x=[ath_val], y=[y],
            mode="markers",
            marker=dict(color=dot_color, size=14, symbol="diamond",
                        line=dict(color="white", width=2)),
            legendgroup="athlete",
            showlegend="athlete" not in _seen, name=ath_name,
            hovertemplate=f"{label}: {ath_val:.2f}<extra></extra>",
        ))
        _seen.add("athlete")

    if not labels:
        return go.Figure()

    ys, ylabels = zip(*labels)
    lay = _layout(f"{ath_name} vs Full Team", "")
    lay.update({
        "yaxis": dict(
            tickmode="array", tickvals=list(ys), ticktext=list(ylabels),
            tickfont=dict(color=p["font"], size=10),
            gridcolor=p["grid"], linecolor=p["ax_line"], showline=True,
            range=[-0.6, len(labels) - 0.4],
        ),
        "xaxis": dict(
            title="Metric Value",
            showgrid=True, gridcolor=p["grid"],
            linecolor=p["ax_line"], showline=True, zeroline=False,
            tickfont=dict(color=p["font"]), title_font=dict(color=p["font"]),
        ),
        "showlegend": True,
        "height": max(300, len(labels) * 52 + 100),
        "margin": dict(t=52, b=60, l=160, r=60),
    })
    fig.update_layout(**lay)
    return fig


# ── Legacy (kept for compatibility, not called from app/PDF) ──────────────────
def position_benchmark_chart(df: pd.DataFrame, athlete_id: str, mov: str) -> go.Figure:
    p = _pal()
    col_l = f"hip_{mov}_left_n_per_kg"
    col_r = f"hip_{mov}_right_n_per_kg"

    ath_df = df[df["athlete_id"] == athlete_id].sort_values("date")
    if ath_df.empty or col_l not in df.columns:
        return go.Figure()
