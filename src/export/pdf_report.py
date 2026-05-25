import io
from datetime import datetime

import numpy as np
import pandas as pd
from fpdf import FPDF

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Brand palette (R, G, B) ───────────────────────────────────────────────────
ATHLETIC_BLUE = (  0,  95, 135)   # #005F87  hub · arms · headlines
PERF_TEAL     = (  0, 163, 163)   # #00A3A3  terminals · credentials · accents
NEAR_BLACK    = ( 20,  20,  20)   # #141414  cover panels
OFF_WHITE     = (245, 244, 242)   # #F5F4F2  backgrounds
WARM_SLATE    = ( 86,  90,  92)   # #565A5C  body text · supporting titles
BONE          = (241, 237, 230)   # #F1EDE6  editorial paper
WHITE         = (255, 255, 255)

# Clinical status colours (semantic, not brand identity)
RED_100   = (254, 226, 226)
RED_700   = (185,  28,  28)
AMBER_100 = (255, 243, 195)
AMBER_700 = (180, 110,   0)   # derived from #FFB400
GREEN_100 = (220, 252, 231)
GREEN_700 = ( 21, 128,  61)


def _fmt(val, spec=".1f", fallback="-"):
    try:
        return fallback if pd.isna(val) else f"{val:{spec}}"
    except Exception:
        return fallback


def _safe(text: str) -> str:
    """Strip/replace characters outside Latin-1 so Helvetica never errors."""
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _mpl_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── PDF class ─────────────────────────────────────────────────────────────────
class _Report(FPDF):
    def __init__(self, athlete_name: str, generated_at: str):
        super().__init__(orientation="P", unit="mm", format="A4")
        self._name = athlete_name
        self._ts   = generated_at
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(14, 14, 14)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*WARM_SLATE)
        self.cell(130, 6,
                  _safe(f"Hip Isometric Strength Report  |  {self._name}"),
                  align="L")
        self.cell(0, 6, f"Page {self.page_no()}", align="R")
        self.ln(2)
        self.set_draw_color(*PERF_TEAL)
        self.line(14, self.get_y(), 196, self.get_y())
        self.ln(4)
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-13)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*WARM_SLATE)
        self.cell(0, 5, f"Generated {self._ts}  |  Chase Chavez  PT - DPT - SCS",
                  align="C")
        self.set_text_color(0, 0, 0)


# ── Section helpers ───────────────────────────────────────────────────────────
def _section_title(pdf: FPDF, text: str):
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*ATHLETIC_BLUE)
    pdf.cell(0, 9, _safe(text), ln=True)
    pdf.set_draw_color(*PERF_TEAL)
    pdf.line(14, pdf.get_y(), 196, pdf.get_y())
    pdf.ln(5)
    pdf.set_text_color(0, 0, 0)


def _draw_node_mark(pdf: FPDF, x0: float, y0: float,
                    size_mm: float = 28.0, reversed: bool = True) -> None:
    """
    Draw the Chase Chavez Node Mark using FPDF2 primitives.

    reversed=True  → white arms + white hub  (for dark/blue covers)
    reversed=False → Athletic Blue arms + hub (for white/off-white contexts)
    """
    s = size_mm / 200.0  # scale: brand viewBox is 200×200 units

    hub_x,  hub_y  = x0 + 100 * s, y0 + 108 * s
    top_x,  top_y  = x0 + 100 * s, y0 +  44 * s
    bl_x,   bl_y   = x0 +  45 * s, y0 + 139 * s
    br_x,   br_y   = x0 + 155 * s, y0 + 139 * s

    arm_rgb  = WHITE        if reversed else ATHLETIC_BLUE
    hub_rgb  = WHITE        if reversed else ATHLETIC_BLUE
    term_rgb = PERF_TEAL    # always teal — non-optional per brand book

    arm_w   = max(0.35, 6.5 * s)
    hub_r   = 13 * s
    term_r  =  9 * s

    # Arms
    pdf.set_draw_color(*arm_rgb)
    pdf.set_line_width(arm_w)
    for tx, ty in [(top_x, top_y), (bl_x, bl_y), (br_x, br_y)]:
        pdf.line(hub_x, hub_y, tx, ty)

    # Terminal nodes (teal) — drawn AFTER arms so they sit on top
    pdf.set_fill_color(*term_rgb)
    pdf.set_draw_color(*term_rgb)
    for tx, ty in [(top_x, top_y), (bl_x, bl_y), (br_x, br_y)]:
        pdf.ellipse(tx - term_r, ty - term_r, 2 * term_r, 2 * term_r, style="F")

    # Hub
    pdf.set_fill_color(*hub_rgb)
    pdf.ellipse(hub_x - hub_r, hub_y - hub_r, 2 * hub_r, 2 * hub_r, style="F")

    # Reset
    pdf.set_line_width(0.2)
    pdf.set_draw_color(0, 0, 0)
    pdf.set_fill_color(*WHITE)


def _add_chart(pdf: FPDF, fig, caption: str = ""):
    img = _mpl_to_png(fig)
    pdf.image(io.BytesIO(img), x=14, w=182)
    if caption:
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(*WARM_SLATE)
        pdf.cell(0, 5, _safe(caption), ln=True, align="C")
        pdf.set_text_color(0, 0, 0)
    pdf.ln(3)


def _kpi_table(pdf: FPDF, mov: str, label: str, latest: pd.Series):
    """Render a metric summary table + asymmetry status bar for one movement."""
    l_nm  = latest.get(f"hip_{mov}_left_nm_per_kg", np.nan)
    r_nm  = latest.get(f"hip_{mov}_right_nm_per_kg",np.nan)
    asym  = latest.get(f"hip_{mov}_asym_pct",       np.nan)
    flag  = latest.get(f"hip_{mov}_asym_flag",      False)

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(*ATHLETIC_BLUE)
    pdf.cell(0, 7, label, ln=True)
    pdf.ln(1)

    rows = [
        ("Metric",         "Left",            "Right",           "Asymmetry"),
        ("Torque (Nm/kg)", _fmt(l_nm, ".2f"), _fmt(r_nm, ".2f"), f"{_fmt(asym)}%"),
    ]

    cw = [52, 42, 42, 42]

    # Header row — teal fill, white text
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(*PERF_TEAL)
    pdf.set_text_color(*WHITE)
    for i, h in enumerate(rows[0]):
        pdf.cell(cw[i], 7, h, border=1, fill=True, align="C")
    pdf.ln()

    # Data rows
    pdf.set_font("Helvetica", "", 9)
    pdf.set_fill_color(*BONE)
    pdf.set_text_color(0, 0, 0)
    for idx, row in enumerate(rows[1:]):
        fill = (idx % 2 == 0)
        for i, val in enumerate(row):
            pdf.cell(cw[i], 7, str(val), border=1, fill=fill, align="C")
        pdf.ln()
    pdf.set_fill_color(*WHITE)

    # Asymmetry status bar
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 8)
    if flag:
        pdf.set_fill_color(*RED_100)
        pdf.set_text_color(*RED_700)
        msg = (f"ASYMMETRY FLAG: {_fmt(asym)}% - "
               "exceeds 15% clinical threshold. Review recommended.")
    elif not pd.isna(asym) and float(asym) > 10:
        pdf.set_fill_color(*AMBER_100)
        pdf.set_text_color(*AMBER_700)
        msg = (f"MONITOR: {_fmt(asym)}% asymmetry - "
               "above 10% warning level. Track closely.")
    else:
        pdf.set_fill_color(*GREEN_100)
        pdf.set_text_color(*GREEN_700)
        msg = f"OK: Symmetry within normal range ({_fmt(asym)}%)."

    pdf.cell(0, 7, msg, border=0, fill=True, ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_fill_color(*WHITE)
    pdf.ln(5)


def _session_table(pdf: FPDF, adf: pd.DataFrame):
    """Full session history as a compact table with readable column headers."""
    _section_title(pdf, "Session History")

    cols = [c for c in (
        "date",
        "bodyweight_kg",
        "hip_abd_left_n", "hip_abd_right_n", "hip_abd_asym_pct",
        "hip_add_left_n", "hip_add_right_n", "hip_add_asym_pct",
    ) if c in adf.columns]

    headers = {
        "date":             "Date",
        "bodyweight_kg":    "BW (kg/lbs)",
        "hip_abd_left_n":   "ABD Left (N)",
        "hip_abd_right_n":  "ABD Right (N)",
        "hip_abd_asym_pct": "ABD Asym (%)",
        "hip_add_left_n":   "ADD Left (N)",
        "hip_add_right_n":  "ADD Right (N)",
        "hip_add_asym_pct": "ADD Asym (%)",
    }

    cw = 182.0 / len(cols)

    # Header row
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(*PERF_TEAL)
    pdf.set_text_color(*WHITE)
    for c in cols:
        pdf.cell(cw, 7, headers.get(c, c), border=1, fill=True, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(0, 0, 0)

    for idx, (_, row) in enumerate(
        adf.sort_values("date", ascending=False).iterrows()
    ):
        abd_asym = row.get("hip_abd_asym_pct", np.nan)
        add_asym = row.get("hip_add_asym_pct", np.nan)

        for c in cols:
            raw = row.get(c, "")
            if c == "date":
                val = pd.to_datetime(raw).strftime("%d %b %Y")
            elif c == "bodyweight_kg" and isinstance(raw, float) and not pd.isna(raw):
                val = f"{raw:.1f} / {raw * 2.20462:.1f}"
            elif isinstance(raw, float):
                val = _fmt(raw, ".1f")
            else:
                val = str(raw)

            flag_cell = (
                (c == "hip_abd_asym_pct"
                 and not pd.isna(abd_asym) and float(abd_asym) > 15)
                or
                (c == "hip_add_asym_pct"
                 and not pd.isna(add_asym) and float(add_asym) > 15)
            )

            # Alternate row shading
            row_fill = (idx % 2 == 0)

            if flag_cell:
                pdf.set_fill_color(*RED_100)
                pdf.set_text_color(*RED_700)
                pdf.cell(cw, 6, val, border=1, fill=True, align="C")
                pdf.set_fill_color(*WHITE)
                pdf.set_text_color(0, 0, 0)
            elif row_fill:
                pdf.set_fill_color(*BONE)
                pdf.cell(cw, 6, val, border=1, fill=True, align="C")
                pdf.set_fill_color(*WHITE)
            else:
                pdf.cell(cw, 6, val, border=1, align="C")
        pdf.ln()


# ── PDF-only matplotlib chart helpers ─────────────────────────────────────────
_LC = "#00A3A3"   # left  (teal)
_RC = "#005F87"   # right (blue)
_FC = "#EF4444"   # flag  (red)
_WC = "#FFB400"   # warn  (amber)
_GC = "#22C55E"   # ok    (green)


def _best_col(df: pd.DataFrame, mov: str, side: str):
    nm = f"hip_{mov}_{side}_nm_per_kg"
    n  = f"hip_{mov}_{side}_n_per_kg"
    if nm in df.columns and df[nm].notna().any():
        return nm, "Nm/kg"
    if n in df.columns and df[n].notna().any():
        return n, "N/kg"
    return None, None


def _latest_snap(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "athlete_id" not in df.columns:
        return df
    return df.sort_values("date").groupby("athlete_id", sort=False).last().reset_index()


def _pdf_torque_trend(adf: pd.DataFrame, mov: str):
    col_l, unit = _best_col(adf, mov, "left")
    col_r, _    = _best_col(adf, mov, "right")
    if col_l is None:
        return None
    fig, ax = plt.subplots(figsize=(9.6, 3.6))
    ax.set_facecolor("#F9F9F9")
    fig.patch.set_facecolor("white")
    data = adf.sort_values("date")
    v_l = data[["date", col_l]].dropna()
    if not v_l.empty:
        ax.plot(v_l["date"], v_l[col_l], color=_LC, marker="o",
                linewidth=2, markersize=5, label="Left")
    if col_r:
        v_r = data[["date", col_r]].dropna()
        if not v_r.empty:
            ax.plot(v_r["date"], v_r[col_r], color=_RC, marker="s",
                    linewidth=2, markersize=5, label="Right")
    lbl = "Abduction" if mov == "abd" else "Adduction"
    ax.set_title(f"Hip {lbl} — Torque / BW", fontsize=12, color="#005F87", pad=8)
    ax.set_xlabel("Date", fontsize=9)
    ax.set_ylabel(unit, fontsize=9)
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def _pdf_team_by_position(df: pd.DataFrame, mov: str):
    latest = _latest_snap(df)
    if "tier" not in latest.columns:
        return None
    col_l, unit = _best_col(latest, mov, "left")
    col_r, _    = _best_col(latest, mov, "right")
    if col_l is None:
        return None
    tier_order = ["Skill", "Mid", "Big"]
    ml, mr, labels = [], [], []
    for t in tier_order:
        sub = latest[latest["tier"] == t]
        if sub.empty:
            continue
        ml.append(float(sub[col_l].dropna().mean()) if sub[col_l].notna().any() else 0)
        mr.append(float(sub[col_r].dropna().mean()) if col_r and col_r in sub.columns and sub[col_r].notna().any() else 0)
        labels.append(f"{t}\nn={len(sub)}")
    if not labels:
        return None
    x  = np.arange(len(labels))
    w  = 0.35
    lbl = "Abduction" if mov == "abd" else "Adduction"
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.set_facecolor("#F9F9F9")
    fig.patch.set_facecolor("white")
    bl = ax.bar(x - w/2, ml, w, color=_LC, alpha=0.85, label="Left")
    br = ax.bar(x + w/2, mr, w, color=_RC, alpha=0.85, label="Right")
    for bar in list(bl) + list(br):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(unit, fontsize=9)
    ax.set_title(f"Hip {lbl} — Mean Torque by Tier", fontsize=12, color="#005F87", pad=8)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def _pdf_team_dist(df: pd.DataFrame, mov: str):
    latest = _latest_snap(df)
    if "tier" not in latest.columns:
        return None
    col_l, unit = _best_col(latest, mov, "left")
    col_r, _    = _best_col(latest, mov, "right")
    if col_l is None:
        return None
    tier_order = ["Skill", "Mid", "Big"]
    lbl = "Abduction" if mov == "abd" else "Adduction"
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=False)
    fig.patch.set_facecolor("white")
    for ax, col, side, color in [(axes[0], col_l, "Left", _LC),
                                  (axes[1], col_r, "Right", _RC)]:
        if col is None or col not in latest.columns:
            ax.set_visible(False)
            continue
        ax.set_facecolor("#F9F9F9")
        grps  = [latest[latest["tier"] == t][col].dropna().values for t in tier_order
                 if t in latest["tier"].values]
        glbls = [t for t in tier_order if t in latest["tier"].values]
        grps  = [(g, l) for g, l in zip(grps, glbls) if len(g) > 0]
        if not grps:
            ax.set_visible(False)
            continue
        gdata, glbls2 = zip(*grps)
        bp = ax.boxplot(list(gdata), patch_artist=True, widths=0.4)
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        for elem in ["whiskers", "caps", "medians"]:
            for line in bp[elem]:
                line.set_color(color)
        ax.set_xticklabels(glbls2, fontsize=9)
        ax.set_title(side, fontsize=10, color="#005F87")
        ax.set_ylabel(unit, fontsize=8)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Hip {lbl} — Distribution by Tier", fontsize=12, color="#005F87")
    fig.tight_layout()
    return fig


def _pdf_asym_rank(df: pd.DataFrame, mov: str,
                   flag_pct: float = 15.0, warn_pct: float = 10.0):
    latest = _latest_snap(df)
    col = f"hip_{mov}_asym_pct"
    if col not in latest.columns:
        return None
    data = (latest[["athlete_name", col]]
            .dropna(subset=[col])
            .sort_values(col, ascending=True)
            .reset_index(drop=True))
    if data.empty:
        return None
    colors = [_FC if v > flag_pct else _WC if v > warn_pct else _LC
              for v in data[col]]
    names  = [n[:22] + "…" if len(n) > 22 else n for n in data["athlete_name"]]
    h      = min(max(3.0, len(names) * 0.28), 8.5)
    lbl    = "Abduction" if mov == "abd" else "Adduction"
    fig, ax = plt.subplots(figsize=(9.6, h))
    ax.set_facecolor("#F9F9F9")
    fig.patch.set_facecolor("white")
    ax.barh(names, data[col], color=colors, alpha=0.85)
    ax.axvline(flag_pct, color=_FC, linestyle="--", linewidth=1.2,
               label=f"{flag_pct:.0f}% flag")
    ax.axvline(warn_pct, color=_WC, linestyle=":", linewidth=1.0,
               label=f"{warn_pct:.0f}% warn")
    ax.set_title(f"Hip {lbl} — Asymmetry Rankings", fontsize=12, color="#005F87", pad=8)
    ax.set_xlabel("Asymmetry %", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    fig.tight_layout()
    return fig


def _pdf_risk_matrix(df: pd.DataFrame, flag_pct: float = 15.0, warn_pct: float = 10.0):
    latest   = _latest_snap(df)
    nm_cols  = []
    unit     = "N/kg"
    for _mov in ("abd", "add"):
        for _side in ("left", "right"):
            col, u = _best_col(latest, _mov, _side)
            if col:
                nm_cols.append(col)
                unit = u
    asym_cols = [c for c in ["hip_abd_asym_pct", "hip_add_asym_pct"]
                 if c in latest.columns]
    if not nm_cols or not asym_cols:
        return None
    data = latest.copy()
    data["_avg"]  = data[nm_cols].mean(axis=1)
    data["_asym"] = data[asym_cols].max(axis=1)
    data = data.dropna(subset=["_avg", "_asym"])
    if data.empty:
        return None
    x_mid  = float(data["_avg"].median())
    y_top  = max(flag_pct * 1.7, float(data["_asym"].max()) * 1.2)
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    ax.set_facecolor("#F9F9F9")
    fig.patch.set_facecolor("white")
    ax.axhspan(flag_pct, y_top, xmin=0, xmax=0.5, color=_FC, alpha=0.07)
    ax.axhspan(flag_pct, y_top, xmin=0.5, xmax=1.0, color="#FF8C00", alpha=0.05)
    ax.axvline(x_mid,    color="#999", linestyle=":", linewidth=1, alpha=0.6)
    ax.axhline(flag_pct, color=_FC, linestyle="--", linewidth=1.2,
               label=f"{flag_pct:.0f}% flag")
    ax.axhline(warn_pct, color=_WC, linestyle=":", linewidth=1.0,
               label=f"{warn_pct:.0f}% warn")
    tier_clr = {"Skill": _LC, "Mid": _WC, "Big": _RC}
    for tier, grp in data.groupby("tier"):
        color  = tier_clr.get(str(tier), "#888888")
        bw_col = grp.get("bodyweight_kg") if "bodyweight_kg" in grp.columns else None
        sizes  = ((bw_col.fillna(85).clip(60, 145) - 60) / 85 * 80 + 30
                  if bw_col is not None else [50] * len(grp))
        ax.scatter(grp["_avg"], grp["_asym"], s=sizes, c=color, alpha=0.8,
                   label=str(tier), edgecolors="white", linewidths=0.5)
    ax.set_xlabel(f"Avg Torque ({unit} BW)", fontsize=9)
    ax.set_ylabel("Max Asymmetry (%)", fontsize=9)
    ax.set_title("Risk Matrix — Strength vs Asymmetry", fontsize=12, color="#005F87", pad=8)
    ax.legend(fontsize=9, title="Tier")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.2, linestyle="--")
    fig.tight_layout()
    return fig


# ── Public API ────────────────────────────────────────────────────────────────
def generate_athlete_pdf(
    athlete_name: str,
    adf: pd.DataFrame,
    df: pd.DataFrame,
    athlete_id: str,
    latest: pd.Series,
    prev,
) -> bytes:
    ts  = datetime.now().strftime("%d %b %Y  %H:%M")
    pdf = _Report(athlete_name, ts)
    pdf.add_page()

    # ── Page 1: Branded cover header ────────────────────────────────────────
    # Athletic Blue band
    pdf.set_fill_color(*ATHLETIC_BLUE)
    pdf.rect(0, 0, 210, 52, "F")
    _draw_node_mark(pdf, x0=164, y0=10, size_mm=32, reversed=True)

    # Athlete name — white
    pdf.set_xy(14, 12)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(*WHITE)
    pdf.cell(0, 11, _safe(athlete_name))

    # Credential line — teal
    pos      = str(latest.get("position", "-"))
    tier_val = str(latest.get("tier",     "-"))
    bw       = _fmt(latest.get("bodyweight_kg"), ".1f")

    info_parts = []
    jersey_raw = latest.get("jersey_number", latest.get("jersey", None))
    if jersey_raw is not None:
        try:
            info_parts.append(f"#{int(float(jersey_raw))}")
        except (TypeError, ValueError):
            pass
    if pos and pos != "-":
        info_parts.append(pos)
    if tier_val and tier_val != "-":
        info_parts.append(tier_val)
    if bw and bw != "-":
        try:
            bw_lbs = f"{float(latest.get('bodyweight_kg')) * 2.20462:.1f}"
            info_parts.append(f"{bw} kg / {bw_lbs} lbs")
        except Exception:
            info_parts.append(f"{bw} kg")

    pdf.set_xy(14, 27)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*PERF_TEAL)
    pdf.cell(0, 6, _safe("  |  ".join(info_parts)))

    # Bone info bar below header
    pdf.set_fill_color(*BONE)
    pdf.rect(0, 52, 210, 14, "F")

    date = pd.to_datetime(latest["date"]).strftime("%d %b %Y")
    ns   = len(adf)

    pdf.set_xy(14, 54)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*WARM_SLATE)
    pdf.cell(0, 6, _safe(f"Latest test: {date}   |   Total sessions: {ns}"), ln=True)

    # Reset to content area
    pdf.set_xy(14, 70)
    pdf.set_text_color(0, 0, 0)

    # KPI tables
    _kpi_table(pdf, "abd", "Hip Abduction - Latest Values", latest)
    _kpi_table(pdf, "add", "Hip Adduction - Latest Values", latest)

    # ── Page 2: Torque / BW trends (normalised only) ───────────────────────
    pdf.add_page()
    _section_title(pdf, "Torque / Bodyweight Trends")

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*ATHLETIC_BLUE)
    pdf.cell(0, 6, "Hip Abduction", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(1)

    fig_abd = _pdf_torque_trend(adf, "abd")
    if fig_abd is not None:
        _add_chart(pdf, fig_abd,
                   "ABD torque / BW (Nm/kg or N/kg).")
    else:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(*WARM_SLATE)
        pdf.cell(0, 7, "ABD trend unavailable - bodyweight required each session.", ln=True)
        pdf.set_text_color(0, 0, 0)

    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*ATHLETIC_BLUE)
    pdf.cell(0, 6, "Hip Adduction", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(1)

    fig_add = _pdf_torque_trend(adf, "add")
    if fig_add is not None:
        _add_chart(pdf, fig_add,
                   "ADD torque / BW (Nm/kg or N/kg).")
    else:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(*WARM_SLATE)
        pdf.cell(0, 7, "ADD trend unavailable - bodyweight required each session.", ln=True)
        pdf.set_text_color(0, 0, 0)

    return bytes(pdf.output())


# ── Team report ───────────────────────────────────────────────────────────────
class _TeamReport(FPDF):
    def __init__(self, generated_at: str):
        super().__init__(orientation="P", unit="mm", format="A4")
        self._ts = generated_at
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(14, 14, 14)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_xy(self.l_margin, 5)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*WARM_SLATE)
        self.cell(130, 6, "Team Hip Isometric Strength Report", align="L")
        self.cell(0, 6, f"Page {self.page_no()}", align="R")
        self.ln(2)
        self.set_draw_color(*PERF_TEAL)
        self.line(14, self.get_y(), 196, self.get_y())
        self.ln(4)
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-13)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*WARM_SLATE)
        self.cell(0, 5,
                  f"Generated {self._ts}  |  Chase Chavez  PT - DPT - SCS",
                  align="C")
        self.set_text_color(0, 0, 0)


def _normative_table(pdf: FPDF, df: pd.DataFrame):
    """Mean ± SD per tier for all available torque metrics."""
    _section_title(pdf, "Normative Data by Position Group")

    tiers = [t for t in ("Skill", "Mid", "Big")
             if "tier" in df.columns and t in df["tier"].values]
    if not tiers:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(*WARM_SLATE)
        pdf.cell(0, 7, "No tier data available.", ln=True)
        pdf.set_text_color(0, 0, 0)
        return

    # Latest snapshot per athlete
    snap = df.sort_values("date").groupby("athlete_id").last().reset_index()

    metric_cols = [c for c in (
        "hip_abd_left_nm_per_kg",  "hip_abd_right_nm_per_kg",
        "hip_add_left_nm_per_kg",  "hip_add_right_nm_per_kg",
        "hip_abd_asym_pct",        "hip_add_asym_pct",
    ) if c in snap.columns and snap[c].notna().any()]

    metric_labels = {
        "hip_abd_left_nm_per_kg":  "ABD Left (Nm/kg)",
        "hip_abd_right_nm_per_kg": "ABD Right (Nm/kg)",
        "hip_add_left_nm_per_kg":  "ADD Left (Nm/kg)",
        "hip_add_right_nm_per_kg": "ADD Right (Nm/kg)",
        "hip_abd_asym_pct":        "ABD Asym (%)",
        "hip_add_asym_pct":        "ADD Asym (%)",
    }

    if not metric_cols:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(*WARM_SLATE)
        pdf.cell(0, 7, "No torque metric columns available.", ln=True)
        pdf.set_text_color(0, 0, 0)
        return

    # Table header
    n_tiers = len(tiers)
    lw = 56
    cw = (182 - lw) / n_tiers

    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(*PERF_TEAL)
    pdf.set_text_color(*WHITE)
    pdf.cell(lw, 7, "Metric", border=1, fill=True, align="L")
    for t in tiers:
        n_t = len(snap[snap["tier"] == t])
        pdf.cell(cw, 7, f"{t} (n={n_t})", border=1, fill=True, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(0, 0, 0)
    for idx, col in enumerate(metric_cols):
        fill = (idx % 2 == 0)
        if fill:
            pdf.set_fill_color(*BONE)
        else:
            pdf.set_fill_color(*WHITE)
        pdf.cell(lw, 6, _safe(metric_labels.get(col, col)), border=1, fill=True, align="L")
        for t in tiers:
            grp = snap[snap["tier"] == t][col].dropna()
            if grp.empty:
                cell_val = "-"
            else:
                cell_val = f"{grp.mean():.2f} +/- {grp.std():.2f}"
            pdf.cell(cw, 6, cell_val, border=1, fill=fill, align="C")
        pdf.ln()
    pdf.ln(4)


def generate_team_pdf(df: pd.DataFrame) -> bytes:
    """Generate a team-level PDF report with positional normative data."""
    ts  = datetime.now().strftime("%d %b %Y  %H:%M")
    pdf = _TeamReport(ts)
    pdf.add_page()

    # ── Cover header ────────────────────────────────────────────────────────
    pdf.set_fill_color(*NEAR_BLACK)
    pdf.rect(0, 0, 210, 52, "F")
    _draw_node_mark(pdf, x0=164, y0=10, size_mm=32, reversed=True)

    pdf.set_xy(14, 12)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(*WHITE)
    pdf.cell(0, 11, "Team Hip Strength Report")

    snap = df.sort_values("date").groupby("athlete_id").last().reset_index()
    n_athletes = len(snap)
    date_range = ""
    if "date" in df.columns and not df["date"].isna().all():
        d_min = pd.to_datetime(df["date"]).min().strftime("%d %b %Y")
        d_max = pd.to_datetime(df["date"]).max().strftime("%d %b %Y")
        date_range = f"{d_min} - {d_max}"

    pdf.set_xy(14, 27)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*PERF_TEAL)
    pdf.cell(0, 6,
             _safe(f"{n_athletes} athletes  |  {date_range}"))

    pdf.set_fill_color(*BONE)
    pdf.rect(0, 52, 210, 14, "F")
    pdf.set_xy(14, 54)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*WARM_SLATE)
    now = pd.Timestamp.now()
    tested_14d = int(
        (snap["date"] >= now - pd.Timedelta(days=14)).sum()
        if "date" in snap.columns else 0
    )
    abd_flags = int(snap.get("hip_abd_asym_flag", pd.Series(dtype=bool)).sum())
    add_flags = int(snap.get("hip_add_asym_flag", pd.Series(dtype=bool)).sum())
    pdf.cell(0, 6,
             _safe(f"Tested last 14 days: {tested_14d}   |   "
                   f"ABD flags: {abd_flags}   |   ADD flags: {add_flags}"),
             ln=True)

    pdf.set_xy(14, 70)
    pdf.set_text_color(0, 0, 0)

    # Normative table
    _normative_table(pdf, df)

    # ── Torque by position group ─────────────────────────────────────────────
    pdf.add_page()
    _section_title(pdf, "Torque Output by Position Group")
    for mov, lbl in (("abd", "Hip Abduction"), ("add", "Hip Adduction")):
        fig = _pdf_team_by_position(df, mov)
        if fig is not None:
            _add_chart(pdf, fig, f"{lbl} - mean torque per tier")

    # ── Torque distribution ──────────────────────────────────────────────────
    pdf.add_page()
    _section_title(pdf, "Torque Distribution by Tier")
    for mov, lbl in (("abd", "Hip Abduction"), ("add", "Hip Adduction")):
        fig = _pdf_team_dist(df, mov)
        if fig is not None:
            _add_chart(pdf, fig, f"{lbl} - box plot per tier")

    # ── Asymmetry rankings ───────────────────────────────────────────────────
    pdf.add_page()
    _section_title(pdf, "Asymmetry Rankings")
    for mov, lbl in (("abd", "Hip Abduction"), ("add", "Hip Adduction")):
        fig = _pdf_asym_rank(df, mov)
        if fig is not None:
            _add_chart(pdf, fig,
                       f"{lbl} asymmetry index. Red = >15% flag. Amber = 10-15% watch.")

    # ── Risk matrix ──────────────────────────────────────────────────────────
    pdf.add_page()
    _section_title(pdf, "Risk Matrix - Strength vs Asymmetry")
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*WARM_SLATE)
    pdf.multi_cell(0, 5,
                   "X = avg torque across all movements. "
                   "Y = highest asymmetry (ABD or ADD). "
                   "Bubble size = body weight. "
                   "Top-left quadrant = highest intervention priority.")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)
    fig_risk = _pdf_risk_matrix(df)
    if fig_risk is not None:
        _add_chart(pdf, fig_risk)

    return bytes(pdf.output())
