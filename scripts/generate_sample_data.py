"""
Generate 50-player American football sample dataset.
Position-specific body weight and force norms.
Uses stdlib only — no external packages needed.
Usage: python scripts/generate_sample_data.py
"""
import csv
import math
import os
import random
from datetime import date, timedelta

random.seed(42)

# ── Position profiles ─────────────────────────────────────────────────────────
# bw_range: bodyweight kg
# abd_base: hip abduction baseline force range (N)
# add_ratio: adduction as fraction of abduction
# asym_bias: extra asymmetry probability (kickers/punters have dominant leg)
PROFILES = {
    "QB": dict(bw=(92,  102), abd=(205, 265), add_r=(0.68, 0.82), asym_bias=0.15),
    "WR": dict(bw=(82,   92), abd=(190, 255), add_r=(0.66, 0.80), asym_bias=0.15),
    "TE": dict(bw=(108, 122), abd=(235, 295), add_r=(0.70, 0.83), asym_bias=0.10),
    "RB": dict(bw=(95,  108), abd=(245, 310), add_r=(0.72, 0.85), asym_bias=0.20),
    "FB": dict(bw=(108, 120), abd=(255, 320), add_r=(0.73, 0.86), asym_bias=0.10),
    "LB": dict(bw=(108, 120), abd=(250, 315), add_r=(0.71, 0.84), asym_bias=0.15),
    "DB": dict(bw=(82,   95), abd=(195, 258), add_r=(0.67, 0.81), asym_bias=0.20),
    "K":  dict(bw=(82,   95), abd=(188, 248), add_r=(0.65, 0.78), asym_bias=0.55),
    "P":  dict(bw=(86,   98), abd=(192, 252), add_r=(0.66, 0.79), asym_bias=0.55),
    "LS": dict(bw=(102, 115), abd=(235, 290), add_r=(0.70, 0.83), asym_bias=0.10),
    "OL": dict(bw=(130, 155), abd=(295, 385), add_r=(0.72, 0.84), asym_bias=0.10),
    "DL": dict(bw=(118, 148), abd=(278, 365), add_r=(0.71, 0.83), asym_bias=0.12),
}

# ── Roster: 50 players ────────────────────────────────────────────────────────
FIRST = ["Marcus","Darius","Tyrone","Jordan","Devon","Malik","Trevor","Cody",
         "Brandon","Xavier","Isaiah","Elijah","Caleb","Nathan","Ethan","Ryan",
         "Tyler","Kyle","Chase","Brett","Jamal","Deon","Carlos","Derek","Mason",
         "Logan","Zach","Hunter","Austin","Blake","Damien","Jalen","Cam","Tre",
         "Keion","Theo","Quincy","Andre","Miles","Donte","Rashad","Trevon",
         "Javon","Darius","Royce","Gunnar","Cade","Peyton","Brock","Tanner"]

LAST = ["Williams","Johnson","Brown","Davis","Miller","Wilson","Moore","Taylor",
        "Anderson","Thomas","Jackson","White","Harris","Martin","Thompson","Garcia",
        "Martinez","Robinson","Clark","Rodriguez","Lewis","Lee","Walker","Hall",
        "Allen","Young","Hernandez","King","Wright","Lopez","Hill","Scott","Green",
        "Adams","Baker","Gonzalez","Nelson","Carter","Mitchell","Perez","Roberts",
        "Turner","Phillips","Campbell","Parker","Evans","Edwards","Collins","Stewart","Morris"]

POSITIONS_ROSTER = (
    ["QB"] * 5 +
    ["WR"] * 7 +
    ["TE"] * 4 +
    ["RB"] * 3 +
    ["FB"] * 2 +
    ["LB"] * 5 +
    ["DB"] * 6 +
    ["K"]  * 2 +
    ["P"]  * 1 +
    ["LS"] * 1 +
    ["OL"] * 8 +
    ["DL"] * 6
)  # = 50

BASE_DATE  = date(2024, 9, 3)   # week 1 of season
N_SESSIONS = 7                  # every 2 weeks → ~14 weeks of monitoring

FIELDNAMES = [
    "date", "athlete_id", "athlete_name", "position", "bodyweight_kg",
    "hip_abd_left_n",  "hip_abd_right_n",  "hip_abd_left_nm",  "hip_abd_right_nm",
    "hip_add_left_n",  "hip_add_right_n",  "hip_add_left_nm",  "hip_add_right_nm",
]


# ── Math helpers ──────────────────────────────────────────────────────────────
def _gauss(sd: float) -> float:
    u1 = max(random.random(), 1e-12)
    u2 = random.random()
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2) * sd


def _wc(values, weights):
    r, c = random.random(), 0.0
    for v, w in zip(values, weights):
        c += w
        if r < c:
            return v
    return values[-1]


# ── Generate ──────────────────────────────────────────────────────────────────
random.shuffle(POSITIONS_ROSTER)
used_names = set()
rows = []

for idx, pos in enumerate(POSITIONS_ROSTER):
    p = PROFILES[pos]
    aid = f"P{idx+1:03d}"

    # Unique name
    while True:
        name = f"{random.choice(FIRST)} {random.choice(LAST)}"
        if name not in used_names:
            used_names.add(name)
            break

    bw         = random.uniform(*p["bw"])
    base_abd_l = random.uniform(*p["abd"])

    # Asymmetry: kickers/punters get a strong dominant-side bias
    if random.random() < p["asym_bias"]:
        # Dominant leg — right side 15-30% stronger for abd (kicking motion)
        base_abd_r = base_abd_l * random.uniform(0.70, 0.84)
    else:
        base_abd_r = base_abd_l * random.uniform(0.88, 1.08)

    add_ratio  = random.uniform(*p["add_r"])
    base_add_l = base_abd_l * add_ratio
    base_add_r = base_add_l * random.uniform(0.88, 1.10)

    trend = _wc([0.020, 0.0, -0.015], [0.45, 0.35, 0.20])

    for i in range(N_SESSIONS):
        d = BASE_DATE + timedelta(days=14 * i)
        f = 1.0 + trend * i

        abd_l = max(80.0, base_abd_l * f + _gauss(10))
        abd_r = max(80.0, base_abd_r * f + _gauss(10))
        add_l = max(60.0, base_add_l * f + _gauss(8))
        add_r = max(60.0, base_add_r * f + _gauss(8))

        rows.append({
            "date":             d.isoformat(),
            "athlete_id":       aid,
            "athlete_name":     name,
            "position":         pos,
            "bodyweight_kg":    round(bw + _gauss(0.4), 1),
            "hip_abd_left_n":   round(abd_l, 1),
            "hip_abd_right_n":  round(abd_r, 1),
            "hip_abd_left_nm":  round(abd_l * 0.35, 1),
            "hip_abd_right_nm": round(abd_r * 0.35, 1),
            "hip_add_left_n":   round(add_l, 1),
            "hip_add_right_n":  round(add_r, 1),
            "hip_add_left_nm":  round(add_l * 0.33, 1),
            "hip_add_right_nm": round(add_r * 0.33, 1),
        })

os.makedirs(os.path.join("data", "raw"), exist_ok=True)
out = os.path.join("data", "raw", "sample_data.csv")

with open(out, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)

n_athletes = len(POSITIONS_ROSTER)
pos_counts = {}
for p in POSITIONS_ROSTER:
    pos_counts[p] = pos_counts.get(p, 0) + 1

print(f"Generated {len(rows)} rows  ({n_athletes} athletes x {N_SESSIONS} sessions)")
print(f"Saved -> {out}")
print()
print("Position breakdown:")
for pos, cnt in sorted(pos_counts.items()):
    tier = {"QB":"Skill","WR":"Skill","TE":"Skill","DB":"Skill","K":"Skill",
            "P":"Skill","LS":"Skill","RB":"Mid","FB":"Mid","LB":"Mid",
            "OL":"Big","DL":"Big"}.get(pos,"?")
    print(f"  {pos:4s} ({tier:5s}): {cnt}")
