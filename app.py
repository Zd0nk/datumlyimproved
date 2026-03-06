"""
FPL Optimizer v2 — Proper Architecture
=======================================
Data sources:
  1. FPL API (bootstrap-static, fixtures) — player stats, prices, xG/xA, form
  2. football-data.co.uk — betting odds → match probabilities
  3. Custom xPts model — blends FPL xG/xA + odds-derived team strength + form

Optimisation:
  - PuLP MILP solver for squad selection (not greedy)
  - Proper constraints: budget, max 3/team, formation, 15-man squad

UI: Streamlit with 5 tabs
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
from pulp import (
    LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, value,
    PULP_CBC_CMD,
)
from datetime import datetime
from io import StringIO
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="FPL Optimizer",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

FPL_BASE = "https://fantasy.premierleague.com/api"
FOOTBALL_DATA_URL = "https://www.football-data.co.uk/mmz4281/2526/E0.csv"

POS_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
POS_FULL = {1: "Goalkeeper", 2: "Defender", 3: "Midfielder", 4: "Forward"}

# FPL points system constants
PTS_GOAL = {1: 10, 2: 6, 3: 5, 4: 4}
PTS_ASSIST = 3
PTS_CS = {1: 4, 2: 4, 3: 1, 4: 0}
PTS_APPEARANCE = 2  # 60+ mins
PTS_BONUS_AVG = 0.5  # average bonus per appearance

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .stApp { background-color: #0a0e17; }
    header[data-testid="stHeader"] { background-color: rgba(10,14,23,0.9); backdrop-filter: blur(10px); }

    .main-title {
        background: linear-gradient(135deg, #38bdf8, #818cf8, #a78bfa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 2.2rem; font-weight: 800; letter-spacing: -1px; margin-bottom: 0;
    }
    .sub-title { color: #8892a8; font-size: 0.85rem; margin-top: -8px; }

    .metric-card {
        background: #111827; border: 1px solid #2a3550;
        border-radius: 14px; padding: 1.1rem; text-align: center;
    }
    .metric-label { color: #5a6580; font-size: 0.68rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 3px; }
    .metric-value { color: #e2e8f0; font-size: 1.6rem; font-weight: 700; }
    .metric-sub { color: #8892a8; font-size: 0.75rem; margin-top: 2px; }

    .fdr-1 { background:#065f46; color:#6ee7b7; padding:2px 7px; border-radius:5px; font-size:0.72rem; font-weight:600; display:inline-block; margin:1px; }
    .fdr-2 { background:#14532d; color:#86efac; padding:2px 7px; border-radius:5px; font-size:0.72rem; font-weight:600; display:inline-block; margin:1px; }
    .fdr-3 { background:#78350f; color:#fcd34d; padding:2px 7px; border-radius:5px; font-size:0.72rem; font-weight:600; display:inline-block; margin:1px; }
    .fdr-4 { background:#7c2d12; color:#fdba74; padding:2px 7px; border-radius:5px; font-size:0.72rem; font-weight:600; display:inline-block; margin:1px; }
    .fdr-5 { background:#7f1d1d; color:#fca5a5; padding:2px 7px; border-radius:5px; font-size:0.72rem; font-weight:600; display:inline-block; margin:1px; }

    .transfer-card {
        background: #1a2236; border: 1px solid #2a3550;
        border-radius: 10px; padding: 0.75rem 1rem; margin-bottom: 0.4rem;
    }
    .transfer-out { color: #f87171; font-weight: 600; }
    .transfer-in { color: #34d399; font-weight: 600; }
    .transfer-arrow { color: #38bdf8; font-size: 1.1rem; }

    .gw-bar {
        background: #111827; border: 1px solid #2a3550; border-radius: 12px;
        padding: 0.65rem 1.1rem; display: flex; align-items: center; gap: 1rem;
        margin-bottom: 1rem; flex-wrap: wrap;
    }
    .gw-num { background: linear-gradient(135deg,#38bdf8,#818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 1.05rem; font-weight: 700; }
    .gw-deadline { color: #8892a8; font-size: 0.78rem; }

    .badge { font-size:0.65rem; padding:3px 9px; border-radius:6px; font-weight:600; }
    .badge-green { background:rgba(52,211,153,0.15); color:#34d399; }
    .badge-yellow { background:rgba(251,191,36,0.15); color:#fbbf24; }
    .badge-blue { background:rgba(56,189,248,0.15); color:#38bdf8; }

    .pitch-row-label { color:#5a6580; font-size:0.68rem; text-transform:uppercase; letter-spacing:1px; margin-bottom:5px; margin-top:12px; }
    .pitch-shirt { width:40px; height:40px; border-radius:10px; display:inline-flex; align-items:center; justify-content:center; font-weight:700; font-size:0.75rem; color:white; margin:0 auto; }
    .pitch-shirt-gkp { background:#f59e0b; }
    .pitch-shirt-def { background:#3b82f6; }
    .pitch-shirt-mid { background:#10b981; }
    .pitch-shirt-fwd { background:#ef4444; }
    .pitch-name { font-size:0.7rem; font-weight:600; color:#e2e8f0; margin-top:3px; }
    .pitch-price { font-size:0.58rem; color:#5a6580; }

    .section-header { font-size:1rem; font-weight:700; margin-bottom:0.5rem; margin-top:1rem; }

    .source-tag {
        display:inline-block; font-size:0.6rem; padding:2px 6px; border-radius:4px;
        font-weight:600; margin-left:6px;
    }
    .src-fpl { background:rgba(56,189,248,0.15); color:#38bdf8; }
    .src-odds { background:rgba(251,191,36,0.15); color:#fbbf24; }
    .src-model { background:rgba(167,139,250,0.15); color:#a78bfa; }

    #MainMenu {visibility:hidden;}
    footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA LAYER
# ============================================================

@st.cache_data(ttl=3600)
def load_fpl_data():
    """Source 1: FPL API — player stats, prices, fixtures, xG/xA."""
    try:
        headers = {"User-Agent": "FPL-Optimizer/2.0"}
        b = requests.get(f"{FPL_BASE}/bootstrap-static/", headers=headers, timeout=30).json()
        f = requests.get(f"{FPL_BASE}/fixtures/", headers=headers, timeout=30).json()
        return b, f, None
    except Exception as e:
        return None, None, str(e)


@st.cache_data(ttl=7200)
def load_betting_odds():
    """Source 2: football-data.co.uk — match betting odds for PL 2025-26."""
    try:
        resp = requests.get(FOOTBALL_DATA_URL, timeout=20)
        if resp.status_code != 200:
            return None, "HTTP " + str(resp.status_code)
        df = pd.read_csv(StringIO(resp.text), on_bad_lines="skip")
        if len(df) == 0:
            return None, "Empty CSV"
        return df, None
    except Exception as e:
        return None, str(e)


def odds_to_probabilities(odds_df, teams_map):
    """
    Convert betting odds to match probabilities per team.
    Returns dict: team_name -> {attack_strength, defence_strength, cs_prob, goal_expectation}
    """
    if odds_df is None or len(odds_df) == 0:
        return {}

    # Standardise column names
    cols = odds_df.columns.tolist()
    # We need: HomeTeam, AwayTeam, B365H, B365D, B365A, FTHG, FTAG
    required = ["HomeTeam", "AwayTeam"]
    if not all(c in cols for c in required):
        return {}

    # Use B365 odds (most common), fallback to average odds
    h_col = "B365H" if "B365H" in cols else ("AvgH" if "AvgH" in cols else None)
    d_col = "B365D" if "B365D" in cols else ("AvgD" if "AvgD" in cols else None)
    a_col = "B365A" if "B365A" in cols else ("AvgA" if "AvgA" in cols else None)

    if not all([h_col, d_col, a_col]):
        return {}

    team_stats = {}

    # Process each team's matches
    all_teams = set(odds_df["HomeTeam"].unique()) | set(odds_df["AwayTeam"].unique())

    for team in all_teams:
        home_matches = odds_df[odds_df["HomeTeam"] == team].copy()
        away_matches = odds_df[odds_df["AwayTeam"] == team].copy()

        # Calculate implied probabilities from odds (with overround removal)
        win_probs, cs_probs, goals_for, goals_against = [], [], [], []

        for _, m in home_matches.iterrows():
            try:
                h, d, a = float(m[h_col]), float(m[d_col]), float(m[a_col])
                overround = (1/h + 1/d + 1/a)
                win_probs.append((1/h) / overround)
                # Clean sheet proxy: P(win)*0.35 + P(draw)*0.55 (approx CS given result)
                p_win = (1/h) / overround
                p_draw = (1/d) / overround
                cs_probs.append(p_win * 0.35 + p_draw * 0.55)
            except (ValueError, ZeroDivisionError):
                continue
            # Actual goals if available
            if "FTHG" in m and pd.notna(m.get("FTHG")):
                try:
                    goals_for.append(float(m["FTHG"]))
                    goals_against.append(float(m["FTAG"]))
                except (ValueError, TypeError):
                    pass

        for _, m in away_matches.iterrows():
            try:
                h, d, a = float(m[h_col]), float(m[d_col]), float(m[a_col])
                overround = (1/h + 1/d + 1/a)
                win_probs.append((1/a) / overround)
                p_win = (1/a) / overround
                p_draw = (1/d) / overround
                cs_probs.append(p_win * 0.30 + p_draw * 0.55)
            except (ValueError, ZeroDivisionError):
                continue
            if "FTAG" in m and pd.notna(m.get("FTAG")):
                try:
                    goals_for.append(float(m["FTAG"]))
                    goals_against.append(float(m["FTHG"]))
                except (ValueError, TypeError):
                    pass

        if win_probs:
            avg_gf = np.mean(goals_for) if goals_for else 1.3
            avg_ga = np.mean(goals_against) if goals_against else 1.3
            team_stats[team] = {
                "win_prob": np.mean(win_probs),
                "cs_prob": np.mean(cs_probs),
                "avg_goals_for": avg_gf,
                "avg_goals_against": avg_ga,
                "attack_strength": avg_gf / 1.3,  # relative to league average
                "defence_strength": avg_ga / 1.3,
            }

    return team_stats


# Team name mapping: football-data names → FPL short names
TEAM_NAME_MAP = {
    "Arsenal": "ARS", "Aston Villa": "AVL", "Bournemouth": "BOU",
    "Brentford": "BRE", "Brighton": "BHA", "Chelsea": "CHE",
    "Crystal Palace": "CRY", "Everton": "EVE", "Fulham": "FUL",
    "Ipswich": "IPS", "Leicester": "LEI", "Liverpool": "LIV",
    "Man City": "MCI", "Man United": "MUN", "Newcastle": "NEW",
    "Nott'm Forest": "NFO", "Southampton": "SOU", "Spurs": "TOT",
    "West Ham": "WHU", "Wolves": "WOL",
    # 2025-26 promoted teams (adjust as needed)
    "Leeds": "LEE", "Burnley": "BUR", "Sunderland": "SUN",
    "Sheffield Utd": "SHU", "Norwich": "NOR", "Middlesbrough": "MID",
    "Luton": "LUT",
}


def build_xpts_model(players_df, team_odds, teams_map, fixtures, current_gw_id):
    """
    Source 3: Custom expected points model.

    For each player, for the next N gameweeks, estimate xPts by combining:
      - FPL API xG/xA per 90
      - Betting odds (team attack/defence strength, CS probability)
      - Playing time probability
      - FPL scoring system

    Returns: dict player_id -> {gw: xpts, ...} for next 6 GWs
    """
    # Build FPL team short_name -> odds stats mapping
    odds_by_fpl = {}
    for odds_name, fpl_short in TEAM_NAME_MAP.items():
        if odds_name in team_odds:
            odds_by_fpl[fpl_short] = team_odds[odds_name]

    # Build opponent map per team per GW
    upcoming = {}  # team_id -> [{gw, opp_id, home, difficulty}]
    for t_id in teams_map:
        upcoming[t_id] = []
    for f in fixtures:
        ev = f.get("event")
        if ev and current_gw_id <= ev < current_gw_id + 6:
            if f["team_h"] in upcoming:
                upcoming[f["team_h"]].append({
                    "gw": ev, "opp_id": f["team_a"], "home": True,
                    "difficulty": f.get("team_h_difficulty", 3)
                })
            if f["team_a"] in upcoming:
                upcoming[f["team_a"]].append({
                    "gw": ev, "opp_id": f["team_h"], "home": False,
                    "difficulty": f.get("team_a_difficulty", 3)
                })

    # League average goals per game (approx)
    league_avg_goals = 1.35

    xpts_all = {}
    for _, p in players_df.iterrows():
        pid = p["id"]
        pos = p["pos_id"]
        team_short = p["team"]
        mins = p["minutes"]
        starts = max(p.get("starts", 0), 0)
        total_gws_played = max(current_gw_id - 1, 1)  # GWs elapsed so far

        # ============================================================
        # EXPECTED MINUTES MODEL
        # ============================================================
        # Key insight: a player's expected minutes next GW should be based on
        # their actual average minutes per GW this season, not just a binary
        # "available or not". This prevents fringe/youth players being inflated.

        # Average minutes per GW this season
        avg_mins_per_gw = mins / total_gws_played

        # FPL's chance_of_playing (None = no news = likely available)
        chance = p.get("chance_playing", None)
        if chance is not None and not pd.isna(chance):
            availability = float(chance) / 100.0
        else:
            # No news — availability based on recent playing pattern
            if avg_mins_per_gw >= 60:
                availability = 0.95  # regular starter
            elif avg_mins_per_gw >= 30:
                availability = 0.75  # rotation / sub risk
            elif avg_mins_per_gw >= 10:
                availability = 0.40  # mainly a sub
            elif mins > 0:
                availability = 0.15  # fringe player
            else:
                availability = 0.0   # hasn't played

        # Expected minutes next GW (capped at 90)
        # Blend: recent avg mins * availability
        expected_mins = min(avg_mins_per_gw * availability, 90)

        # Convert to "expected 90s" for scaling xG/xA
        expected_90s = expected_mins / 90.0

        # Playing probability (for appearance points — did they get on the pitch?)
        play_prob = min(availability, 0.98)

        # Full 60+ min probability (for clean sheet, appearance pts)
        full_game_prob = expected_mins / 90.0 if expected_mins >= 45 else expected_mins / 180.0

        # ============================================================
        # PER-90 STATS (from FPL API xG/xA data)
        # ============================================================
        mins_played = max(mins, 1)
        nineties = mins_played / 90.0
        xg_per90 = float(p.get("xg_per90", 0) or 0)
        xa_per90 = float(p.get("xa_per90", 0) or 0)

        # Fallback: use actual goals/assists per 90 ONLY if enough minutes
        # to be a reliable sample (>= 270 mins = ~3 full games)
        if xg_per90 == 0 and p["goals"] > 0 and mins >= 270:
            xg_per90 = p["goals"] / nineties
        if xa_per90 == 0 and p["assists"] > 0 and mins >= 270:
            xa_per90 = p["assists"] / nineties

        # Apply regression to the mean for low-sample players
        # (fewer minutes = regress more towards position average)
        sample_weight = min(nineties / 10.0, 1.0)  # full weight at 10+ 90s
        pos_avg_xg = {1: 0.0, 2: 0.02, 3: 0.12, 4: 0.35}
        pos_avg_xa = {1: 0.01, 2: 0.05, 3: 0.10, 4: 0.12}
        xg_per90 = xg_per90 * sample_weight + pos_avg_xg.get(pos, 0.1) * (1 - sample_weight)
        xa_per90 = xa_per90 * sample_weight + pos_avg_xa.get(pos, 0.08) * (1 - sample_weight)

        player_gw_xpts = {}
        fix_list = upcoming.get(p["team_id"], [])

        for fix in fix_list:
            gw = fix["gw"]
            opp_team = teams_map.get(fix["opp_id"], {})
            opp_short = opp_team.get("short_name", "???")

            # Get opponent defensive strength from odds
            opp_odds = odds_by_fpl.get(opp_short, {})
            team_attack_odds = odds_by_fpl.get(team_short, {})

            # Adjust xG based on opponent defence strength
            opp_def_str = opp_odds.get("defence_strength", 1.0)
            team_atk_str = team_attack_odds.get("attack_strength", 1.0)

            # Scale factor: easier opponent = higher xG
            # opp_def_str > 1 means they concede more → easier
            scale = (opp_def_str * 0.5 + team_atk_str * 0.3 + 0.2)
            home_boost = 1.1 if fix["home"] else 0.95

            adj_xg = xg_per90 * scale * home_boost
            adj_xa = xa_per90 * scale * home_boost

            # Clean sheet probability
            opp_atk_str = opp_odds.get("attack_strength", 1.0)
            team_def_str = team_attack_odds.get("defence_strength", 1.0)
            # Lower opponent attack + lower own conceding = higher CS prob
            base_cs = 0.30  # league average ~30%
            cs_prob = base_cs * (1.0 / max(opp_atk_str, 0.3)) * (1.0 / max(team_def_str, 0.3))
            cs_prob = min(cs_prob, 0.65)  # cap at 65%

            # Use odds-derived CS if available
            team_cs_from_odds = team_attack_odds.get("cs_prob")
            if team_cs_from_odds:
                cs_prob = (cs_prob + team_cs_from_odds) / 2

            # Calculate expected FPL points for this fixture
            xpts = 0.0

            # Appearance points (2 pts if 60+ mins, 1 pt if <60 mins)
            xpts += 2.0 * full_game_prob + 1.0 * max(play_prob - full_game_prob, 0)

            # Goal points (scale by expected 90s played, not just binary)
            xpts += adj_xg * expected_90s * PTS_GOAL.get(pos, 4)

            # Assist points
            xpts += adj_xa * expected_90s * PTS_ASSIST

            # Clean sheet points (GK and DEF mainly — need 60+ mins)
            xpts += cs_prob * PTS_CS.get(pos, 0) * full_game_prob

            # Bonus points estimate (proportional to involvement)
            xpts += PTS_BONUS_AVG * play_prob

            # Goals conceded penalty for GK/DEF (approx -0.5 per goal after first)
            if pos in [1, 2]:
                expected_conceded = league_avg_goals * opp_atk_str * home_boost
                xpts -= max(0, (expected_conceded - 1)) * 0.5 * full_game_prob

            # Save points for GKs (~1 pt per game on average)
            if pos == 1:
                xpts += 1.0 * full_game_prob

            player_gw_xpts[gw] = round(max(xpts, 0), 2)

        # Total xPts over next 6 GWs
        xpts_all[pid] = player_gw_xpts

    return xpts_all


# ============================================================
# MILP SOLVER
# ============================================================

def solve_optimal_squad(players_df, xpts_col="xpts_total", budget=1000):
    """
    MILP optimisation using PuLP.
    Selects 15 players maximising total expected points subject to:
      - Budget: sum(cost) <= budget (in 0.1m units)
      - Exactly 2 GKP, 5 DEF, 5 MID, 3 FWD
      - Max 3 players per team
    Returns: DataFrame of selected 15 players, or None
    """
    eligible = players_df[
        (players_df["minutes"] > 45) &
        (players_df["status"].isin(["a", "d", ""])) &
        (players_df[xpts_col] > 0)
    ].copy()

    if len(eligible) < 15:
        return None, "Not enough eligible players"

    # Ensure no NaN values in key columns (NaN crashes PuLP)
    eligible[xpts_col] = eligible[xpts_col].fillna(0).astype(float)
    eligible["now_cost"] = eligible["now_cost"].fillna(0).astype(int)
    eligible = eligible[eligible[xpts_col].notna() & eligible["now_cost"].notna()]

    if len(eligible) < 15:
        return None, "Not enough eligible players after NaN removal"

    prob = LpProblem("FPL_Squad", LpMaximize)

    # Build lookup dicts for fast, safe access
    eligible = eligible.reset_index(drop=True)
    pid_to_idx = {row["id"]: i for i, row in eligible.iterrows()}
    xpts_vals = eligible[xpts_col].tolist()
    cost_vals = eligible["now_cost"].tolist()
    pos_vals = eligible["pos_id"].tolist()
    team_vals = eligible["team_id"].tolist()

    # Decision variables: x_i = 1 if player i is selected
    player_ids = eligible["id"].tolist()
    x = {pid: LpVariable(f"x_{pid}", cat="Binary") for pid in player_ids}

    # Objective: maximise total xPts
    prob += lpSum(x[pid] * xpts_vals[pid_to_idx[pid]] for pid in player_ids)

    # Budget constraint
    prob += lpSum(x[pid] * cost_vals[pid_to_idx[pid]] for pid in player_ids) <= budget

    # Squad size = 15
    prob += lpSum(x[pid] for pid in player_ids) == 15

    # Position constraints: 2 GK, 5 DEF, 5 MID, 3 FWD
    for pos_id, count in [(1, 2), (2, 5), (3, 5), (4, 3)]:
        pos_pids = [pid for pid in player_ids if pos_vals[pid_to_idx[pid]] == pos_id]
        prob += lpSum(x[pid] for pid in pos_pids) == count

    # Max 3 per team
    for team_id in set(team_vals):
        team_pids = [pid for pid in player_ids if team_vals[pid_to_idx[pid]] == team_id]
        prob += lpSum(x[pid] for pid in team_pids) <= 3

    # Solve
    try:
        solver = PULP_CBC_CMD(msg=0, timeLimit=30)
        prob.solve(solver)
    except Exception as e:
        return None, f"Solver error: {e}"

    if LpStatus[prob.status] != "Optimal":
        return None, f"Solver status: {LpStatus[prob.status]}"

    # Extract selected players
    selected_ids = [pid for pid in player_ids if value(x[pid]) is not None and value(x[pid]) > 0.5]
    squad = eligible[eligible["id"].isin(selected_ids)].copy()

    return squad, None


def solve_best_xi(squad_df, xpts_col="xpts_next_gw"):
    """
    From a 15-man squad, pick the best starting XI using MILP.
    Must have exactly 1 GK, and valid formation (3-5-2, 4-4-2, etc.)
    """
    if squad_df is None or len(squad_df) < 11:
        return None, None

    prob = LpProblem("FPL_XI", LpMaximize)
    squad_df = squad_df.reset_index(drop=True)
    pids = squad_df["id"].tolist()
    pid_to_idx = {row["id"]: i for i, row in squad_df.iterrows()}
    xpts_vals = squad_df[xpts_col].fillna(0).tolist()
    pos_list = squad_df["pos_id"].tolist()
    x = {pid: LpVariable(f"xi_{pid}", cat="Binary") for pid in pids}

    # Objective
    prob += lpSum(x[pid] * xpts_vals[pid_to_idx[pid]] for pid in pids)

    # Exactly 11 starters
    prob += lpSum(x[pid] for pid in pids) == 11

    # Exactly 1 GK
    gk_pids = [pid for pid in pids if pos_list[pid_to_idx[pid]] == 1]
    prob += lpSum(x[pid] for pid in gk_pids) == 1

    # DEF: 3-5
    def_pids = [pid for pid in pids if pos_list[pid_to_idx[pid]] == 2]
    prob += lpSum(x[pid] for pid in def_pids) >= 3
    prob += lpSum(x[pid] for pid in def_pids) <= 5

    # MID: 2-5
    mid_pids = [pid for pid in pids if pos_list[pid_to_idx[pid]] == 3]
    prob += lpSum(x[pid] for pid in mid_pids) >= 2
    prob += lpSum(x[pid] for pid in mid_pids) <= 5

    # FWD: 1-3
    fwd_pids = [pid for pid in pids if pos_list[pid_to_idx[pid]] == 4]
    prob += lpSum(x[pid] for pid in fwd_pids) >= 1
    prob += lpSum(x[pid] for pid in fwd_pids) <= 3

    try:
        solver = PULP_CBC_CMD(msg=0, timeLimit=15)
        prob.solve(solver)
    except Exception:
        return None, None

    if LpStatus[prob.status] != "Optimal":
        return None, None

    xi_ids = [pid for pid in pids if value(x[pid]) is not None and value(x[pid]) > 0.5]
    xi = squad_df[squad_df["id"].isin(xi_ids)].copy()
    bench = squad_df[~squad_df["id"].isin(xi_ids)].copy()
    return xi, bench


# ============================================================
# DATA ENRICHMENT
# ============================================================

def enrich_data(bootstrap, fixtures, team_odds):
    """Combine all data sources into a single enriched DataFrame."""
    players_raw = bootstrap["elements"]
    teams = {t["id"]: t for t in bootstrap["teams"]}
    events = bootstrap["events"]
    current_gw = (
        next((e for e in events if e["is_current"]), None) or
        next((e for e in events if e["is_next"]), None) or
        (events[0] if events else None)
    )
    gw_id = current_gw["id"] if current_gw else 1

    # Build upcoming fixtures
    upcoming = {t_id: [] for t_id in teams}
    for f in sorted(fixtures, key=lambda x: x.get("event", 0) or 0):
        ev = f.get("event")
        if ev and gw_id <= ev < gw_id + 6:
            if f["team_h"] in upcoming:
                upcoming[f["team_h"]].append({
                    "gw": ev, "opp": f["team_a"], "home": True,
                    "difficulty": f.get("team_h_difficulty", 3)
                })
            if f["team_a"] in upcoming:
                upcoming[f["team_a"]].append({
                    "gw": ev, "opp": f["team_h"], "home": False,
                    "difficulty": f.get("team_a_difficulty", 3)
                })

    # Recent results
    recent = {t_id: [] for t_id in teams}
    for f in sorted(fixtures, key=lambda x: x.get("event", 0) or 0, reverse=True):
        if f.get("finished") and f.get("team_h_score") is not None:
            h = "W" if f["team_h_score"] > f["team_a_score"] else ("D" if f["team_h_score"] == f["team_a_score"] else "L")
            a = "W" if f["team_a_score"] > f["team_h_score"] else ("D" if f["team_a_score"] == f["team_h_score"] else "L")
            if len(recent.get(f["team_h"], [])) < 5:
                recent[f["team_h"]].append(h)
            if len(recent.get(f["team_a"], [])) < 5:
                recent[f["team_a"]].append(a)

    # Build player rows
    rows = []
    for p in players_raw:
        td = teams.get(p["team"], {})
        price = p["now_cost"] / 10
        mins = p.get("minutes", 0) or 0
        pts = p.get("total_points", 0) or 0

        rows.append({
            "id": p["id"],
            "name": p.get("web_name", ""),
            "first_name": p.get("first_name", ""),
            "second_name": p.get("second_name", ""),
            "team_id": p["team"],
            "team": td.get("short_name", "???"),
            "team_name": td.get("name", "???"),
            "pos_id": p["element_type"],
            "pos": POS_MAP.get(p["element_type"], "?"),
            "price": price,
            "now_cost": p["now_cost"],
            "total_points": pts,
            "form": float(p.get("form", 0) or 0),
            "form_str": str(p.get("form", "0.0")),
            "ict_index": round(float(p.get("ict_index", 0) or 0), 1),
            "minutes": mins,
            "starts": p.get("starts", 0) or 0,
            "goals": p.get("goals_scored", 0) or 0,
            "assists": p.get("assists", 0) or 0,
            "clean_sheets": p.get("clean_sheets", 0) or 0,
            "xg_per90": float(p.get("expected_goals_per_90", 0) or 0),
            "xa_per90": float(p.get("expected_assists_per_90", 0) or 0),
            "xgi_per90": float(p.get("expected_goal_involvements_per_90", 0) or 0),
            "xgc_per90": float(p.get("expected_goals_conceded_per_90", 0) or 0),
            "selected_pct": float(p.get("selected_by_percent", 0) or 0),
            "transfers_in": p.get("transfers_in_event", 0) or 0,
            "transfers_out": p.get("transfers_out_event", 0) or 0,
            "status": p.get("status", "a"),
            "chance_playing": p.get("chance_of_playing_next_round"),
            "news": p.get("news", ""),
            "ppg": float(p.get("points_per_game", 0) or 0),
            "upcoming": upcoming.get(p["team"], []),
            "team_form": recent.get(p["team"], []),
        })

    df = pd.DataFrame(rows)

    # Build xPts model
    xpts_map = build_xpts_model(df, team_odds, teams, fixtures, gw_id)

    # Add xPts columns
    df["xpts_next_gw"] = df["id"].map(
        lambda pid: list(xpts_map.get(pid, {}).values())[0] if xpts_map.get(pid) else 0
    )
    df["xpts_total"] = df["id"].map(
        lambda pid: sum(xpts_map.get(pid, {}).values())
    )

    # Avg fixture difficulty
    df["avg_difficulty"] = df["upcoming"].apply(
        lambda u: round(np.mean([f["difficulty"] for f in u[:4]]), 2) if u else 3.0
    )

    # Value metric
    df["value"] = df.apply(
        lambda r: round(r["xpts_total"] / max(r["price"], 1), 2), axis=1
    )

    return df, teams, current_gw, upcoming, fixtures, xpts_map


# ============================================================
# UI HELPERS
# ============================================================

def get_formation_str(xi_df):
    """Get formation string like '4-4-2' from starting XI."""
    if xi_df is None:
        return "-"
    d = len(xi_df[xi_df["pos_id"] == 2])
    m = len(xi_df[xi_df["pos_id"] == 3])
    f = len(xi_df[xi_df["pos_id"] == 4])
    return f"{d}-{m}-{f}"


def render_fdr(upcoming, teams):
    """Render fixture difficulty badges."""
    badges = []
    for f in upcoming[:5]:
        opp = teams.get(f["opp"], {}).get("short_name", "???")
        pre = "" if f["home"] else "@"
        d = f.get("difficulty", 3)
        badges.append(f'<span class="fdr-{d}">{pre}{opp}</span>')
    return " ".join(badges) if badges else "-"


# ============================================================
# MAIN APP
# ============================================================

def main():
    st.markdown('<div class="main-title">⚽ FPL Optimizer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">'
        'MILP-optimised squad selection · FPL API + Betting Odds + Custom xPts Model'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    # === Load data ===
    with st.spinner("Loading FPL API data..."):
        bootstrap, fixtures_raw, fpl_err = load_fpl_data()

    if fpl_err or bootstrap is None:
        st.error(f"Failed to load FPL data: {fpl_err}")
        if st.button("🔄 Retry"):
            st.cache_data.clear()
            st.rerun()
        return

    with st.spinner("Loading betting odds from football-data.co.uk..."):
        odds_df, odds_err = load_betting_odds()

    odds_status = "✅ Loaded" if odds_df is not None else f"⚠️ {odds_err or 'Unavailable'}"
    team_odds = odds_to_probabilities(odds_df, TEAM_NAME_MAP) if odds_df is not None else {}

    with st.spinner("Building xPts model & enriching data..."):
        df, teams, current_gw, upcoming_map, fixtures_list, xpts_map = enrich_data(
            bootstrap, fixtures_raw, team_odds
        )

    # === GW Info ===
    if current_gw:
        deadline = datetime.fromisoformat(current_gw["deadline_time"].replace("Z", "+00:00"))
        status = "Completed" if current_gw.get("finished") else ("In Progress" if current_gw.get("is_current") else "Upcoming")
        bc = "badge-green" if status == "Completed" else ("badge-yellow" if status == "In Progress" else "badge-blue")
        st.markdown(f"""<div class="gw-bar">
            <span class="gw-num">Gameweek {current_gw['id']}</span>
            <span class="gw-deadline">Deadline: {deadline.strftime('%a %d %b, %H:%M')}</span>
            <span class="badge {bc}">{status}</span>
            <span style="color:#5a6580; font-size:0.7rem;">Odds: {odds_status} · {len(team_odds)} teams matched</span>
        </div>""", unsafe_allow_html=True)

    # === Tabs ===
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard", "👥 Player Projections", "⭐ Optimal Squad (MILP)",
        "🔄 Transfer Planner", "📅 Fixtures"
    ])

    active = df[df["minutes"] > 0].copy()
    qualified = df[df["minutes"] > 45].copy()

    # ==================== DASHBOARD ====================
    with tab1:
        if len(active) > 0:
            c1, c2, c3, c4 = st.columns(4)
            top_xpts = qualified.loc[qualified["xpts_total"].idxmax()] if len(qualified) > 0 else None
            top_scorer = active.loc[active["total_points"].idxmax()]
            top_value = qualified.loc[qualified["value"].idxmax()] if len(qualified) > 0 else None
            top_form = active.loc[active["form"].idxmax()]

            cards = [
                (c1, "Highest xPts (6GW)", f"{top_xpts['xpts_total']:.1f}" if top_xpts is not None else "-",
                 f"{top_xpts['name']} (£{top_xpts['price']:.1f}m)" if top_xpts is not None else ""),
                (c2, "Top Scorer", str(int(top_scorer["total_points"])),
                 f"{top_scorer['name']} ({top_scorer['team']})"),
                (c3, "Best Value (xPts/£)", f"{top_value['value']:.1f}" if top_value is not None else "-",
                 f"{top_value['name']} (£{top_value['price']:.1f}m)" if top_value is not None else ""),
                (c4, "Hottest Form", f"{top_form['form']:.1f}",
                 f"{top_form['name']} ({top_form['team']})"),
            ]
            for col, label, val, sub in cards:
                with col:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{val}</div>
                        <div class="metric-sub">{sub}</div>
                    </div>""", unsafe_allow_html=True)

        st.markdown("")
        st.markdown('<div class="section-header">Top xPts Picks — Next 6 Gameweeks <span class="source-tag src-model">xPts Model</span></div>', unsafe_allow_html=True)
        if len(qualified) > 0:
            tp = qualified.nlargest(15, "xpts_total")[
                ["name", "team", "pos", "price", "total_points", "xpts_next_gw", "xpts_total", "value"]
            ].copy()
            tp.columns = ["Player", "Team", "Pos", "Price (£m)", "Actual Pts", "xPts Next GW", "xPts 6GW", "Value"]
            tp = tp.reset_index(drop=True)
            tp.index += 1
            st.dataframe(tp, use_container_width=True, height=540)

        st.markdown("")
        st.markdown('<div class="section-header">Differentials — Under 10% Ownership</div>', unsafe_allow_html=True)
        diffs = qualified[(qualified["selected_pct"] < 10) & (qualified["xpts_total"] > 0)].nlargest(10, "xpts_total")
        if len(diffs) > 0:
            dd = diffs[["name", "team", "pos", "price", "selected_pct", "xpts_total", "value"]].copy()
            dd.columns = ["Player", "Team", "Pos", "Price", "Own%", "xPts 6GW", "Value"]
            dd = dd.reset_index(drop=True)
            dd.index += 1
            st.dataframe(dd, use_container_width=True, height=380)

    # ==================== PLAYER PROJECTIONS ====================
    with tab2:
        fc1, fc2, fc3, fc4, fc5 = st.columns([2, 1, 1, 1, 1])
        with fc1:
            search = st.text_input("🔍 Search", "", key="ps2")
        with fc2:
            pos_f = st.selectbox("Position", ["All"] + list(POS_FULL.values()), key="pf2")
        with fc3:
            team_f = st.selectbox("Team", ["All"] + sorted(df["team_name"].unique().tolist()), key="tf2")
        with fc4:
            price_f = st.selectbox("Price", ["All", "Under £5m", "£5-7m", "£7-10m", "Over £10m"], key="prf2")
        with fc5:
            so = {"xPts (6GW)": "xpts_total", "xPts Next GW": "xpts_next_gw", "Total Pts": "total_points",
                  "Form": "form", "Value": "value", "ICT": "ict_index"}
            sort_f = st.selectbox("Sort by", list(so.keys()), key="sf2")

        fl = active.copy()
        if search:
            sl = search.lower()
            fl = fl[fl["name"].str.lower().str.contains(sl, na=False) |
                     fl["second_name"].str.lower().str.contains(sl, na=False)]
        if pos_f != "All":
            fl = fl[fl["pos_id"] == {v: k for k, v in POS_FULL.items()}[pos_f]]
        if team_f != "All":
            fl = fl[fl["team_name"] == team_f]
        if price_f == "Under £5m": fl = fl[fl["price"] < 5]
        elif price_f == "£5-7m": fl = fl[(fl["price"] >= 5) & (fl["price"] < 7)]
        elif price_f == "£7-10m": fl = fl[(fl["price"] >= 7) & (fl["price"] < 10)]
        elif price_f == "Over £10m": fl = fl[fl["price"] >= 10]

        fl = fl.sort_values(so[sort_f], ascending=False)
        sd = fl.head(80)[["name", "team", "pos", "price", "total_points", "form_str",
                           "xpts_next_gw", "xpts_total", "xg_per90", "xa_per90",
                           "selected_pct", "value"]].copy()
        sd.columns = ["Player", "Team", "Pos", "Price", "Pts", "Form",
                       "xPts GW", "xPts 6GW", "xG/90", "xA/90", "Own%", "Value"]
        sd = sd.reset_index(drop=True)
        sd.index += 1
        st.dataframe(sd, use_container_width=True, height=700)
        st.caption(f"Showing {min(80, len(fl))} of {len(fl)} players · xPts model blends FPL xG/xA + betting odds")

    # ==================== OPTIMAL SQUAD (MILP) ====================
    with tab3:
        st.markdown(
            '<div class="section-header">⭐ MILP-Optimised Squad '
            '<span class="source-tag src-model">PuLP Solver</span></div>',
            unsafe_allow_html=True,
        )
        st.caption("Mathematically optimal 15-man squad maximising total xPts over 6 gameweeks. "
                    "Constraints: £100m budget, 2 GK / 5 DEF / 5 MID / 3 FWD, max 3 per team.")

        if len(qualified) > 0:
            with st.spinner("Running MILP solver..."):
                squad, solve_err = solve_optimal_squad(qualified, "xpts_total", 1000)

            if squad is not None and len(squad) == 15:
                # Solve best XI
                xi, bench = solve_best_xi(squad, "xpts_next_gw")

                total_cost = squad["now_cost"].sum() / 10
                total_xpts = squad["xpts_total"].sum()
                xi_xpts = xi["xpts_next_gw"].sum() if xi is not None else 0
                formation = get_formation_str(xi)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Cost", f"£{total_cost:.1f}m")
                c2.metric("Squad xPts (6GW)", f"{total_xpts:.1f}")
                c3.metric("XI xPts (Next GW)", f"{xi_xpts:.1f}")
                c4.metric("Formation", formation)

                st.markdown("")

                # Render pitch view
                if xi is not None:
                    for pid, plabel in [(4, "Forwards"), (3, "Midfielders"), (2, "Defenders"), (1, "Goalkeeper")]:
                        pp = xi[xi["pos_id"] == pid]
                        if len(pp) > 0:
                            st.markdown(f"<div class='pitch-row-label'>{plabel}</div>", unsafe_allow_html=True)
                            cols = st.columns(max(len(pp), 1))
                            for i, (_, p) in enumerate(pp.iterrows()):
                                sc = f"pitch-shirt-{p['pos'].lower()}"
                                with cols[i]:
                                    st.markdown(f"""<div style="text-align:center;">
                                        <div class="pitch-shirt {sc}">{p['xpts_next_gw']:.1f}</div>
                                        <div class="pitch-name">{p['name']}</div>
                                        <div class="pitch-price">£{p['price']:.1f}m · {p['form_str']}</div>
                                    </div>""", unsafe_allow_html=True)

                if bench is not None and len(bench) > 0:
                    st.markdown("**Bench**")
                    bcols = st.columns(len(bench))
                    for i, (_, p) in enumerate(bench.iterrows()):
                        with bcols[i]:
                            st.markdown(f"""<div style="text-align:center;opacity:0.65;">
                                <div class="pitch-name">{p['name']}</div>
                                <div class="pitch-price">{p['pos']} · £{p['price']:.1f}m · {p['xpts_next_gw']:.1f}xPts</div>
                            </div>""", unsafe_allow_html=True)

                st.markdown("")
                st.subheader("Full Squad Breakdown")
                sq = squad.sort_values(["pos_id", "xpts_total"], ascending=[True, False])
                sq_show = sq[["name", "team", "pos", "price", "total_points", "xpts_next_gw", "xpts_total", "value"]].copy()
                sq_show.columns = ["Player", "Team", "Pos", "Price", "Actual Pts", "xPts GW", "xPts 6GW", "Value"]
                sq_show = sq_show.reset_index(drop=True)
                sq_show.index += 1
                st.dataframe(sq_show, use_container_width=True)
            else:
                st.warning(f"Could not find optimal squad: {solve_err}")

    # ==================== TRANSFER PLANNER ====================
    with tab4:
        st.markdown('<div class="section-header">🔄 Transfer Suggestions <span class="source-tag src-model">xPts Model</span></div>', unsafe_allow_html=True)
        st.caption("Finds the highest-xPts replacements for underperforming popular players, matched by position.")

        if len(qualified) > 0:
            cands = qualified.nlargest(30, "xpts_total")
            outs = qualified[qualified["selected_pct"] > 15].nsmallest(15, "xpts_total")
            transfers, ui, uo = [], set(), set()

            for _, ip in cands.iterrows():
                for _, op in outs.iterrows():
                    if ip["id"] in ui or op["id"] in uo:
                        continue
                    if ip["pos_id"] != op["pos_id"] or ip["id"] == op["id"]:
                        continue
                    if ip["xpts_total"] <= op["xpts_total"] * 1.15:
                        continue
                    reasons = []
                    if ip["xpts_next_gw"] > op["xpts_next_gw"]:
                        reasons.append(f"+{ip['xpts_next_gw'] - op['xpts_next_gw']:.1f} xPts next GW")
                    if ip["avg_difficulty"] < op["avg_difficulty"]:
                        reasons.append("Easier fixtures")
                    if ip["form"] > op["form"]:
                        reasons.append("Better form")
                    if not reasons:
                        reasons.append(f"+{ip['xpts_total'] - op['xpts_total']:.1f} xPts over 6GW")
                    transfers.append({"out": op, "in": ip, "reasons": reasons})
                    ui.add(ip["id"])
                    uo.add(op["id"])
                    if len(transfers) >= 8:
                        break
                if len(transfers) >= 8:
                    break

            if transfers:
                for t in transfers:
                    o, i = t["out"], t["in"]
                    rs = " · ".join(t["reasons"])
                    st.markdown(f"""<div class="transfer-card">
                        <span class="transfer-out">▼ {o['name']}</span>
                        <span style="color:#5a6580;font-size:0.72rem;"> {o['pos']} · {o['team']} · £{o['price']:.1f}m · {o['xpts_total']:.1f}xPts</span>
                        &nbsp;<span class="transfer-arrow">→</span>&nbsp;
                        <span class="transfer-in">▲ {i['name']}</span>
                        <span style="color:#5a6580;font-size:0.72rem;"> {i['pos']} · {i['team']} · £{i['price']:.1f}m · {i['xpts_total']:.1f}xPts</span>
                        <br><span style="color:#8892a8;font-size:0.7rem;">{rs}</span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.info("No clear transfer improvements found.")

        st.markdown("")
        cr, cf = st.columns(2)
        with cr:
            st.subheader("📈 Most Transferred In")
            ri = active.nlargest(10, "transfers_in")[["name", "team", "pos", "price", "transfers_in", "xpts_total"]].copy()
            ri.columns = ["Player", "Team", "Pos", "Price", "In", "xPts 6GW"]
            ri = ri.reset_index(drop=True); ri.index += 1
            st.dataframe(ri, use_container_width=True)
        with cf:
            st.subheader("📉 Most Transferred Out")
            fo = active.nlargest(10, "transfers_out")[["name", "team", "pos", "price", "transfers_out", "xpts_total"]].copy()
            fo.columns = ["Player", "Team", "Pos", "Price", "Out", "xPts 6GW"]
            fo = fo.reset_index(drop=True); fo.index += 1
            st.dataframe(fo, use_container_width=True)

    # ==================== FIXTURES ====================
    with tab5:
        st.markdown('<div class="section-header">Fixture Difficulty — Next 6 Gameweeks <span class="source-tag src-odds">Odds-enhanced</span></div>', unsafe_allow_html=True)
        gw_id = current_gw["id"] if current_gw else 1
        gw_range = list(range(gw_id, gw_id + 6))

        fm = {t_id: {} for t_id in teams}
        for f in fixtures_list:
            if f.get("event") in gw_range:
                if f["team_h"] in fm:
                    fm[f["team_h"]][f["event"]] = {"opp": f["team_a"], "home": True, "diff": f.get("team_h_difficulty", 3)}
                if f["team_a"] in fm:
                    fm[f["team_a"]][f["event"]] = {"opp": f["team_h"], "home": False, "diff": f.get("team_a_difficulty", 3)}

        rows = []
        for t_id, td in teams.items():
            row = {"Team": td["short_name"]}
            diffs = []
            t_short = td["short_name"]
            t_odds = {v: team_odds.get(k) for k, v in TEAM_NAME_MAP.items()}.get(t_short, {})

            for gw in gw_range:
                fix = fm.get(t_id, {}).get(gw)
                if fix:
                    opp = teams.get(fix["opp"], {}).get("short_name", "???")
                    pre = "" if fix["home"] else "@"
                    row[f"GW{gw}"] = f"{pre}{opp} ({fix['diff']})"
                    diffs.append(fix["diff"])
                else:
                    row[f"GW{gw}"] = "-"

            row["Avg FDR"] = round(np.mean(diffs), 1) if diffs else 3.0

            # Add odds-derived CS probability if available
            if t_odds and isinstance(t_odds, dict):
                row["CS%"] = f"{t_odds.get('cs_prob', 0)*100:.0f}%"
                row["Atk Str"] = f"{t_odds.get('attack_strength', 1):.2f}"
            else:
                row["CS%"] = "-"
                row["Atk Str"] = "-"

            rows.append(row)

        fdf = pd.DataFrame(rows).sort_values("Avg FDR").reset_index(drop=True)
        fdf.index += 1
        st.dataframe(fdf, use_container_width=True, height=740)

    # === Footer ===
    st.markdown("---")
    st.markdown(
        f"<div style='text-align:center; color:#5a6580; font-size:0.7rem;'>"
        f"Data: FPL API + football-data.co.uk odds · Solver: PuLP (MILP) · "
        f"Cached 1hr · {datetime.now().strftime('%d %b %Y, %H:%M')}"
        f"</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
