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
import math
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
    page_title="Datumly — FPL Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

FPL_BASE = "https://fantasy.premierleague.com/api"
FOOTBALL_DATA_URL = "https://www.football-data.co.uk/mmz4281/2526/E0.csv"

POS_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

DATUMLY_LOGO_B64 = "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"
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
        background: linear-gradient(135deg, #f02d6e, #e8456e, #ff6b8a);
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
    .gw-num { background: linear-gradient(135deg,#f02d6e,#ff6b8a); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 1.05rem; font-weight: 700; }
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


@st.cache_data(ttl=3600)
def load_recent_gw_live_data(current_gw_id, n_recent=7):
    """
    Fetch per-GW live stats for the last N completed gameweeks.
    Uses event/{gw}/live/ endpoint — one call per GW, returns all players.
    Returns: dict {player_id: [{gw, minutes, xG, xA, xGC, goals, assists, ...}, ...]}
    """
    headers = {"User-Agent": "FPL-Optimizer/2.0"}
    player_gw_data = {}

    start_gw = max(1, current_gw_id - n_recent)
    for gw in range(start_gw, current_gw_id):
        try:
            resp = requests.get(
                f"{FPL_BASE}/event/{gw}/live/",
                headers=headers, timeout=15,
            )
            if resp.status_code != 200:
                continue
            data = resp.json()
            elements = data.get("elements", [])
            for el in elements:
                pid = el["id"]
                stats = el.get("stats", {})
                if stats.get("minutes", 0) == 0:
                    continue
                if pid not in player_gw_data:
                    player_gw_data[pid] = []
                player_gw_data[pid].append({
                    "gw": gw,
                    "minutes": stats.get("minutes", 0),
                    "goals": stats.get("goals_scored", 0),
                    "assists": stats.get("assists", 0),
                    "xG": float(stats.get("expected_goals", 0) or 0),
                    "xA": float(stats.get("expected_assists", 0) or 0),
                    "xGC": float(stats.get("expected_goals_conceded", 0) or 0),
                    "xGI": float(stats.get("expected_goal_involvements", 0) or 0),
                    "clean_sheets": stats.get("clean_sheets", 0),
                    "bonus": stats.get("bonus", 0),
                    "total_points": stats.get("total_points", 0),
                })
        except Exception:
            continue

    return player_gw_data


def compute_form_weighted_xg(player_gw_data, n_recent=7):
    """
    Compute form-weighted xG/90 and xA/90 from recent GW data.
    Uses exponential decay: most recent GW has highest weight.
    Returns: dict {player_id: {xg_form_per90, xa_form_per90, xgc_form_per90, form_minutes}}
    """
    result = {}
    decay_factor = 0.85  # each older GW is worth 85% of the next

    for pid, gw_list in player_gw_data.items():
        # Sort by GW descending (most recent first)
        sorted_gws = sorted(gw_list, key=lambda x: x["gw"], reverse=True)[:n_recent]

        if not sorted_gws:
            continue

        weighted_xg = 0
        weighted_xa = 0
        weighted_xgc = 0
        weighted_mins = 0
        total_weight = 0

        for i, gw_data in enumerate(sorted_gws):
            weight = decay_factor ** i  # most recent = 1.0, then 0.85, 0.72, 0.61...
            mins = gw_data["minutes"]
            if mins > 0:
                weighted_xg += gw_data["xG"] * weight
                weighted_xa += gw_data["xA"] * weight
                weighted_xgc += gw_data["xGC"] * weight
                weighted_mins += mins * weight
                total_weight += weight

        if weighted_mins > 0 and total_weight > 0:
            nineties = weighted_mins / 90.0
            result[pid] = {
                "xg_form_per90": weighted_xg / nineties,
                "xa_form_per90": weighted_xa / nineties,
                "xgc_form_per90": weighted_xgc / nineties,
                "form_minutes": weighted_mins / total_weight,  # avg weighted mins
                "form_gws": len(sorted_gws),
            }

    return result


def detect_blank_double_gws(fixtures, planning_gw_id, n_gws=6, teams=None):
    """
    Detect blank and double gameweeks from fixture data.
    Returns: dict {team_id: {gw: fixture_count}} where 0 = blank, 2+ = double
    """
    if teams is None:
        teams = {}

    team_fixture_counts = {}
    for t_id in teams:
        team_fixture_counts[t_id] = {}
        for gw in range(planning_gw_id, planning_gw_id + n_gws):
            team_fixture_counts[t_id][gw] = 0

    for f in fixtures:
        ev = f.get("event")
        if ev and planning_gw_id <= ev < planning_gw_id + n_gws:
            if f["team_h"] in team_fixture_counts:
                team_fixture_counts[f["team_h"]][ev] = team_fixture_counts[f["team_h"]].get(ev, 0) + 1
            if f["team_a"] in team_fixture_counts:
                team_fixture_counts[f["team_a"]][ev] = team_fixture_counts[f["team_a"]].get(ev, 0) + 1

    return team_fixture_counts


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


def build_xpts_model(players_df, team_odds, teams_map, fixtures, current_gw_id,
                     form_xg_data=None, team_fixture_counts=None):
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
        # PER-90 STATS — blend season average with form-weighted recent
        # ============================================================
        mins_played = max(mins, 1)
        nineties = mins_played / 90.0

        # Season-average xG/xA from FPL API
        season_xg = float(p.get("xg_per90", 0) or 0)
        season_xa = float(p.get("xa_per90", 0) or 0)

        # Fallback: use actual goals/assists per 90 ONLY if enough minutes
        if season_xg == 0 and p["goals"] > 0 and mins >= 270:
            season_xg = p["goals"] / nineties
        if season_xa == 0 and p["assists"] > 0 and mins >= 270:
            season_xa = p["assists"] / nineties

        # Form-weighted xG/xA from recent 7 GWs (if available)
        form_data = (form_xg_data or {}).get(pid)
        if form_data and form_data.get("form_gws", 0) >= 3:
            form_xg = form_data["xg_form_per90"]
            form_xa = form_data["xa_form_per90"]
            # Blend: 60% recent form, 40% season average
            # (recent form is more predictive but noisier)
            xg_per90 = form_xg * 0.6 + season_xg * 0.4
            xa_per90 = form_xa * 0.6 + season_xa * 0.4
        else:
            xg_per90 = season_xg
            xa_per90 = season_xa

        # Also get form-weighted xGC for GKs/DEFs
        form_xgc = None
        if form_data and form_data.get("form_gws", 0) >= 3:
            form_xgc = form_data.get("xgc_form_per90")

        # Apply regression to the mean for low-sample players
        sample_weight = min(nineties / 10.0, 1.0)
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

            # Base CS from odds — use form-weighted xGC if available, else season
            actual_xgc_per90 = float(p.get("xgc_per90", 0) or 0)
            if form_xgc is not None and form_xgc > 0:
                # Blend form xGC with season xGC (form is more recent)
                actual_xgc_per90 = form_xgc * 0.65 + actual_xgc_per90 * 0.35
            if pos in [1, 2] and actual_xgc_per90 > 0:
                # Use actual xGC as a strong signal for defensive quality
                # Poisson approximation: P(0 goals) ≈ e^(-xGC)
                base_cs = math.exp(-actual_xgc_per90 * opp_atk_str)
            else:
                # Fallback: odds-based estimate, penalised by opponent attack
                base_cs = 0.30 * (1.0 / max(opp_atk_str, 0.5))
                # Penalise teams that concede a lot (team_def_str > 1 = concede more)
                base_cs *= (1.0 / max(team_def_str, 0.5))
                base_cs = min(base_cs, 0.50)

            # Blend with odds-derived CS if available (but don't override Poisson)
            team_cs_from_odds = team_attack_odds.get("cs_prob")
            if team_cs_from_odds and actual_xgc_per90 == 0:
                cs_prob = (base_cs * 0.4 + team_cs_from_odds * 0.6)
            else:
                cs_prob = base_cs

            # Hard cap — no team keeps a CS more than 50% of the time
            cs_prob = min(cs_prob, 0.50)

            # Apply recent form adjustment: if team has lost 3+ of last 5, reduce CS further
            try:
                team_form_list = p["team_form"] if "team_form" in p.index else []
                if isinstance(team_form_list, list) and len(team_form_list) >= 3:
                    recent_losses = sum(1 for r in team_form_list[:5] if r == "L")
                    if recent_losses >= 3:
                        cs_prob *= 0.6  # 40% penalty for bad recent form
                    elif recent_losses >= 2:
                        cs_prob *= 0.8  # 20% penalty
            except (KeyError, TypeError):
                pass

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

            # Goals conceded penalty for GK/DEF
            # FPL rule: -1 point per 2 goals conceded (i.e. -0.5 per goal) from goal 1
            if pos in [1, 2]:
                expected_conceded = league_avg_goals * opp_atk_str
                if not fix["home"]:
                    expected_conceded *= 1.1  # away teams concede slightly more
                xpts -= expected_conceded * 0.5 * full_game_prob

            # Save points for GKs: ~1pt per 3 saves
            # Better estimate: GKs on bad teams face more shots but also concede more
            # Average ~3.5 saves/game across PL, but scale by opponent attack
            if pos == 1:
                expected_saves = 3.5 * opp_atk_str * 0.7  # not all shots are saveable
                save_points = (expected_saves / 3.0)  # 1 pt per 3 saves
                xpts += save_points * full_game_prob

            # Accumulate xPts — important for DGWs where a player has 2 fixtures in same GW
            gw_xpts_so_far = player_gw_xpts.get(gw, 0)
            player_gw_xpts[gw] = round(gw_xpts_so_far + max(xpts, 0), 2)

        # Apply blank GW override
        # Blank GW (0 fixtures) = 0 xPts — fixture loop won't have added anything,
        # but we set explicitly to be safe.
        # DGWs are already handled: the fixture loop processes both fixtures and
        # accumulates xPts, so a DGW player naturally gets ~2x a single-GW player.
        # This means DGW players only rank higher if their per-fixture xPts justify it.
        if team_fixture_counts:
            team_counts = team_fixture_counts.get(p["team_id"], {})
            for gw in list(player_gw_xpts.keys()):
                fixture_count = team_counts.get(gw, 1)
                if fixture_count == 0:
                    player_gw_xpts[gw] = 0.0

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

    # Determine the NEXT gameweek to plan for (transfers are for the future)
    # Priority: is_next (upcoming), or if is_current is not finished yet use that,
    # otherwise find the first unfinished GW
    next_gw = next((e for e in events if e.get("is_next")), None)
    current_gw_obj = next((e for e in events if e.get("is_current")), None)

    if next_gw:
        planning_gw = next_gw
    elif current_gw_obj and not current_gw_obj.get("finished"):
        planning_gw = current_gw_obj
    else:
        # All current GWs finished, find first unfinished
        unfinished = [e for e in events if not e.get("finished")]
        planning_gw = unfinished[0] if unfinished else (events[-1] if events else None)

    # For display purposes, current_gw is the latest active/completed
    current_gw = current_gw_obj or planning_gw

    # Planning GW ID — this is what we use for fixture windows and xPts
    planning_gw_id = planning_gw["id"] if planning_gw else 1
    gw_id = planning_gw_id  # all fixture lookups use this

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

    # Load form-weighted xG data from recent GWs
    player_gw_data = load_recent_gw_live_data(gw_id, n_recent=7)
    form_xg_data = compute_form_weighted_xg(player_gw_data, n_recent=7)

    # Detect blank/double gameweeks
    team_fixture_counts = detect_blank_double_gws(fixtures, gw_id, n_gws=6, teams=teams)

    # Build xPts model (uses planning_gw_id, so only future fixtures)
    xpts_map = build_xpts_model(df, team_odds, teams, fixtures, gw_id,
                                 form_xg_data=form_xg_data,
                                 team_fixture_counts=team_fixture_counts)

    # Add xPts columns
    # xpts_next_gw = xPts for the specific next gameweek (planning_gw_id)
    df["xpts_next_gw"] = df["id"].map(
        lambda pid: xpts_map.get(pid, {}).get(planning_gw_id, 0)
    )
    # xpts_total = sum of xPts over next 6 GWs (all from planning_gw_id onwards)
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

    return df, teams, current_gw, planning_gw_id, upcoming, fixtures, xpts_map, team_fixture_counts


# ============================================================
# MANAGER TEAM FETCHER
# ============================================================

@st.cache_data(ttl=1800)
def fetch_manager_team(manager_id, current_gw_id):
    """
    Fetch a manager's current squad, bank, free transfers,
    and purchase prices (for correct selling price calculation).
    """
    try:
        headers = {"User-Agent": "FPL-Optimizer/2.0"}

        # 1. Basic manager info
        entry = requests.get(
            f"{FPL_BASE}/entry/{manager_id}/",
            headers=headers, timeout=15,
        ).json()

        manager_name = f"{entry.get('player_first_name', '')} {entry.get('player_last_name', '')}"
        team_name = entry.get("name", "Unknown")
        overall_rank = entry.get("summary_overall_rank") or "-"
        total_points = entry.get("summary_overall_points", 0)

        # 2. History endpoint — gives bank and transfer count per GW
        history = requests.get(
            f"{FPL_BASE}/entry/{manager_id}/history/",
            headers=headers, timeout=15,
        ).json()

        current_hist = history.get("current", [])
        bank = 0
        free_transfers = 1  # default

        if current_hist:
            latest_gw = current_hist[-1]
            bank = latest_gw.get("bank", 0)

            # Calculate free transfers available for NEXT gameweek
            # Logic: start with 1 FT per GW, can bank up to max 5
            # We look at the last few GWs to count how many were banked
            ft = 1  # everyone gets 1 at start of season
            for gw_data in current_hist:
                transfers_made = gw_data.get("event_transfers", 0)
                transfers_cost = gw_data.get("event_transfers_cost", 0)

                if transfers_cost > 0:
                    # They took hits: they used all FTs + some extra
                    ft = 1  # reset to 1 for next GW (used everything + more)
                elif transfers_made == 0:
                    # Banked a FT
                    ft = min(ft + 1, 5)
                elif transfers_made <= ft:
                    # Used some/all FTs without a hit
                    ft = max(1, ft - transfers_made + 1)  # +1 for the new GW's FT
                else:
                    ft = 1

            free_transfers = ft

        # 3. Transfer history — gives purchase prices (element_in_cost)
        transfers = requests.get(
            f"{FPL_BASE}/entry/{manager_id}/transfers/",
            headers=headers, timeout=15,
        ).json()

        # Build purchase price map: player_id -> purchase_price (in 0.1m units)
        # Latest transfer for each player is their purchase price
        purchase_prices = {}
        for t in transfers:
            purchase_prices[t["element_in"]] = t["element_in_cost"]

        # 4. Current picks
        picks_data = None
        for gw in [current_gw_id, current_gw_id - 1]:
            if gw < 1:
                continue
            try:
                resp = requests.get(
                    f"{FPL_BASE}/entry/{manager_id}/event/{gw}/picks/",
                    headers=headers, timeout=15,
                )
                if resp.status_code == 200:
                    picks_data = resp.json()
                    break
            except Exception:
                continue

        if picks_data is None:
            return None, "Could not fetch team picks"

        picks = picks_data.get("picks", [])
        active_chip = picks_data.get("active_chip")

        # entry_history within picks gives exact FT info in some API versions
        entry_hist = picks_data.get("entry_history", {})
        if entry_hist:
            bank = entry_hist.get("bank", bank)

        squad_ids = [p["element"] for p in picks]
        captains = {p["element"]: p.get("is_captain", False) for p in picks}
        vice_captains = {p["element"]: p.get("is_vice_captain", False) for p in picks}
        positions_in_team = {p["element"]: p.get("position", 0) for p in picks}

        # picks may contain selling_price in newer API
        selling_prices_api = {}
        for p in picks:
            if "selling_price" in p:
                selling_prices_api[p["element"]] = p["selling_price"]

        return {
            "manager_name": manager_name.strip(),
            "team_name": team_name,
            "overall_rank": overall_rank,
            "total_points": total_points,
            "bank": bank,
            "free_transfers": free_transfers,
            "squad_ids": squad_ids,
            "captains": captains,
            "vice_captains": vice_captains,
            "positions": positions_in_team,
            "active_chip": active_chip,
            "purchase_prices": purchase_prices,
            "selling_prices_api": selling_prices_api,
        }, None

    except requests.exceptions.HTTPError:
        return None, "Manager ID not found"
    except Exception as e:
        return None, str(e)


def calculate_selling_price(player_id, current_price, purchase_prices, selling_prices_api):
    """
    Calculate the correct FPL selling price.
    Rule: you get 50% of profit (rounded down).
    selling_price = purchase_price + floor((current_price - purchase_price) / 2)
    If current_price < purchase_price, selling_price = current_price (full loss).
    """
    # If the API gave us the selling price directly, use it
    if player_id in selling_prices_api:
        return selling_prices_api[player_id]

    purchase = purchase_prices.get(player_id, current_price)
    if current_price <= purchase:
        return current_price  # no profit or a loss — sell at current
    profit = current_price - purchase
    return purchase + (profit // 2)  # 50% of profit, rounded down


def find_optimal_transfers(squad_df, all_players_df, bank, free_transfers,
                           purchase_prices, selling_prices_api,
                           n_transfers=1, xpts_col="xpts_total",
                           hit_cost=4):
    """
    Find the best transfers using correct sale/buy prices and hit-aware logic.

    For each candidate transfer:
    - OUT player sold at their SELLING price (50% profit rule)
    - IN player bought at current market price (now_cost)
    - If n_transfers > free_transfers, apply -4 per extra transfer
    - Only suggest if net xPts gain > hit penalty
    """
    if squad_df is None or len(squad_df) == 0:
        return []

    squad_ids = set(squad_df["id"].tolist())

    # Available players not in squad
    available = all_players_df[
        (~all_players_df["id"].isin(squad_ids)) &
        (all_players_df["minutes"] > 45) &
        (all_players_df["status"].isin(["a", "d", ""])) &
        (all_players_df[xpts_col] > 0)
    ].copy()

    # Calculate selling price for each squad player
    squad_df = squad_df.copy()
    squad_df["sell_price"] = squad_df.apply(
        lambda r: calculate_selling_price(
            r["id"], r["now_cost"], purchase_prices, selling_prices_api
        ), axis=1
    )

    # Hit penalty: transfers beyond free ones cost 4 pts each
    extra_transfers = max(0, n_transfers - free_transfers)
    total_hit = extra_transfers * hit_cost

    suggestions = []

    if n_transfers == 1:
        for _, out_p in squad_df.iterrows():
            # Budget available = bank + selling price of outgoing player
            budget_available = bank + out_p["sell_price"]
            remaining_squad = squad_df[squad_df["id"] != out_p["id"]]
            team_counts = remaining_squad["team_id"].value_counts().to_dict()

            cands = available[
                (available["pos_id"] == out_p["pos_id"]) &
                (available["now_cost"] <= budget_available)
            ].copy()
            cands = cands[cands["team_id"].map(lambda tid: team_counts.get(tid, 0) < 3)]

            if len(cands) == 0:
                continue

            best_in = cands.loc[cands[xpts_col].idxmax()]
            xpts_gain = best_in[xpts_col] - out_p[xpts_col]
            net_gain = xpts_gain - total_hit

            if net_gain > 0.5:  # only suggest if meaningfully better
                suggestions.append({
                    "out": [out_p.to_dict()],
                    "in": [best_in.to_dict()],
                    "xpts_gain": round(xpts_gain, 1),
                    "net_gain": round(net_gain, 1),
                    "hit": total_hit,
                    "cost_change": round((best_in["now_cost"] - out_p["sell_price"]) / 10, 1),
                    "budget_after": round((budget_available - best_in["now_cost"]) / 10, 1),
                })

    elif n_transfers >= 2:
        squad_list = squad_df.to_dict("records")
        seen = set()
        for i, out1 in enumerate(squad_list):
            for j, out2 in enumerate(squad_list):
                if j <= i:
                    continue
                pair_key = (min(out1["id"], out2["id"]), max(out1["id"], out2["id"]))
                if pair_key in seen:
                    continue
                seen.add(pair_key)

                freed = bank + out1["sell_price"] + out2["sell_price"]
                remaining = squad_df[~squad_df["id"].isin([out1["id"], out2["id"]])]
                tc = remaining["team_id"].value_counts().to_dict()

                best_pair_xpts = 0
                best_pair = None

                cands1 = available[
                    (available["pos_id"] == out1["pos_id"]) &
                    (available["now_cost"] <= freed)
                ]
                cands1 = cands1[cands1["team_id"].map(lambda t: tc.get(t, 0) < 3)]

                for _, in1 in cands1.nlargest(5, xpts_col).iterrows():
                    remaining_budget = freed - in1["now_cost"]
                    tc2 = tc.copy()
                    tc2[in1["team_id"]] = tc2.get(in1["team_id"], 0) + 1

                    cands2 = available[
                        (available["pos_id"] == out2["pos_id"]) &
                        (available["now_cost"] <= remaining_budget) &
                        (available["id"] != in1["id"])
                    ]
                    cands2 = cands2[cands2["team_id"].map(lambda t: tc2.get(t, 0) < 3)]

                    if len(cands2) == 0:
                        continue

                    in2 = cands2.loc[cands2[xpts_col].idxmax()]
                    pair_xpts = in1[xpts_col] + in2[xpts_col]

                    if pair_xpts > best_pair_xpts:
                        best_pair_xpts = pair_xpts
                        best_pair = (in1, in2)

                if best_pair:
                    old_xpts = out1[xpts_col] + out2[xpts_col]
                    xpts_gain = best_pair_xpts - old_xpts
                    net_gain = xpts_gain - total_hit

                    if net_gain > 0.5:
                        total_in_cost = best_pair[0]["now_cost"] + best_pair[1]["now_cost"]
                        suggestions.append({
                            "out": [out1, out2],
                            "in": [best_pair[0].to_dict(), best_pair[1].to_dict()],
                            "xpts_gain": round(xpts_gain, 1),
                            "net_gain": round(net_gain, 1),
                            "hit": total_hit,
                            "cost_change": round((total_in_cost - out1["sell_price"] - out2["sell_price"]) / 10, 1),
                            "budget_after": round((freed - total_in_cost) / 10, 1),
                        })

    suggestions.sort(key=lambda x: x["net_gain"], reverse=True)
    return suggestions[:10]




def solve_best_xi_for_gw(squad_df, xpts_map, gw_id):
    """Pick best starting XI from 15-man squad for a specific gameweek."""
    if squad_df is None or len(squad_df) < 11:
        return None, None

    # Build per-GW xPts column
    sq = squad_df.copy()
    sq["xpts_gw"] = sq["id"].map(lambda pid: xpts_map.get(pid, {}).get(gw_id, 0))
    sq["xpts_gw"] = sq["xpts_gw"].fillna(0)

    prob = LpProblem(f"FPL_XI_GW{gw_id}", LpMaximize)
    sq = sq.reset_index(drop=True)
    pids = sq["id"].tolist()
    pid_to_idx = {row["id"]: i for i, row in sq.iterrows()}
    xpts_vals = sq["xpts_gw"].tolist()
    pos_list = sq["pos_id"].tolist()
    x = {pid: LpVariable(f"xi_{gw_id}_{pid}", cat="Binary") for pid in pids}

    prob += lpSum(x[pid] * xpts_vals[pid_to_idx[pid]] for pid in pids)
    prob += lpSum(x[pid] for pid in pids) == 11

    gk_pids = [pid for pid in pids if pos_list[pid_to_idx[pid]] == 1]
    prob += lpSum(x[pid] for pid in gk_pids) == 1
    def_pids = [pid for pid in pids if pos_list[pid_to_idx[pid]] == 2]
    prob += lpSum(x[pid] for pid in def_pids) >= 3
    prob += lpSum(x[pid] for pid in def_pids) <= 5
    mid_pids = [pid for pid in pids if pos_list[pid_to_idx[pid]] == 3]
    prob += lpSum(x[pid] for pid in mid_pids) >= 2
    prob += lpSum(x[pid] for pid in mid_pids) <= 5
    fwd_pids = [pid for pid in pids if pos_list[pid_to_idx[pid]] == 4]
    prob += lpSum(x[pid] for pid in fwd_pids) >= 1
    prob += lpSum(x[pid] for pid in fwd_pids) <= 3

    try:
        prob.solve(PULP_CBC_CMD(msg=0, timeLimit=10))
    except Exception:
        return None, None

    if LpStatus[prob.status] != "Optimal":
        return None, None

    xi_ids = [pid for pid in pids if value(x[pid]) is not None and value(x[pid]) > 0.5]
    xi = sq[sq["id"].isin(xi_ids)].copy()
    bench = sq[~sq["id"].isin(xi_ids)].copy()
    return xi, bench


def find_best_single_transfer_for_gw(squad_df, all_players_df, bank,
                                      purchase_prices, selling_prices_api,
                                      xpts_map, gw_id, exclude_ids=None,
                                      horizon_end=None):
    """
    Find the single best transfer considering the REMAINING HORIZON.
    
    For a GW31 transfer with horizon ending at GW35:
    - Compare players on sum(xPts from GW31 to GW35), not just GW31
    - This ensures transfers are forward-looking, not myopic
    
    Also returns the single-GW gain for display purposes.
    """
    if squad_df is None or len(squad_df) == 0:
        return None

    if exclude_ids is None:
        exclude_ids = set()
    
    # Default horizon: 6 GWs from this GW
    if horizon_end is None:
        horizon_end = gw_id + 6
    
    horizon_gws = list(range(gw_id, horizon_end))

    squad_ids = set(squad_df["id"].tolist()) | exclude_ids
    available = all_players_df[
        (~all_players_df["id"].isin(squad_ids)) &
        (all_players_df["minutes"] > 45) &
        (all_players_df["status"].isin(["a", "d", ""]))
    ].copy()

    # Remaining horizon xPts (what matters for the transfer decision)
    squad_df = squad_df.copy()
    squad_df["xpts_horizon"] = squad_df["id"].map(
        lambda pid: sum(xpts_map.get(pid, {}).get(gw, 0) for gw in horizon_gws)
    )
    available["xpts_horizon"] = available["id"].map(
        lambda pid: sum(xpts_map.get(pid, {}).get(gw, 0) for gw in horizon_gws)
    )
    
    # Also get single-GW xPts for display
    squad_df["xpts_gw"] = squad_df["id"].map(lambda pid: xpts_map.get(pid, {}).get(gw_id, 0))
    available["xpts_gw"] = available["id"].map(lambda pid: xpts_map.get(pid, {}).get(gw_id, 0))

    squad_df["sell_price"] = squad_df.apply(
        lambda r: calculate_selling_price(r["id"], r["now_cost"], purchase_prices, selling_prices_api),
        axis=1,
    )

    best = None
    best_gain = -999

    for _, out_p in squad_df.iterrows():
        budget_avail = bank + out_p["sell_price"]
        remaining = squad_df[squad_df["id"] != out_p["id"]]
        tc = remaining["team_id"].value_counts().to_dict()

        cands = available[
            (available["pos_id"] == out_p["pos_id"]) &
            (available["now_cost"] <= budget_avail)
        ]
        cands = cands[cands["team_id"].map(lambda tid: tc.get(tid, 0) < 3)]

        if len(cands) == 0:
            continue

        # Pick best by HORIZON xPts, not single GW
        top = cands.loc[cands["xpts_horizon"].idxmax()]
        horizon_gain = top["xpts_horizon"] - out_p["xpts_horizon"]
        gw_gain = top["xpts_gw"] - out_p["xpts_gw"]
        
        if horizon_gain > best_gain and horizon_gain > 0.05:
            best_gain = horizon_gain
            best = {
                "out": out_p.to_dict(),
                "in": top.to_dict(),
                "xpts_gain": round(horizon_gain, 2),  # horizon gain for decision-making
                "xpts_gw_gain": round(gw_gain, 2),     # single GW gain for display
                "new_bank": int(budget_avail - top["now_cost"]),
            }

    return best


def solve_free_hit_squad(all_players_df, xpts_map, gw_id, budget=1000):
    """Free Hit: pick best possible 15-man squad for a single GW."""
    eligible = all_players_df[
        (all_players_df["minutes"] > 45) &
        (all_players_df["status"].isin(["a", "d", ""]))
    ].copy()
    eligible["xpts_gw"] = eligible["id"].map(lambda pid: xpts_map.get(pid, {}).get(gw_id, 0))
    # For Free Hit, only pick players who actually have a fixture this GW
    eligible = eligible[eligible["xpts_gw"] > 0].copy()
    eligible["xpts_gw"] = eligible["xpts_gw"].fillna(0)
    eligible["now_cost"] = eligible["now_cost"].fillna(0).astype(int)
    if len(eligible) < 15:
        return None

    eligible = eligible.reset_index(drop=True)
    pid_map = {row["id"]: i for i, row in eligible.iterrows()}
    xv = eligible["xpts_gw"].tolist()
    cv = eligible["now_cost"].tolist()
    pv = eligible["pos_id"].tolist()
    tv = eligible["team_id"].tolist()
    pids = eligible["id"].tolist()

    prob = LpProblem(f"FH_GW{gw_id}", LpMaximize)
    x = {pid: LpVariable(f"fh_{gw_id}_{pid}", cat="Binary") for pid in pids}
    prob += lpSum(x[pid] * xv[pid_map[pid]] for pid in pids)
    prob += lpSum(x[pid] * cv[pid_map[pid]] for pid in pids) <= budget
    prob += lpSum(x[pid] for pid in pids) == 15
    for pos_id, cnt in [(1, 2), (2, 5), (3, 5), (4, 3)]:
        prob += lpSum(x[pid] for pid in pids if pv[pid_map[pid]] == pos_id) == cnt
    for tid in set(tv):
        prob += lpSum(x[pid] for pid in pids if tv[pid_map[pid]] == tid) <= 3
    try:
        prob.solve(PULP_CBC_CMD(msg=0, timeLimit=30))
    except Exception:
        return None
    if LpStatus[prob.status] != "Optimal":
        return None
    sel = [pid for pid in pids if value(x[pid]) is not None and value(x[pid]) > 0.5]
    return eligible[eligible["id"].isin(sel)].copy()


def solve_wildcard_squad(all_players_df, xpts_map, planning_gw, n_future, budget=1000,
                         team_fixture_counts=None):
    """
    Wildcard: best 15-man squad optimised for total xPts over remaining GWs,
    WITH blank-awareness — ensures at least 11 players have a fixture in every GW.

    This prevents the solver from picking 15 players from teams that blank in GW31
    just because they have great fixtures in GWs 30, 32-35.
    """
    gw_range = list(range(planning_gw, planning_gw + n_future))

    eligible = all_players_df[
        (all_players_df["minutes"] > 45) &
        (all_players_df["status"].isin(["a", "d", ""]))
    ].copy()

    # Calculate total xPts across horizon
    eligible["xpts_rem"] = eligible["id"].map(
        lambda pid: sum(xpts_map.get(pid, {}).get(gw, 0) for gw in gw_range)
    )
    eligible = eligible[eligible["xpts_rem"] > 0].copy()
    eligible["xpts_rem"] = eligible["xpts_rem"].fillna(0)
    eligible["now_cost"] = eligible["now_cost"].fillna(0).astype(int)
    if len(eligible) < 15:
        return None

    # Pre-compute which players have a fixture in each GW
    # (a player has a fixture if their team has >= 1 fixture that GW)
    player_has_fixture = {}  # {pid: {gw: bool}}
    for _, p in eligible.iterrows():
        pid = p["id"]
        tid = p["team_id"]
        player_has_fixture[pid] = {}
        for gw in gw_range:
            if team_fixture_counts:
                fc = team_fixture_counts.get(tid, {}).get(gw, 1)
                player_has_fixture[pid][gw] = (fc > 0)
            else:
                # No fixture count data — check if xPts > 0 as proxy
                player_has_fixture[pid][gw] = (xpts_map.get(pid, {}).get(gw, 0) > 0)

    eligible = eligible.reset_index(drop=True)
    pid_map = {row["id"]: i for i, row in eligible.iterrows()}
    xv = eligible["xpts_rem"].tolist()
    cv = eligible["now_cost"].tolist()
    pv = eligible["pos_id"].tolist()
    tv = eligible["team_id"].tolist()
    pids = eligible["id"].tolist()

    prob = LpProblem("Wildcard", LpMaximize)
    x = {pid: LpVariable(f"wc_{pid}", cat="Binary") for pid in pids}

    # Objective: maximise total xPts
    prob += lpSum(x[pid] * xv[pid_map[pid]] for pid in pids)

    # Budget
    prob += lpSum(x[pid] * cv[pid_map[pid]] for pid in pids) <= budget

    # Squad = 15
    prob += lpSum(x[pid] for pid in pids) == 15

    # Position constraints
    for pos_id, cnt in [(1, 2), (2, 5), (3, 5), (4, 3)]:
        prob += lpSum(x[pid] for pid in pids if pv[pid_map[pid]] == pos_id) == cnt

    # Max 3 per team
    for tid in set(tv):
        prob += lpSum(x[pid] for pid in pids if tv[pid_map[pid]] == tid) <= 3

    # === BLANK GW CONSTRAINT ===
    # For each GW in the horizon, at least 11 selected players must have a fixture.
    # This prevents the solver from loading up on blanking teams.
    for gw in gw_range:
        playing_pids = [pid for pid in pids if player_has_fixture.get(pid, {}).get(gw, True)]
        if playing_pids:
            prob += lpSum(x[pid] for pid in playing_pids) >= 11

    try:
        prob.solve(PULP_CBC_CMD(msg=0, timeLimit=45))
    except Exception:
        return None
    if LpStatus[prob.status] != "Optimal":
        # If infeasible with 11-player constraint (e.g. massive blank), try 8
        prob2 = LpProblem("Wildcard_relaxed", LpMaximize)
        x2 = {pid: LpVariable(f"wc2_{pid}", cat="Binary") for pid in pids}
        prob2 += lpSum(x2[pid] * xv[pid_map[pid]] for pid in pids)
        prob2 += lpSum(x2[pid] * cv[pid_map[pid]] for pid in pids) <= budget
        prob2 += lpSum(x2[pid] for pid in pids) == 15
        for pos_id, cnt in [(1, 2), (2, 5), (3, 5), (4, 3)]:
            prob2 += lpSum(x2[pid] for pid in pids if pv[pid_map[pid]] == pos_id) == cnt
        for tid in set(tv):
            prob2 += lpSum(x2[pid] for pid in pids if tv[pid_map[pid]] == tid) <= 3
        for gw in gw_range:
            playing_pids = [pid for pid in pids if player_has_fixture.get(pid, {}).get(gw, True)]
            if playing_pids:
                prob2 += lpSum(x2[pid] for pid in playing_pids) >= 8
        try:
            prob2.solve(PULP_CBC_CMD(msg=0, timeLimit=45))
        except Exception:
            return None
        if LpStatus[prob2.status] != "Optimal":
            return None
        sel = [pid for pid in pids if value(x2[pid]) is not None and value(x2[pid]) > 0.5]
        return eligible[eligible["id"].isin(sel)].copy()

    sel = [pid for pid in pids if value(x[pid]) is not None and value(x[pid]) > 0.5]
    return eligible[eligible["id"].isin(sel)].copy()


def find_best_captain(squad_df, xpts_map, gw_id):
    """Find best captain (highest xPts) for a specific GW."""
    if squad_df is None or len(squad_df) == 0:
        return None
    sq = squad_df.copy()
    sq["xpts_gw"] = sq["id"].map(lambda pid: xpts_map.get(pid, {}).get(gw_id, 0))
    return sq.loc[sq["xpts_gw"].idxmax()]


def build_rolling_plan(my_squad_df, all_players_df, bank, free_transfers,
                       purchase_prices, selling_prices_api, xpts_map,
                       planning_gw_id, n_gws=6, chip_schedule=None,
                       team_fixture_counts=None):
    """
    Chip-aware rolling planner.
    chip_schedule: {gw_id: chip_name} e.g. {31: "wildcard", 33: "bench_boost"}
    Chips: "wildcard", "free_hit", "triple_captain", "bench_boost"
    """
    if chip_schedule is None:
        chip_schedule = {}

    plan = []
    current_squad = my_squad_df.copy()
    current_bank = bank
    current_ft = free_transfers
    current_purchase = purchase_prices.copy()
    current_selling = selling_prices_api.copy()
    pre_fh_squad = None

    for i in range(n_gws):
        gw = planning_gw_id + i
        chip = chip_schedule.get(gw, None)

        has_fixtures = any(xpts_map.get(pid, {}).get(gw, 0) > 0 for pid in current_squad["id"])
        if not has_fixtures:
            break

        gw_entry = {
            "gw": gw, "chip": chip, "transfer": None, "hit": 0,
            "squad": current_squad.copy(), "xi": None, "bench": None,
            "captain": None, "captain_multiplier": 2, "bench_boost": False,
        }

        # === FREE HIT ===
        if chip == "free_hit":
            pre_fh_squad = current_squad.copy()
            total_val = int(current_bank + current_squad["now_cost"].sum())
            fh_squad = solve_free_hit_squad(all_players_df, xpts_map, gw, total_val)
            if fh_squad is not None:
                gw_entry["squad"] = fh_squad
                xi, bench = solve_best_xi_for_gw(fh_squad, xpts_map, gw)
            else:
                xi, bench = solve_best_xi_for_gw(current_squad, xpts_map, gw)
            gw_entry["xi"] = xi
            gw_entry["bench"] = bench
            gw_entry["captain"] = find_best_captain(xi, xpts_map, gw) if xi is not None else None
            current_ft = min(current_ft + 1, 5)
            plan.append(gw_entry)
            continue

        # Restore after free hit
        if pre_fh_squad is not None:
            current_squad = pre_fh_squad
            pre_fh_squad = None

        # === WILDCARD ===
        if chip == "wildcard":
            total_val = int(current_bank + current_squad["now_cost"].sum())
            wc_squad = solve_wildcard_squad(all_players_df, xpts_map, gw, n_gws - i, total_val,
                                               team_fixture_counts=team_fixture_counts)
            if wc_squad is not None:
                current_squad = wc_squad
                gw_entry["squad"] = wc_squad
                for _, p in wc_squad.iterrows():
                    current_purchase[p["id"]] = p["now_cost"]
                current_bank = total_val - wc_squad["now_cost"].sum()
            current_ft = 1
            xi, bench = solve_best_xi_for_gw(current_squad, xpts_map, gw)
            gw_entry["xi"] = xi
            gw_entry["bench"] = bench
            gw_entry["captain"] = find_best_captain(xi, xpts_map, gw) if xi is not None else None
            plan.append(gw_entry)
            continue

        # === NORMAL / TC / BB ===
        # Use up to all available FTs, plus hits if gain > 4pts
        transfers_made = []
        transfers_ft_used = 0
        total_hit = 0
        recently_sold = set()

        # The horizon shrinks as we move through GWs:
        # GW30 transfer considers GW30-35, GW31 considers GW31-35, etc.
        horizon_end = planning_gw_id + n_gws

        # Keep finding improving transfers until no more gains
        max_transfers = min(current_ft + 3, 8)
        for t_num in range(max_transfers):
            transfer = find_best_single_transfer_for_gw(
                current_squad, all_players_df, current_bank,
                current_purchase, current_selling, xpts_map, gw,
                exclude_ids=recently_sold,
                horizon_end=horizon_end,
            )

            if transfer is None:
                break  # no improving transfer found

            # Is this a free transfer or a hit?
            is_free = (t_num < current_ft)
            hit_cost = 0 if is_free else 4

            # Decision thresholds:
            # Free transfers: make if ANY positive gain (even 0.1 xPts — it's free!)
            # Hits: only make if gain clearly exceeds 4pt cost
            if is_free:
                if transfer["xpts_gain"] < 0.05:
                    break  # negligible gain, not worth the effort
            else:
                net_gain = transfer["xpts_gain"] - hit_cost
                if net_gain < 0.5:
                    break  # hit not worth it
                total_hit += hit_cost

            # Apply transfer
            transfers_made.append(transfer)
            out_id = transfer["out"]["id"]
            in_id = transfer["in"]["id"]
            recently_sold.add(out_id)  # don't suggest buying back
            current_squad = current_squad[current_squad["id"] != out_id]
            in_player = all_players_df[all_players_df["id"] == in_id]
            if len(in_player) > 0:
                current_squad = pd.concat([current_squad, in_player.iloc[:1]], ignore_index=True)
            current_bank = transfer["new_bank"]
            current_purchase[in_id] = transfer["in"]["now_cost"]
            if out_id in current_selling:
                del current_selling[out_id]
            if is_free:
                transfers_ft_used += 1

        # Store all transfers for this GW
        gw_entry["transfers"] = transfers_made
        gw_entry["hit"] = total_hit
        gw_entry["ft_used"] = transfers_ft_used

        # FT accounting: spent some FTs, then gain 1 for next GW
        remaining_ft = current_ft - transfers_ft_used
        current_ft = min(remaining_ft + 1, 5)

        xi, bench = solve_best_xi_for_gw(current_squad, xpts_map, gw)
        gw_entry["xi"] = xi
        gw_entry["bench"] = bench
        gw_entry["squad"] = current_squad.copy()
        gw_entry["captain"] = find_best_captain(xi, xpts_map, gw) if xi is not None else None

        if chip == "triple_captain":
            gw_entry["captain_multiplier"] = 3
        if chip == "bench_boost":
            gw_entry["bench_boost"] = True

        plan.append(gw_entry)

    return plan


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
    # Header with logo
    st.markdown(
        f'<div style="display:flex; align-items:center; gap:1rem; margin-bottom:0.5rem;">'
        f'<img src="data:image/png;base64,{DATUMLY_LOGO_B64}" style="height:120px;" />'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-title">'
        'Data-driven FPL intelligence · xG · xA · CS% · xPts · FDR'
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
        df, teams, current_gw, planning_gw_id, upcoming_map, fixtures_list, xpts_map, team_fixture_counts = enrich_data(
            bootstrap, fixtures_raw, team_odds
        )

    # === GW Info ===
    if current_gw:
        deadline = datetime.fromisoformat(current_gw["deadline_time"].replace("Z", "+00:00"))
        status = "Completed" if current_gw.get("finished") else ("In Progress" if current_gw.get("is_current") else "Upcoming")
        bc = "badge-green" if status == "Completed" else ("badge-yellow" if status == "In Progress" else "badge-blue")
        planning_str = f"Planning for GW{planning_gw_id}" if planning_gw_id != current_gw.get("id") else ""
        st.markdown(f"""<div class="gw-bar">
            <span class="gw-num">Gameweek {current_gw['id']}</span>
            <span class="gw-deadline">Deadline: {deadline.strftime('%a %d %b, %H:%M')}</span>
            <span class="badge {bc}">{status}</span>
            {f'<span class="badge badge-blue">{planning_str}</span>' if planning_str else ''}
            <span style="color:#5a6580; font-size:0.7rem;">Odds: {odds_status} · {len(team_odds)} teams matched</span>
        </div>""", unsafe_allow_html=True)

    # === Tabs ===
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏠 My Team", "📊 Dashboard", "👥 Player Projections",
        "⭐ Optimal Squad (MILP)", "🔄 Transfer Planner", "📅 Fixtures"
    ])

    active = df[df["minutes"] > 0].copy()
    qualified = df[df["minutes"] > 45].copy()

    # ==================== MY TEAM ====================
    with tab1:
        st.markdown(
            '<div class="section-header">🏠 My Team — Enter Your FPL ID</div>',
            unsafe_allow_html=True,
        )

        # FPL ID input
        col_id, col_btn = st.columns([3, 1])
        with col_id:
            fpl_id = st.text_input(
                "FPL Team ID",
                value=st.session_state.get("fpl_id", ""),
                placeholder="e.g. 123456",
                help="Find your ID in the URL when you view your team on the FPL website",
                key="fpl_id_input",
            )
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            load_team = st.button("Load My Team", use_container_width=True)

        if load_team and fpl_id:
            st.session_state["fpl_id"] = fpl_id

        if fpl_id and fpl_id.strip().isdigit():
            manager_id = int(fpl_id.strip())
            gw_id = current_gw["id"] if current_gw else 1

            with st.spinner("Fetching your team..."):
                team_data, team_err = fetch_manager_team(manager_id, gw_id)

            if team_err:
                st.error(f"Could not load team: {team_err}")
                st.info("Make sure your FPL ID is correct. You can find it in the URL when viewing your team on the FPL website.")
            elif team_data:
                # Manager info header
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">Manager</div>
                        <div class="metric-value" style="font-size:1.1rem;">{team_data['team_name']}</div>
                        <div class="metric-sub">{team_data['manager_name']}</div>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">Overall Rank</div>
                        <div class="metric-value">{team_data['overall_rank']:,}</div>
                        <div class="metric-sub">{team_data['total_points']} pts</div>
                    </div>""", unsafe_allow_html=True)
                with c3:
                    bank_display = team_data['bank'] / 10
                    ft_display = team_data.get('free_transfers', 1)
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">Bank / Free Transfers</div>
                        <div class="metric-value">£{bank_display:.1f}m</div>
                        <div class="metric-sub">~{ft_display} FT available</div>
                    </div>""", unsafe_allow_html=True)
                with c4:
                    squad_xpts = 0
                    my_squad = df[df["id"].isin(team_data["squad_ids"])].copy()
                    if len(my_squad) > 0:
                        squad_xpts = my_squad["xpts_total"].sum()
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">Squad xPts (6GW)</div>
                        <div class="metric-value">{squad_xpts:.1f}</div>
                        <div class="metric-sub">{len(my_squad)} players loaded</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("")

                if len(my_squad) > 0:
                    # Mark starters vs bench
                    my_squad["is_starter"] = my_squad["id"].map(
                        lambda pid: team_data["positions"].get(pid, 99) <= 11
                    )
                    my_squad["is_captain"] = my_squad["id"].map(
                        lambda pid: team_data["captains"].get(pid, False)
                    )
                    my_squad["is_vice"] = my_squad["id"].map(
                        lambda pid: team_data["vice_captains"].get(pid, False)
                    )

                    # Show current squad
                    st.subheader("Your Current Squad")
                    starters = my_squad[my_squad["is_starter"]].sort_values(
                        ["pos_id", "xpts_total"], ascending=[True, False]
                    )
                    bench = my_squad[~my_squad["is_starter"]].sort_values("pos_id")

                    # Pitch view of starters
                    for pid_val, plabel in [(4, "Forwards"), (3, "Midfielders"), (2, "Defenders"), (1, "Goalkeeper")]:
                        pp = starters[starters["pos_id"] == pid_val]
                        if len(pp) > 0:
                            st.markdown(f"<div class='pitch-row-label'>{plabel}</div>", unsafe_allow_html=True)
                            cols = st.columns(max(len(pp), 1))
                            for i, (_, p) in enumerate(pp.iterrows()):
                                sc = f"pitch-shirt-{p['pos'].lower()}"
                                cap_badge = " (C)" if p["is_captain"] else (" (V)" if p["is_vice"] else "")
                                with cols[i]:
                                    st.markdown(f"""<div style="text-align:center;">
                                        <div class="pitch-shirt {sc}">{p['xpts_next_gw']:.1f}</div>
                                        <div class="pitch-name">{p['name']}{cap_badge}</div>
                                        <div class="pitch-price">£{p['price']:.1f}m · {p['xpts_total']:.1f} xPts</div>
                                    </div>""", unsafe_allow_html=True)

                    if len(bench) > 0:
                        st.markdown("**Bench**")
                        bcols = st.columns(max(len(bench), 1))
                        for i, (_, p) in enumerate(bench.iterrows()):
                            with bcols[i]:
                                st.markdown(f"""<div style="text-align:center;opacity:0.6;">
                                    <div class="pitch-name">{p['name']}</div>
                                    <div class="pitch-price">{p['pos']} · £{p['price']:.1f}m · {p['xpts_total']:.1f} xPts</div>
                                </div>""", unsafe_allow_html=True)

                    st.markdown("")

                    # === CHIP & TRANSFER PLANNER ===
                    st.markdown(
                        '<div class="section-header">🗓️ Gameweek-by-Gameweek Planner '
                        '<span class="source-tag src-model">Rolling Planner</span></div>',
                        unsafe_allow_html=True,
                    )

                    ft_available = team_data.get("free_transfers", 1)
                    st.caption(f"You have **{ft_available} free transfer(s)** available.")

                    # Step 1: Chip selection
                    st.markdown("**Step 1: Set your chip schedule**")
                    gw_options = [planning_gw_id + i for i in range(6)]
                    chip_cols = st.columns(4)
                    with chip_cols[0]:
                        wc_gw = st.selectbox("🃏 Wildcard", ["None"] + gw_options, key="wc_gw")
                    with chip_cols[1]:
                        fh_gw = st.selectbox("⚡ Free Hit", ["None"] + gw_options, key="fh_gw")
                    with chip_cols[2]:
                        tc_gw = st.selectbox("👑 Triple Captain", ["None"] + gw_options, key="tc_gw")
                    with chip_cols[3]:
                        bb_gw = st.selectbox("💪 Bench Boost", ["None"] + gw_options, key="bb_gw")

                    chip_schedule = {}
                    if wc_gw != "None":
                        chip_schedule[int(wc_gw)] = "wildcard"
                    if fh_gw != "None":
                        chip_schedule[int(fh_gw)] = "free_hit"
                    if tc_gw != "None":
                        chip_schedule[int(tc_gw)] = "triple_captain"
                    if bb_gw != "None":
                        chip_schedule[int(bb_gw)] = "bench_boost"

                    # Validate
                    chip_gws_used = [g for g in [wc_gw, fh_gw, tc_gw, bb_gw] if g != "None"]
                    if len(chip_gws_used) != len(set(chip_gws_used)):
                        st.error("You can only play one chip per gameweek.")

                    # Show chip summary
                    if chip_schedule:
                        chip_labels = {"wildcard": "🃏 Wildcard", "free_hit": "⚡ Free Hit",
                                       "triple_captain": "👑 Triple Captain", "bench_boost": "💪 Bench Boost"}
                        chip_summary = " · ".join([
                            f"GW{gw}: {chip_labels.get(c, c)}" for gw, c in sorted(chip_schedule.items())
                        ])
                        st.info(f"Chip plan: {chip_summary}")
                    else:
                        st.markdown(
                            "<span style='color:#5a6580;font-size:0.8rem;'>No chips selected — planning with normal transfers only</span>",
                            unsafe_allow_html=True,
                        )

                    st.markdown("")

                    # Step 2: Generate plan
                    st.markdown("**Step 2: Generate your optimal plan**")
                    generate = st.button("🚀 Generate 6-Gameweek Plan", use_container_width=True, type="primary")

                    if generate or st.session_state.get("plan_generated"):
                        st.session_state["plan_generated"] = True

                        with st.spinner("Building 6-gameweek rolling plan..."):
                            plan = build_rolling_plan(
                                my_squad, df,
                                bank=team_data["bank"],
                                free_transfers=ft_available,
                                purchase_prices=team_data.get("purchase_prices", {}),
                                selling_prices_api=team_data.get("selling_prices_api", {}),
                                xpts_map=xpts_map,
                                planning_gw_id=planning_gw_id,
                                n_gws=6,
                                chip_schedule=chip_schedule,
                                team_fixture_counts=team_fixture_counts,
                            )

                        st.markdown("")

                        # Plan summary
                        if plan:
                            total_xpts = 0
                            total_hits = 0
                            total_transfers = 0
                            for gw_e in plan:
                                xi = gw_e.get("xi")
                                if xi is not None and "xpts_gw" in xi.columns:
                                    cap = gw_e.get("captain")
                                    cap_mult = gw_e.get("captain_multiplier", 2)
                                    xi_pts = xi["xpts_gw"].sum()
                                    # Add captain bonus (extra x1 or x2)
                                    if cap is not None:
                                        cap_xpts = xpts_map.get(cap.get("id", 0), {}).get(gw_e["gw"], 0)
                                        xi_pts += cap_xpts * (cap_mult - 1)
                                    # Add bench if bench boost
                                    if gw_e.get("bench_boost") and gw_e.get("bench") is not None:
                                        xi_pts += gw_e["bench"]["xpts_gw"].sum() if "xpts_gw" in gw_e["bench"].columns else 0
                                    total_xpts += xi_pts
                                total_hits += gw_e.get("hit", 0)
                                total_transfers += len(gw_e.get("transfers", []))

                            sc1, sc2, sc3 = st.columns(3)
                            with sc1:
                                st.markdown(f"""<div class="metric-card">
                                    <div class="metric-label">Projected Total (6GW)</div>
                                    <div class="metric-value">{total_xpts:.0f} xPts</div>
                                    <div class="metric-sub">After captain bonus</div>
                                </div>""", unsafe_allow_html=True)
                            with sc2:
                                st.markdown(f"""<div class="metric-card">
                                    <div class="metric-label">Transfers Planned</div>
                                    <div class="metric-value">{total_transfers}</div>
                                    <div class="metric-sub">{total_hits}pts in hits</div>
                                </div>""", unsafe_allow_html=True)
                            with sc3:
                                chips_used = sum(1 for g in plan if g.get("chip"))
                                st.markdown(f"""<div class="metric-card">
                                    <div class="metric-label">Chips Used</div>
                                    <div class="metric-value">{chips_used}</div>
                                    <div class="metric-sub">of {len(chip_schedule)} planned</div>
                                </div>""", unsafe_allow_html=True)

                            st.markdown("")

                            for gw_entry in plan:
                                gw = gw_entry["gw"]
                                transfer = gw_entry["transfer"]
                                xi = gw_entry["xi"]
                                bench = gw_entry["bench"]
                                hit = gw_entry["hit"]
                                chip = gw_entry.get("chip")
                                captain = gw_entry.get("captain")
                                cap_mult = gw_entry.get("captain_multiplier", 2)
                                is_bb = gw_entry.get("bench_boost", False)

                                # Expander label with chip badge
                                chip_labels = {
                                    "wildcard": "🃏 WILDCARD",
                                    "free_hit": "⚡ FREE HIT",
                                    "triple_captain": "👑 TRIPLE CAPTAIN",
                                    "bench_boost": "💪 BENCH BOOST",
                                }
                                chip_str = f" — {chip_labels.get(chip, '')}" if chip else ""
                                with st.expander(f"**Gameweek {gw}**{chip_str}", expanded=(gw == planning_gw_id)):

                                    # Transfer / chip action
                                    if chip == "wildcard":
                                        squad_count = len(gw_entry.get("squad", []))
                                        st.markdown(
                                            f"<div class='transfer-card'>"
                                            f"<span style='color:#a78bfa;font-weight:700;'>🃏 WILDCARD ACTIVE</span>"
                                            f"<br><span style='color:#8892a8;font-size:0.72rem;'>"
                                            f"Full squad rebuilt via MILP solver ({squad_count} players) — optimised for remaining GWs</span>"
                                            f"</div>",
                                            unsafe_allow_html=True,
                                        )
                                    elif chip == "free_hit":
                                        st.markdown(
                                            f"<div class='transfer-card'>"
                                            f"<span style='color:#38bdf8;font-weight:700;'>⚡ FREE HIT ACTIVE</span>"
                                            f"<br><span style='color:#8892a8;font-size:0.72rem;'>"
                                            f"Best possible squad for this single GW — reverts to your team next week</span>"
                                            f"</div>",
                                            unsafe_allow_html=True,
                                        )
                                    else:
                                        transfers_list = gw_entry.get("transfers", [])
                                        gw_hit = gw_entry.get("hit", 0)
                                        ft_used = gw_entry.get("ft_used", 0)
                                        n_total = len(transfers_list)

                                        if n_total > 0:
                                            # Summary line
                                            hit_parts = []
                                            if ft_used > 0:
                                                hit_parts.append(f"{ft_used} free")
                                            paid = n_total - ft_used
                                            if paid > 0:
                                                hit_parts.append(f"{paid} hit (-{paid * 4}pts)")
                                            summary = " + ".join(hit_parts)
                                            total_gain = sum(t["xpts_gain"] for t in transfers_list)

                                            st.markdown(
                                                f"<span style='color:#8892a8;font-size:0.75rem;'>"
                                                f"**{n_total} transfer{'s' if n_total > 1 else ''}** ({summary}) · "
                                                f"Horizon xPts gain: <span style='color:#34d399;font-weight:600;'>+{total_gain:.1f}</span>"
                                                f"{f' · Net after hit: +{total_gain - gw_hit:.1f}' if gw_hit > 0 else ''}"
                                                f"</span>",
                                                unsafe_allow_html=True,
                                            )

                                            for t_idx, t in enumerate(transfers_list):
                                                o = t["out"]
                                                i_p = t["in"]
                                                sp = calculate_selling_price(
                                                    o["id"], o["now_cost"],
                                                    team_data.get("purchase_prices", {}),
                                                    team_data.get("selling_prices_api", {})
                                                )
                                                is_free = (t_idx < ft_used)
                                                tag = "Free" if is_free else "-4pt hit"
                                                tag_color = "#34d399" if is_free else "#f87171"
                                                gw_gain_str = f" · This GW: +{t.get('xpts_gw_gain', 0):.1f}" if "xpts_gw_gain" in t else ""

                                                st.markdown(f"""<div class="transfer-card">
                                                    <span class="transfer-out">▼ {o['name']}</span>
                                                    <span style="color:#5a6580;font-size:0.7rem;">
                                                        {o.get('pos','?')} · {o.get('team','?')} · SP £{sp/10:.1f}m
                                                    </span>
                                                    &nbsp;<span class="transfer-arrow">→</span>&nbsp;
                                                    <span class="transfer-in">▲ {i_p['name']}</span>
                                                    <span style="color:#5a6580;font-size:0.7rem;">
                                                        {i_p.get('pos','?')} · {i_p.get('team','?')} · £{i_p['now_cost']/10:.1f}m
                                                    </span>
                                                    <br>
                                                    <span style="color:#34d399;font-size:0.72rem;font-weight:600;">
                                                        +{t['xpts_gain']:.1f} xPts remaining horizon{gw_gain_str}
                                                    </span>
                                                    <span style="color:{tag_color};font-size:0.7rem;font-weight:600;"> · {tag}</span>
                                                    <span style="color:#5a6580;font-size:0.7rem;"> ·
                                                        £{t['new_bank']/10:.1f}m ITB
                                                    </span>
                                                </div>""", unsafe_allow_html=True)
                                        else:
                                            st.markdown(
                                                "<div class='transfer-card'>"
                                                "<span style='color:#8892a8;'>No transfer — banking free transfer</span>"
                                                "</div>",
                                                unsafe_allow_html=True,
                                            )
                                    # Best XI pitch view
                                    if xi is not None and len(xi) >= 11:
                                        formation = get_formation_str(xi)
                                        xi_total = xi["xpts_gw"].sum() if "xpts_gw" in xi.columns else 0
                                        st.markdown(
                                            f"<span style='color:#8892a8;font-size:0.78rem;'>"
                                            f"Best XI: {formation} · Projected {xi_total:.1f} xPts</span>",
                                            unsafe_allow_html=True,
                                        )

                                        for pid_val, plabel in [(4, "FWD"), (3, "MID"), (2, "DEF"), (1, "GK")]:
                                            pp = xi[xi["pos_id"] == pid_val]
                                            if len(pp) > 0:
                                                cols = st.columns(max(len(pp), 1))
                                                for ci, (_, p) in enumerate(pp.iterrows()):
                                                    sc = f"pitch-shirt-{POS_MAP.get(p['pos_id'],'mid').lower()}"
                                                    gw_xpts = p.get("xpts_gw", 0)
                                                    with cols[ci]:
                                                        st.markdown(f"""<div style="text-align:center;">
                                                            <div class="pitch-shirt {sc}">{gw_xpts:.1f}</div>
                                                            <div class="pitch-name">{p['name']}</div>
                                                            <div class="pitch-price">£{p['price']:.1f}m</div>
                                                        </div>""", unsafe_allow_html=True)

                                        if bench is not None and len(bench) > 0:
                                            if is_bb:
                                                bench_label = "💪 BENCH BOOST — all bench players score:"
                                            else:
                                                bench_label = "Bench:"
                                            bench_names = ", ".join([
                                                f"{r['name']} ({r.get('xpts_gw', 0):.1f})"
                                                for _, r in bench.iterrows()
                                            ])
                                            st.markdown(
                                                f"<span style='color:#5a6580;font-size:0.68rem;'>"
                                                f"{'💪 ' if is_bb else ''}{bench_label} {bench_names}</span>",
                                                unsafe_allow_html=True,
                                            )

                                    # Captain recommendation
                                    if captain is not None:
                                        cap_xpts = xpts_map.get(captain.get("id", 0), {}).get(gw, 0)
                                        cap_label = "Triple Captain" if cap_mult == 3 else "Captain"
                                        cap_emoji = "👑" if cap_mult == 3 else "©️"
                                        st.markdown(
                                            f"<span style='color:#fbbf24;font-size:0.78rem;font-weight:600;'>"
                                            f"{cap_emoji} {cap_label}: {captain.get('name', '?')} "
                                            f"({cap_xpts:.1f} xPts × {cap_mult} = {cap_xpts * cap_mult:.1f})</span>",
                                            unsafe_allow_html=True,
                                        )
                        else:
                            st.info("Could not build a rolling plan — not enough fixture data.")

                    # Full squad table
                    st.markdown("")
                    st.subheader("Full Squad Breakdown")
                    sq_show = my_squad.sort_values(["is_starter", "pos_id", "xpts_total"],
                                                    ascending=[False, True, False])
                    # Add selling price column
                    sq_show = sq_show.copy()
                    sq_show["sell_price"] = sq_show.apply(
                        lambda r: calculate_selling_price(
                            r["id"], r["now_cost"],
                            team_data.get("purchase_prices", {}),
                            team_data.get("selling_prices_api", {})
                        ) / 10, axis=1
                    )
                    sq_display = sq_show[["name", "team", "pos", "price", "sell_price", "total_points",
                                          "form_str", "xpts_next_gw", "xpts_total", "is_starter"]].copy()
                    sq_display.columns = ["Player", "Team", "Pos", "Mkt Price", "Sell Price", "Pts",
                                          "Form", "xPts GW", "xPts 6GW", "Starter"]
                    sq_display["Starter"] = sq_display["Starter"].map({True: "XI", False: "Bench"})
                    sq_display = sq_display.reset_index(drop=True)
                    sq_display.index += 1
                    st.dataframe(sq_display, use_container_width=True)
                else:
                    st.warning("Could not match squad players to current data.")
        else:
            st.info("Enter your FPL Team ID above to get personalised transfer suggestions. "
                    "You can find it in the URL when you view your team on the FPL website "
                    "(e.g. fantasy.premierleague.com/entry/**123456**/event/1)")

    # ==================== DASHBOARD ====================
    with tab2:
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
    with tab3:
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
    with tab4:
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
    with tab5:
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
    with tab6:
        st.markdown('<div class="section-header">Fixture Difficulty — Next 6 Gameweeks <span class="source-tag src-odds">Odds-enhanced</span></div>', unsafe_allow_html=True)
        gw_range = list(range(planning_gw_id, planning_gw_id + 6))

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
                fc = team_fixture_counts.get(t_id, {}).get(gw, 1)
                if fc == 0:
                    row[f"GW{gw}"] = "BLANK"
                elif fix:
                    opp = teams.get(fix["opp"], {}).get("short_name", "???")
                    pre = "" if fix["home"] else "@"
                    dgw = " [DGW]" if fc >= 2 else ""
                    row[f"GW{gw}"] = f"{pre}{opp} ({fix['diff']}){dgw}"
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
        f"Datumly · Data-driven FPL intelligence · "
        f"FPL API + football-data.co.uk + PuLP MILP · "
        f"{datetime.now().strftime('%d %b %Y, %H:%M')}"
        f"</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
