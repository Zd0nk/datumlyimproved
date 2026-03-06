import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="FPL Optimizer",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

FPL_BASE = "https://fantasy.premierleague.com/api"
POS_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
POS_FULL = {1: "Goalkeeper", 2: "Defender", 3: "Midfielder", 4: "Forward"}

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
    .sub-title { color: #8892a8; font-size: 0.9rem; margin-top: -10px; }

    .metric-card {
        background: #111827; border: 1px solid #2a3550;
        border-radius: 14px; padding: 1.2rem; text-align: center;
    }
    .metric-label { color: #5a6580; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
    .metric-value { color: #e2e8f0; font-size: 1.8rem; font-weight: 700; letter-spacing: -0.5px; }
    .metric-sub { color: #8892a8; font-size: 0.78rem; margin-top: 2px; }

    .fdr-1 { background:#065f46; color:#6ee7b7; padding:3px 8px; border-radius:6px; font-size:0.75rem; font-weight:600; display:inline-block; margin:1px; }
    .fdr-2 { background:#14532d; color:#86efac; padding:3px 8px; border-radius:6px; font-size:0.75rem; font-weight:600; display:inline-block; margin:1px; }
    .fdr-3 { background:#78350f; color:#fcd34d; padding:3px 8px; border-radius:6px; font-size:0.75rem; font-weight:600; display:inline-block; margin:1px; }
    .fdr-4 { background:#7c2d12; color:#fdba74; padding:3px 8px; border-radius:6px; font-size:0.75rem; font-weight:600; display:inline-block; margin:1px; }
    .fdr-5 { background:#7f1d1d; color:#fca5a5; padding:3px 8px; border-radius:6px; font-size:0.75rem; font-weight:600; display:inline-block; margin:1px; }

    .transfer-card {
        background: #1a2236; border: 1px solid #2a3550;
        border-radius: 10px; padding: 0.8rem 1rem; margin-bottom: 0.5rem;
    }
    .transfer-out { color: #f87171; font-weight: 600; }
    .transfer-in { color: #34d399; font-weight: 600; }
    .transfer-arrow { color: #38bdf8; font-size: 1.2rem; }

    .gw-bar {
        background: #111827; border: 1px solid #2a3550; border-radius: 12px;
        padding: 0.7rem 1.2rem; display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;
    }
    .gw-num { background: linear-gradient(135deg,#38bdf8,#818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 1.1rem; font-weight: 700; }
    .gw-deadline { color: #8892a8; font-size: 0.8rem; }

    .badge { font-size:0.68rem; padding:3px 10px; border-radius:6px; font-weight:600; }
    .badge-green { background:rgba(52,211,153,0.15); color:#34d399; }
    .badge-yellow { background:rgba(251,191,36,0.15); color:#fbbf24; }
    .badge-blue { background:rgba(56,189,248,0.15); color:#38bdf8; }

    .pitch-row-label { color:#5a6580; font-size:0.7rem; text-transform:uppercase; letter-spacing:1px; margin-bottom:6px; margin-top:14px; }
    .pitch-shirt { width:42px; height:42px; border-radius:10px; display:inline-flex; align-items:center; justify-content:center; font-weight:700; font-size:0.8rem; color:white; margin:0 auto; }
    .pitch-shirt-gkp { background:#f59e0b; }
    .pitch-shirt-def { background:#3b82f6; }
    .pitch-shirt-mid { background:#10b981; }
    .pitch-shirt-fwd { background:#ef4444; }
    .pitch-name { font-size:0.72rem; font-weight:600; color:#e2e8f0; margin-top:4px; }
    .pitch-price { font-size:0.62rem; color:#5a6580; }

    #MainMenu {visibility:hidden;}
    footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data(ttl=3600)
def load_fpl_data():
    """Fetch data from FPL API (server-side = no CORS)."""
    try:
        headers = {"User-Agent": "FPL-Optimizer/1.0"}
        bootstrap = requests.get(f"{FPL_BASE}/bootstrap-static/", headers=headers, timeout=30).json()
        fixtures = requests.get(f"{FPL_BASE}/fixtures/", headers=headers, timeout=30).json()
        return bootstrap, fixtures, None
    except Exception as e:
        return None, None, str(e)


def enrich_data(bootstrap, fixtures):
    """Process raw FPL data into enriched player DataFrame."""
    players = bootstrap["elements"]
    teams = {t["id"]: t for t in bootstrap["teams"]}
    events = bootstrap["events"]
    current_gw = next((e for e in events if e["is_current"]), None) or \
                 next((e for e in events if e["is_next"]), None) or \
                 (events[0] if events else None)
    gw_id = current_gw["id"] if current_gw else 1

    # Upcoming fixtures per team
    upcoming = {t_id: [] for t_id in teams}
    for f in sorted(fixtures, key=lambda x: x.get("event", 0) or 0):
        ev = f.get("event")
        if ev and gw_id <= ev < gw_id + 6:
            if f["team_h"] in upcoming:
                upcoming[f["team_h"]].append({"gw": ev, "opp": f["team_a"], "home": True, "difficulty": f.get("team_h_difficulty", 3)})
            if f["team_a"] in upcoming:
                upcoming[f["team_a"]].append({"gw": ev, "opp": f["team_h"], "home": False, "difficulty": f.get("team_a_difficulty", 3)})

    # Recent results per team
    recent = {t_id: [] for t_id in teams}
    for f in sorted(fixtures, key=lambda x: x.get("event", 0) or 0, reverse=True):
        if f.get("finished") and f.get("team_h_score") is not None:
            h = "W" if f["team_h_score"] > f["team_a_score"] else ("D" if f["team_h_score"] == f["team_a_score"] else "L")
            a = "W" if f["team_a_score"] > f["team_h_score"] else ("D" if f["team_a_score"] == f["team_h_score"] else "L")
            if len(recent.get(f["team_h"], [])) < 5: recent[f["team_h"]].append(h)
            if len(recent.get(f["team_a"], [])) < 5: recent[f["team_a"]].append(a)

    # Build rows
    rows = []
    for p in players:
        td = teams.get(p["team"], {})
        price = p["now_cost"] / 10
        form = float(p.get("form", 0) or 0)
        mins = p.get("minutes", 0) or 0
        pts = p.get("total_points", 0) or 0
        ict = float(p.get("ict_index", 0) or 0)
        goals = p.get("goals_scored", 0) or 0
        assists = p.get("assists", 0) or 0

        ppg = pts / max(mins / 90, 1)
        value = pts / max(price, 1)
        xg = (goals * 1.5 + assists) / max(mins / 90, 0.5)

        uf = upcoming.get(p["team"], [])[:4]
        avg_d = np.mean([f["difficulty"] for f in uf]) if uf else 3.0
        fs = (5 - avg_d) * 2
        composite = form * 3 + value * 0.8 + fs * 2 + ict * 0.03 + ppg * 1.5 + xg * 5

        rows.append({
            "id": p["id"], "name": p.get("web_name", ""),
            "first_name": p.get("first_name", ""), "second_name": p.get("second_name", ""),
            "team_id": p["team"], "team": td.get("short_name", "???"),
            "team_name": td.get("name", "???"),
            "pos_id": p["element_type"], "pos": POS_MAP.get(p["element_type"], "?"),
            "price": price, "now_cost": p["now_cost"],
            "total_points": pts, "form": form, "form_str": str(p.get("form", "0.0")),
            "ict_index": round(ict, 1), "ppg": round(ppg, 2), "value": round(value, 2),
            "xg_proxy": round(xg, 2), "composite": round(composite, 1),
            "avg_difficulty": round(avg_d, 2), "minutes": mins,
            "goals": goals, "assists": assists,
            "clean_sheets": p.get("clean_sheets", 0) or 0,
            "selected_pct": float(p.get("selected_by_percent", 0) or 0),
            "transfers_in": p.get("transfers_in_event", 0) or 0,
            "transfers_out": p.get("transfers_out_event", 0) or 0,
            "status": p.get("status", "a"), "news": p.get("news", ""),
        })

    return pd.DataFrame(rows), teams, current_gw, upcoming, fixtures


def pick_team(pool, requirements, max_budget):
    """Greedy team picker within budget and max 3 per team."""
    picked, tc, spent = [], {}, 0
    for pos_id in [4, 3, 2, 1]:
        pp = pool[pool["pos_id"] == pos_id]
        need = requirements.get(pos_id, 0)
        count = 0
        for _, p in pp.iterrows():
            if count >= need: break
            if p["id"] in [x["id"] for x in picked]: continue
            if tc.get(p["team_id"], 0) >= 3: continue
            if spent + p["now_cost"] > max_budget: continue
            picked.append(p)
            tc[p["team_id"]] = tc.get(p["team_id"], 0) + 1
            spent += p["now_cost"]
            count += 1
        if count < need: return None
    return pd.DataFrame(picked)


# ============================================================
# MAIN APP
# ============================================================
def main():
    st.markdown('<div class="main-title">⚽ FPL Optimizer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Optimal transfers & team selection powered by live FPL data</div>', unsafe_allow_html=True)
    st.markdown("")

    with st.spinner("Fetching live FPL data..."):
        bootstrap, fixtures_raw, error = load_fpl_data()

    if error or bootstrap is None:
        st.error(f"Failed to load FPL data: {error}")
        st.info("The FPL API may be down or the season hasn't started. Try refreshing.")
        if st.button("🔄 Retry"):
            st.cache_data.clear()
            st.rerun()
        return

    df, teams, current_gw, upcoming_map, fixtures_list = enrich_data(bootstrap, fixtures_raw)

    # GW info bar
    if current_gw:
        deadline = datetime.fromisoformat(current_gw["deadline_time"].replace("Z", "+00:00"))
        deadline_str = deadline.strftime("%a %d %b, %H:%M")
        status = "Completed" if current_gw.get("finished") else ("In Progress" if current_gw.get("is_current") else "Upcoming")
        bc = "badge-green" if status == "Completed" else ("badge-yellow" if status == "In Progress" else "badge-blue")
        st.markdown(f"""<div class="gw-bar">
            <span class="gw-num">Gameweek {current_gw['id']}</span>
            <span class="gw-deadline">Deadline: {deadline_str}</span>
            <span class="badge {bc}">{status}</span>
        </div>""", unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "👥 Player Rankings", "🔄 Transfer Picks", "⭐ Best XV", "📅 Fixtures"])

    active = df[df["minutes"] > 0].copy()
    qualified = df[df["minutes"] > 90].copy()

    # ==================== DASHBOARD ====================
    with tab1:
        if len(active) > 0:
            c1, c2, c3, c4 = st.columns(4)
            ts = active.loc[active["total_points"].idxmax()]
            bv = active.loc[active["value"].idxmax()]
            bf = active.loc[active["form"].idxmax()]
            mt = active.loc[active["transfers_in"].idxmax()]

            for col, label, val, sub in [
                (c1, "Top Scorer", int(ts["total_points"]), f"{ts['name']} ({ts['team']})"),
                (c2, "Best Value", f"{bv['value']:.1f}", f"{bv['name']} (£{bv['price']:.1f}m)"),
                (c3, "Hottest Form", f"{bf['form']:.1f}", f"{bf['name']} ({bf['team']})"),
                (c4, "Most Transferred In", f"{int(mt['transfers_in']):,}", f"{mt['name']} ({mt['team']})"),
            ]:
                with col:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{val}</div>
                        <div class="metric-sub">{sub}</div>
                    </div>""", unsafe_allow_html=True)

        st.markdown("")
        st.subheader("Top Value Picks This Week")
        if len(qualified) > 0:
            tp = qualified.nlargest(12, "composite")[["name", "team", "pos", "price", "total_points", "form_str", "composite"]].copy()
            tp.columns = ["Player", "Team", "Pos", "Price (£m)", "Points", "Form", "Score"]
            tp = tp.reset_index(drop=True); tp.index += 1
            st.dataframe(tp, use_container_width=True, height=460)

        st.markdown("")
        st.subheader("Differential Gems (< 10% ownership)")
        if len(qualified) > 0:
            diffs = qualified[qualified["selected_pct"] < 10].nlargest(10, "composite")
            if len(diffs) > 0:
                dd = diffs[["name", "team", "pos", "price", "total_points", "form_str", "selected_pct", "composite"]].copy()
                dd.columns = ["Player", "Team", "Pos", "Price (£m)", "Points", "Form", "Own%", "Score"]
                dd = dd.reset_index(drop=True); dd.index += 1
                st.dataframe(dd, use_container_width=True, height=390)

    # ==================== PLAYER RANKINGS ====================
    with tab2:
        fc1, fc2, fc3, fc4, fc5 = st.columns([2, 1, 1, 1, 1])
        with fc1: search = st.text_input("🔍 Search", "", key="ps")
        with fc2: pos_f = st.selectbox("Position", ["All"] + list(POS_FULL.values()), key="pf")
        with fc3: team_f = st.selectbox("Team", ["All"] + sorted(df["team_name"].unique().tolist()), key="tf")
        with fc4: price_f = st.selectbox("Price", ["All", "Under £5.0m", "£5.0-£7.0m", "£7.0-£10.0m", "Over £10.0m"], key="prf")
        with fc5:
            so = {"Composite Score": "composite", "Total Points": "total_points", "Form": "form", "Value": "value", "ICT Index": "ict_index"}
            sort_f = st.selectbox("Sort by", list(so.keys()), key="sf")

        fl = active.copy()
        if search:
            sl = search.lower()
            fl = fl[fl["name"].str.lower().str.contains(sl, na=False) | fl["first_name"].str.lower().str.contains(sl, na=False) | fl["second_name"].str.lower().str.contains(sl, na=False)]
        if pos_f != "All":
            pid = {v: k for k, v in POS_FULL.items()}[pos_f]
            fl = fl[fl["pos_id"] == pid]
        if team_f != "All": fl = fl[fl["team_name"] == team_f]
        if price_f == "Under £5.0m": fl = fl[fl["price"] < 5]
        elif price_f == "£5.0-£7.0m": fl = fl[(fl["price"] >= 5) & (fl["price"] < 7)]
        elif price_f == "£7.0-£10.0m": fl = fl[(fl["price"] >= 7) & (fl["price"] < 10)]
        elif price_f == "Over £10.0m": fl = fl[fl["price"] >= 10]

        fl = fl.sort_values(so[sort_f], ascending=False)
        sd = fl.head(80)[["name", "team", "pos", "price", "total_points", "form_str", "ict_index", "xg_proxy", "selected_pct", "composite"]].copy()
        sd.columns = ["Player", "Team", "Pos", "Price", "Pts", "Form", "ICT", "xG Inv", "Own%", "Score"]
        sd = sd.reset_index(drop=True); sd.index += 1
        st.dataframe(sd, use_container_width=True, height=700)
        st.caption(f"Showing {min(80, len(fl))} of {len(fl)} players")

    # ==================== TRANSFER PICKS ====================
    with tab3:
        st.subheader("🎯 Recommended Transfers")
        st.caption("Matching positions, at least 15% composite improvement.")

        if len(qualified) > 0:
            cands = qualified.nlargest(30, "composite")
            outs = qualified[qualified["selected_pct"] > 15].nsmallest(15, "composite")
            transfers, ui, uo = [], set(), set()

            for _, ip in cands.iterrows():
                for _, op in outs.iterrows():
                    if ip["id"] in ui or op["id"] in uo: continue
                    if ip["pos_id"] != op["pos_id"] or ip["id"] == op["id"]: continue
                    if ip["composite"] <= op["composite"] * 1.15: continue
                    reasons = []
                    if ip["form"] > op["form"]: reasons.append("Better form")
                    if ip["avg_difficulty"] < op["avg_difficulty"]: reasons.append("Easier fixtures")
                    if ip["value"] > op["value"]: reasons.append("Better value")
                    if not reasons: reasons.append("Higher composite")
                    transfers.append({"out": op, "in": ip, "reasons": reasons})
                    ui.add(ip["id"]); uo.add(op["id"])
                    if len(transfers) >= 8: break
                if len(transfers) >= 8: break

            if transfers:
                for t in transfers:
                    o, i = t["out"], t["in"]
                    rs = " · ".join(t["reasons"])
                    st.markdown(f"""<div class="transfer-card">
                        <span class="transfer-out">▼ {o['name']}</span>
                        <span style="color:#5a6580;font-size:0.75rem;"> {o['pos']} · {o['team']} · £{o['price']:.1f}m · {int(o['total_points'])}pts</span>
                        &nbsp;&nbsp;<span class="transfer-arrow">→</span>&nbsp;&nbsp;
                        <span class="transfer-in">▲ {i['name']}</span>
                        <span style="color:#5a6580;font-size:0.75rem;"> {i['pos']} · {i['team']} · £{i['price']:.1f}m · {int(i['total_points'])}pts</span>
                        <br><span style="color:#8892a8;font-size:0.72rem;">{rs}</span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.info("No clear transfer improvements found.")

        st.markdown("")
        cr, cf = st.columns(2)
        with cr:
            st.subheader("📈 Most Transferred In")
            ri = active.nlargest(10, "transfers_in")[["name", "team", "pos", "price", "transfers_in", "composite"]].copy()
            ri.columns = ["Player", "Team", "Pos", "Price", "In", "Score"]
            ri = ri.reset_index(drop=True); ri.index += 1
            st.dataframe(ri, use_container_width=True)
        with cf:
            st.subheader("📉 Most Transferred Out")
            fo = active.nlargest(10, "transfers_out")[["name", "team", "pos", "price", "transfers_out", "composite"]].copy()
            fo.columns = ["Player", "Team", "Pos", "Price", "Out", "Score"]
            fo = fo.reset_index(drop=True); fo.index += 1
            st.dataframe(fo, use_container_width=True)

    # ==================== BEST XV ====================
    with tab4:
        st.subheader("⭐ Optimal 15-Man Squad")
        st.caption("Within £100m, max 3 per team. Best formation auto-selected.")

        if len(qualified) > 0:
            avail = qualified[qualified["status"].isin(["a", "d"])].sort_values("composite", ascending=False)
            formations = [
                {"d": 4, "m": 4, "f": 2, "l": "4-4-2"}, {"d": 3, "m": 5, "f": 2, "l": "3-5-2"},
                {"d": 4, "m": 3, "f": 3, "l": "4-3-3"}, {"d": 3, "m": 4, "f": 3, "l": "3-4-3"},
                {"d": 5, "m": 3, "f": 2, "l": "5-3-2"}, {"d": 5, "m": 4, "f": 1, "l": "5-4-1"},
            ]
            bt, bs, bfl = None, 0, ""
            for fm in formations:
                r = {1: 1, 2: fm["d"], 3: fm["m"], 4: fm["f"]}
                t = pick_team(avail, r, 1000)
                if t is not None:
                    s = t["composite"].sum()
                    if s > bs: bs, bt, bfl = s, t, fm["l"]

            if bt is not None:
                ids = set(bt["id"])
                tc = bt["team_id"].value_counts().to_dict()
                rem = avail[~avail["id"].isin(ids)]
                bench = []
                for _, p in rem.iterrows():
                    if p["pos_id"] == 1 and tc.get(p["team_id"], 0) < 3:
                        bench.append(p); tc[p["team_id"]] = tc.get(p["team_id"], 0) + 1; break
                for _, p in rem.iterrows():
                    if len(bench) >= 4: break
                    if p["pos_id"] != 1 and p["id"] not in [b["id"] for b in bench] and tc.get(p["team_id"], 0) < 3:
                        bench.append(p); tc[p["team_id"]] = tc.get(p["team_id"], 0) + 1

                bench_df = pd.DataFrame(bench) if bench else pd.DataFrame()
                all_sq = pd.concat([bt, bench_df]) if len(bench_df) > 0 else bt
                cost = all_sq["now_cost"].sum() / 10

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Cost", f"£{cost:.1f}m")
                c2.metric("Composite Score", f"{bs:.0f}")
                c3.metric("Formation", bfl)
                st.markdown("")

                for pid, plabel in [(4, "Forwards"), (3, "Midfielders"), (2, "Defenders"), (1, "Goalkeeper")]:
                    pp = bt[bt["pos_id"] == pid]
                    if len(pp) > 0:
                        st.markdown(f"<div class='pitch-row-label'>{plabel}</div>", unsafe_allow_html=True)
                        cols = st.columns(max(len(pp), 1))
                        for i, (_, p) in enumerate(pp.iterrows()):
                            sc = f"pitch-shirt-{p['pos'].lower()}"
                            with cols[i]:
                                st.markdown(f"""<div style="text-align:center;">
                                    <div class="pitch-shirt {sc}">{int(p['total_points'])}</div>
                                    <div class="pitch-name">{p['name']}</div>
                                    <div class="pitch-price">£{p['price']:.1f}m · {p['form_str']}</div>
                                </div>""", unsafe_allow_html=True)

                st.markdown(""); st.markdown("**Bench**")
                if len(bench_df) > 0:
                    bc = st.columns(len(bench_df))
                    for i, (_, p) in enumerate(bench_df.iterrows()):
                        with bc[i]:
                            st.markdown(f"""<div style="text-align:center;opacity:0.7;">
                                <div class="pitch-name">{p['name']}</div>
                                <div class="pitch-price">{p['pos']} · £{p['price']:.1f}m</div>
                            </div>""", unsafe_allow_html=True)

                st.markdown(""); st.subheader("Full Squad")
                sq = all_sq[["name", "team", "pos", "price", "total_points", "form_str", "composite"]].copy()
                sq.columns = ["Player", "Team", "Pos", "Price", "Pts", "Form", "Score"]
                sq = sq.reset_index(drop=True); sq.index += 1
                st.dataframe(sq, use_container_width=True)
            else:
                st.warning("Not enough data to generate squad.")

    # ==================== FIXTURES ====================
    with tab5:
        st.subheader("Fixture Difficulty — Next 6 Gameweeks")
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
            for gw in gw_range:
                fix = fm.get(t_id, {}).get(gw)
                if fix:
                    opp = teams.get(fix["opp"], {}).get("short_name", "???")
                    pre = "" if fix["home"] else "@"
                    row[f"GW{gw}"] = f"{pre}{opp} ({fix['diff']})"
                    diffs.append(fix["diff"])
                else:
                    row[f"GW{gw}"] = "-"
            row["Avg"] = round(np.mean(diffs), 1) if diffs else 3.0
            rows.append(row)

        fdf = pd.DataFrame(rows).sort_values("Avg").reset_index(drop=True)
        fdf.index += 1
        st.dataframe(fdf, use_container_width=True, height=740)

    st.markdown("---")
    st.caption(f"Data from FPL API · Cached for 1 hour · Last loaded: {datetime.now().strftime('%d %b %Y, %H:%M')}")


if __name__ == "__main__":
    main()
