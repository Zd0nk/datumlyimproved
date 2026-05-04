"""
Solver correctness validation: brute-force vs MILP.

Generates a small synthetic player pool, exhaustively enumerates valid
15-man squads, finds the true optimal squad+XI by computing the best
objective, and verifies that solve_optimal_squad returns the same objective.

Run directly:    py tests\test_solver_correctness.py
Or with pytest:  py -m pytest tests\test_solver_correctness.py -v

Why this test exists
--------------------
PuLP + CBC is a battle-tested solver, but solver correctness depends on:
  1. The MILP formulation correctly expressing the intended optimisation
  2. The constraint set being complete (no "magic" constraints missing)
  3. Tie-breaking and edge cases behaving sensibly

A gold-standard correctness check is to construct a problem small enough that
brute force is tractable, solve both ways, and confirm the objective values
match. If they don't, the formulation has a bug — not the solver.
"""
import sys
from itertools import combinations
from unittest.mock import MagicMock

import pandas as pd

# --- Mock streamlit before importing app.py -------------------------------
# app.py runs `st.set_page_config` and reads st.session_state at import time,
# so we can't import it cold from a non-Streamlit context. Replace the
# streamlit modules with mocks that satisfy the few attribute accesses
# performed during module load.
class _MockSession(dict):
    def __getattr__(self, k):
        return self.get(k)

    def __setattr__(self, k, v):
        self[k] = v


_mock_st = MagicMock()
_mock_st.session_state = _MockSession({"active_league": "FPL", "active_nav": "my_team"})
sys.modules["streamlit"] = _mock_st
sys.modules["streamlit.components.v1"] = MagicMock()

# Add repo root to path so `from app import ...` works from tests/
import os
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from app import solve_optimal_squad  # noqa: E402


# ============================================================
# Test data generation
# ============================================================
def make_player_pool(seed=42):
    """Synthetic 22-player pool: 3 GK / 7 DEF / 7 MID / 5 FWD across 5 teams.

    Sized so brute force is fast (~13k squad combinations) but the budget
    and max-3-per-team constraints are still binding.
    """
    import random
    random.seed(seed)

    rows = []
    pid = 1

    # GK: 3 players across teams 1-3
    for tid, cost, xpts in [
        (1, 45, 22.5), (2, 50, 28.0), (3, 40, 18.0),
    ]:
        rows.append(dict(id=pid, team_id=tid, pos_id=1, now_cost=cost,
                         xpts_total=xpts, minutes=1500, status="a"))
        pid += 1

    # DEF: 7 players, varied teams
    for tid, cost, xpts in [
        (1, 60, 30.0), (2, 55, 28.0), (3, 50, 24.0), (4, 45, 18.0),
        (5, 70, 35.0), (1, 50, 22.0), (4, 65, 32.0),
    ]:
        rows.append(dict(id=pid, team_id=tid, pos_id=2, now_cost=cost,
                         xpts_total=xpts, minutes=1500, status="a"))
        pid += 1

    # MID: 7 players, premium options included
    for tid, cost, xpts in [
        (1, 110, 48.0), (2, 95, 42.0), (3, 75, 35.0), (4, 60, 28.0),
        (5, 130, 55.0), (2, 55, 25.0), (5, 65, 30.0),
    ]:
        rows.append(dict(id=pid, team_id=tid, pos_id=3, now_cost=cost,
                         xpts_total=xpts, minutes=1500, status="a"))
        pid += 1

    # FWD: 5 players, premium options
    for tid, cost, xpts in [
        (1, 145, 52.0), (3, 90, 35.0), (4, 75, 28.0), (5, 110, 42.0),
        (2, 55, 22.0),
    ]:
        rows.append(dict(id=pid, team_id=tid, pos_id=4, now_cost=cost,
                         xpts_total=xpts, minutes=1500, status="a"))
        pid += 1

    # Apply seed-based jitter so different seeds produce different optima.
    # Costs stay fixed (preserves feasibility); only xPts wobble.
    for row in rows:
        row["xpts_total"] = round(max(row["xpts_total"] + random.uniform(-3, 3), 0), 2)

    return pd.DataFrame(rows)


# ============================================================
# Brute force
# ============================================================
def best_xi_for_squad(squad_df, bench_cost_penalty):
    """Find the optimal XI within a fixed 15-man squad.

    For a fixed squad, the objective decomposes per-position:
      objective = XI_xpts - bench_cost * λ
                = XI_xpts + λ * XI_cost - λ * squad_cost

    For each valid formation (1 GK + 3-5 DEF + 2-5 MID + 1-3 FWD = 11),
    pick top-K per position by (xpts + λ * cost). Greedy is provably optimal
    here because the objective is separable per position.
    """
    sq = squad_df.copy()
    sq["score"] = sq["xpts_total"] + bench_cost_penalty * sq["now_cost"]
    by_pos = {p: sq[sq["pos_id"] == p].sort_values("score", ascending=False)
              for p in [1, 2, 3, 4]}
    squad_cost = sq["now_cost"].sum()

    best_obj = -float("inf")
    best_xi_ids = None
    best_formation = None

    for n_def in range(3, 6):
        for n_mid in range(2, 6):
            n_fwd = 11 - 1 - n_def - n_mid
            if not (1 <= n_fwd <= 3):
                continue
            if n_def > len(by_pos[2]) or n_mid > len(by_pos[3]) or n_fwd > len(by_pos[4]):
                continue

            xi_gk = by_pos[1].head(1)
            xi_def = by_pos[2].head(n_def)
            xi_mid = by_pos[3].head(n_mid)
            xi_fwd = by_pos[4].head(n_fwd)

            xi = pd.concat([xi_gk, xi_def, xi_mid, xi_fwd])
            xi_xpts = xi["xpts_total"].sum()
            xi_cost = xi["now_cost"].sum()
            obj = xi_xpts - (squad_cost - xi_cost) * bench_cost_penalty

            if obj > best_obj:
                best_obj = obj
                best_xi_ids = set(xi["id"].tolist())
                best_formation = (1, n_def, n_mid, n_fwd)

    return best_obj, best_xi_ids, best_formation


def brute_force_optimum(players_df, budget, bench_cost_penalty=0.10,
                        max_per_team=3, locked_ids=None, banned_ids=None):
    """Enumerate all valid 15-man squads and return the optimum."""
    locked_ids = locked_ids or set()
    banned_ids = banned_ids or set()

    # Drop banned
    pool = players_df[~players_df["id"].isin(banned_ids)]

    by_pos = {p: pool[pool["pos_id"] == p]["id"].tolist() for p in [1, 2, 3, 4]}
    pos_counts = {1: 2, 2: 5, 3: 5, 4: 3}

    best_obj = -float("inf")
    best_squad = None
    best_xi = None
    best_formation = None
    n_evaluated = 0
    n_skipped_budget = 0
    n_skipped_team = 0
    n_skipped_locked = 0

    cost_lookup = dict(zip(pool["id"], pool["now_cost"]))
    team_lookup = dict(zip(pool["id"], pool["team_id"]))

    for gk in combinations(by_pos[1], pos_counts[1]):
        for df in combinations(by_pos[2], pos_counts[2]):
            for md in combinations(by_pos[3], pos_counts[3]):
                for fw in combinations(by_pos[4], pos_counts[4]):
                    squad_ids = list(gk) + list(df) + list(md) + list(fw)

                    # Locked IDs must all appear
                    if locked_ids and not locked_ids.issubset(squad_ids):
                        n_skipped_locked += 1
                        continue

                    squad_cost = sum(cost_lookup[pid] for pid in squad_ids)
                    if squad_cost > budget:
                        n_skipped_budget += 1
                        continue

                    team_counts = {}
                    for pid in squad_ids:
                        t = team_lookup[pid]
                        team_counts[t] = team_counts.get(t, 0) + 1
                    if max(team_counts.values()) > max_per_team:
                        n_skipped_team += 1
                        continue

                    n_evaluated += 1
                    squad_df = pool[pool["id"].isin(squad_ids)]
                    obj, xi_ids, formation = best_xi_for_squad(squad_df, bench_cost_penalty)

                    if obj > best_obj:
                        best_obj = obj
                        best_squad = set(squad_ids)
                        best_xi = xi_ids
                        best_formation = formation

    return {
        "objective": best_obj,
        "squad_ids": best_squad,
        "xi_ids": best_xi,
        "formation": best_formation,
        "n_evaluated": n_evaluated,
        "n_skipped_budget": n_skipped_budget,
        "n_skipped_team": n_skipped_team,
        "n_skipped_locked": n_skipped_locked,
    }


# ============================================================
# Solver-side objective computation (mirror solve_optimal_squad's objective)
# ============================================================
def compute_solver_objective(squad_df, bench_cost_penalty=0.10):
    """Recompute the objective from the solver's returned squad+XI.

    Mirrors the formulation in solve_optimal_squad:
        XI_xpts - bench_cost * λ
    """
    xi = squad_df[squad_df["is_xi"]]
    bench = squad_df[~squad_df["is_xi"]]
    xi_xpts = xi["xpts_total"].sum()
    bench_cost = bench["now_cost"].sum()
    return xi_xpts - bench_cost * bench_cost_penalty


# ============================================================
# Tests
# ============================================================
def _check_optimum(seed, budget, **solver_kwargs):
    """Run brute force + solver, assert objective values match.

    Also verifies that locked/banned constraints are honoured by the solver
    (not just the objective — the actual selection must respect the contracts).
    """
    pool = make_player_pool(seed=seed)
    print(f"\n--- seed={seed}, budget=GBP{budget/10:.1f}m ---")
    locked_ids = solver_kwargs.get("locked_ids") or set()
    banned_ids = solver_kwargs.get("banned_ids") or set()

    bf = brute_force_optimum(pool, budget=budget, **solver_kwargs)
    print(f"Brute force: evaluated {bf['n_evaluated']:,} squads "
          f"(skipped {bf['n_skipped_budget']:,} budget, "
          f"{bf['n_skipped_team']:,} team)")
    print(f"  objective = {bf['objective']:.3f}, formation = {bf['formation']}")

    if bf["objective"] == -float("inf"):
        print("  no feasible squad — skipping solver comparison")
        return None, None

    squad_df, err = solve_optimal_squad(pool, xpts_col="xpts_total",
                                         budget=budget, **solver_kwargs)
    assert squad_df is not None, f"Solver failed: {err}"
    solver_obj = compute_solver_objective(squad_df)
    print(f"Solver:      objective = {solver_obj:.3f}")

    diff = abs(solver_obj - bf["objective"])
    print(f"delta = {diff:.6f}")

    assert diff < 1e-3, (
        f"Solver objective ({solver_obj:.4f}) differs from brute-force "
        f"optimum ({bf['objective']:.4f}) by {diff:.4f}. "
        f"Solver squad: {sorted(squad_df['id'].tolist())}, "
        f"BF squad: {sorted(bf['squad_ids'])}"
    )

    # Also verify the solver's selection IS feasible under brute-force rules
    solver_ids = set(squad_df["id"].tolist())
    solver_squad_cost = squad_df["now_cost"].sum()
    assert solver_squad_cost <= budget, f"Solver picked over budget: {solver_squad_cost} > {budget}"
    team_counts = squad_df["team_id"].value_counts()
    assert (team_counts <= 3).all(), f"Solver violated max-3-per-team: {team_counts.to_dict()}"
    pos_counts = squad_df["pos_id"].value_counts().to_dict()
    assert pos_counts.get(1, 0) == 2, f"GK count wrong: {pos_counts}"
    assert pos_counts.get(2, 0) == 5, f"DEF count wrong: {pos_counts}"
    assert pos_counts.get(3, 0) == 5, f"MID count wrong: {pos_counts}"
    assert pos_counts.get(4, 0) == 3, f"FWD count wrong: {pos_counts}"
    assert squad_df["is_xi"].sum() == 11, f"XI count wrong: {squad_df['is_xi'].sum()}"

    # Lock/ban contract checks
    if locked_ids:
        missing = locked_ids - solver_ids
        assert not missing, f"Solver dropped locked players: {missing}"
    if banned_ids:
        intruders = banned_ids & solver_ids
        assert not intruders, f"Solver picked banned players: {intruders}"

    return solver_obj, bf["objective"]


def test_basic_optimum():
    """Solver finds true optimum on default pool (GBP 100m budget)."""
    _check_optimum(seed=42, budget=1000)


def test_tighter_budget():
    """Solver still finds optimum under a tighter budget (GBP 95m)."""
    _check_optimum(seed=42, budget=950)


def test_different_seed():
    """Different player distribution; solver still optimal."""
    _check_optimum(seed=7, budget=1000)


def test_locked_player():
    """Locked player must appear and solver still finds optimum-with-lock.
    Pid 15: premium MID (T5, GBP 13.0m, ~55 xPts) — expensive but leaves room."""
    locked = {15}
    _check_optimum(seed=42, budget=1000, locked_ids=locked)


def test_banned_player():
    """Banned player must not appear; solver finds optimum without them.
    Pid 18: premium FWD (T1, GBP 14.5m, ~52 xPts) — banning forces a different optimum."""
    banned = {18}
    _check_optimum(seed=42, budget=1000, banned_ids=banned)


# ============================================================
# CLI runner
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Solver correctness validation: brute-force vs MILP")
    print("=" * 70)

    tests = [
        ("basic", test_basic_optimum),
        ("tighter budget", test_tighter_budget),
        ("different seed", test_different_seed),
        ("locked player", test_locked_player),
        ("banned player", test_banned_player),
    ]

    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"\n[PASS] {name}")
        except AssertionError as e:
            failures.append((name, str(e)))
            print(f"\n[FAIL] {name}: {e}")

    print("\n" + "=" * 70)
    if failures:
        print(f"{len(failures)} of {len(tests)} test(s) FAILED")
        for name, msg in failures:
            print(f"  - {name}: {msg}")
        sys.exit(1)
    else:
        print(f"All {len(tests)} tests PASSED — solver matches brute-force optimum")
