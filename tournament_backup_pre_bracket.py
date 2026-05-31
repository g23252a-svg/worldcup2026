"""
2026 FIFA World Cup knockout-stage engine.

Depends on functions already defined in app.py:
  - simulate_match(home_row, away_row, team_ratings, seed=None) -> (gh, ga, meta)
  - get_team_elo(row, team_ratings) -> (elo, src)
  - expected_goals_from_elo(eA, eB, base_goals=2.6) -> (lamA, lamB)
  - simulate_group_once(group_letter, df_teams, team_ratings, seed=None) -> (df_table, df_matches)

and on annexc_data.third_place_assignment.
"""

import numpy as np
import pandas as pd

from annexc_data import third_place_assignment

GROUP_LETTERS = list("ABCDEFGHIJKL")

# Round-of-32 match template (Wikipedia / FIFA official, matches 73-88).
# Each entry: match_no, home_spec, away_spec
#   ("W","E")  = winner of group E
#   ("R","C")  = runner-up of group C
#   ("3","E")  = 3rd-placed team assigned to the group-E winner slot (resolved via Annex C)
# For 3rd-place slots we store which WINNER slot they belong to, then resolve.
R32_TEMPLATE = [
    (73, ("R", "A"), ("R", "B")),
    (74, ("W", "E"), ("3", "E")),   # Winner E vs best-3rd (slot E)
    (75, ("W", "F"), ("R", "C")),
    (76, ("W", "C"), ("R", "F")),
    (77, ("W", "I"), ("3", "I")),   # Winner I vs best-3rd (slot I)
    (78, ("R", "E"), ("R", "I")),
    (79, ("W", "A"), ("3", "A")),   # Winner A vs best-3rd (slot A)
    (80, ("W", "L"), ("3", "L")),   # Winner L vs best-3rd (slot L)
    (81, ("W", "D"), ("3", "D")),   # Winner D vs best-3rd (slot D)
    (82, ("W", "G"), ("3", "G")),   # Winner G vs best-3rd (slot G)
    (83, ("R", "K"), ("R", "L")),
    (84, ("W", "H"), ("R", "J")),
    (85, ("W", "B"), ("3", "B")),   # Winner B vs best-3rd (slot B)
    (86, ("W", "J"), ("R", "H")),
    (87, ("W", "K"), ("3", "K")),   # Winner K vs best-3rd (slot K)
    (88, ("R", "D"), ("R", "G")),
]

# Round-of-16 and later: winner-of-match edges (Wikipedia bracket).
R16_EDGES = [
    (89, 74, 77),
    (90, 73, 75),
    (91, 76, 78),
    (92, 79, 80),
    (93, 83, 84),
    (94, 81, 82),
    (95, 86, 88),
    (96, 85, 87),
]
QF_EDGES = [
    (97, 89, 90),
    (98, 93, 94),
    (99, 91, 92),
    (100, 95, 96),
]
SF_EDGES = [
    (101, 97, 98),
    (102, 99, 100),
]
FINAL_NO = 104
THIRD_PLACE_NO = 103


def rank_third_place_teams(group_results):
    """
    group_results: dict group_letter -> {"1": code1, "2": code2, "3": rec3}
        where rec3 is a dict with keys PTS, GD, GF, team_code, team_name_ko, group.
    Returns: list of the 8 best 3rd-place records (sorted), and the set of 8 group letters.
    Tiebreak: PTS, GD, GF (FIFA). Stable by group letter for full determinism.
    """
    thirds = []
    for g in GROUP_LETTERS:
        rec = group_results[g]["3"]
        thirds.append(rec)
    thirds_sorted = sorted(
        thirds,
        key=lambda r: (r["PTS"], r["GD"], r["GF"], -ord(r["group"])),
        reverse=True,
    )
    best8 = thirds_sorted[:8]
    qualifying_groups = sorted([r["group"] for r in best8])
    return best8, qualifying_groups


def build_r32_matchups(group_results, best8, qualifying_groups):
    """
    Returns list of dicts: {match_no, home_code, away_code, home_name, away_name}
    Resolves the 8 third-place slots via the Annex C table.
    """
    assignment = third_place_assignment(qualifying_groups)  # slot-group -> 3rd-group
    if assignment is None:
        raise ValueError(f"No Annex C entry for groups {qualifying_groups}")

    # map 3rd-group letter -> its record
    third_by_group = {r["group"]: r for r in best8}

    def resolve(spec):
        kind, g = spec
        if kind == "W":
            return group_results[g]["1"]
        if kind == "R":
            return group_results[g]["2"]
        if kind == "3":
            # g here is the WINNER slot; find which 3rd-group fills it
            third_group = assignment[g]
            return third_by_group[third_group]["team_code"]
        raise ValueError(spec)

    matches = []
    for no, home_spec, away_spec in R32_TEMPLATE:
        hc = resolve(home_spec)
        ac = resolve(away_spec)
        matches.append({"match_no": no, "home_code": hc, "away_code": ac})
    return matches, assignment


def _resolve_ko_winner(home_code, away_code, df_teams, team_ratings, simulate_match):
    """
    One knockout match. If drawn in 'regulation' (Poisson), resolve by an
    Elo-weighted coin flip standing in for extra-time + penalties.
    Returns winner_code (and loser_code).
    """
    home_row = df_teams[df_teams["team_code"] == home_code].iloc[0]
    away_row = df_teams[df_teams["team_code"] == away_code].iloc[0]
    gh, ga, meta = simulate_match(home_row, away_row, team_ratings)
    if gh > ga:
        return home_code, away_code
    if ga > gh:
        return away_code, home_code
    # Draw -> ET/penalties. Win prob from Elo (logistic), slight pull toward 0.5
    eh = meta["elo_home"]
    ea = meta["elo_away"]
    p_home = 1.0 / (1.0 + 10.0 ** (-(eh - ea) / 400.0))
    # dampen toward coin-flip a bit (penalties are noisy)
    p_home = 0.5 + 0.6 * (p_home - 0.5)
    if np.random.random() < p_home:
        return home_code, away_code
    return away_code, home_code


def simulate_knockout_once(group_results, df_teams, team_ratings, simulate_match):
    """
    Runs R32 -> Final once. Returns dict with:
      champion, runner_up, third, fourth, results(match_no->winner), r32(list)
    group_results must already be computed (1/2/3 per group).
    """
    best8, qgroups = rank_third_place_teams(group_results)
    r32, assignment = build_r32_matchups(group_results, best8, qgroups)

    winners = {}   # match_no -> winner_code
    losers = {}

    # R32
    for m in r32:
        w, l = _resolve_ko_winner(
            m["home_code"], m["away_code"], df_teams, team_ratings, simulate_match
        )
        winners[m["match_no"]] = w
        losers[m["match_no"]] = l

    # R16, QF, SF
    for no, a, b in R16_EDGES + QF_EDGES + SF_EDGES:
        w, l = _resolve_ko_winner(
            winners[a], winners[b], df_teams, team_ratings, simulate_match
        )
        winners[no] = w
        losers[no] = l

    # 3rd place: losers of SF (101, 102)
    w3, l3 = _resolve_ko_winner(
        losers[101], losers[102], df_teams, team_ratings, simulate_match
    )
    winners[THIRD_PLACE_NO] = w3
    losers[THIRD_PLACE_NO] = l3

    # Final: winners of SF
    champ, runup = _resolve_ko_winner(
        winners[101], winners[102], df_teams, team_ratings, simulate_match
    )
    winners[FINAL_NO] = champ
    losers[FINAL_NO] = runup

    return {
        "champion": champ,
        "runner_up": runup,
        "third": w3,
        "fourth": l3,
        "qualifying_groups": qgroups,
        "r32": r32,
        "assignment": assignment,
        "winners": winners,
    }


def _group_results_from_tables(df_teams, team_ratings, simulate_group_once, seed=None):
    """
    Simulate all 12 groups once, return group_results dict
    group_letter -> {"1":code,"2":code,"3":rec3, "table":df}
    """
    if seed is not None:
        np.random.seed(seed)
    out = {}
    for g in GROUP_LETTERS:
        df_table, _ = simulate_group_once(g, df_teams, team_ratings)
        if df_table.empty or len(df_table) < 3:
            raise ValueError(f"Group {g} produced <3 teams")
        row1 = df_table.iloc[0]
        row2 = df_table.iloc[1]
        row3 = df_table.iloc[2]
        out[g] = {
            "1": row1["team_code"],
            "2": row2["team_code"],
            "3": {
                "team_code": row3["team_code"],
                "team_name_ko": row3.get("team_name_ko", row3["team_code"]),
                "group": g,
                "PTS": int(row3["PTS"]),
                "GD": int(row3["GD"]),
                "GF": int(row3["GF"]),
            },
            "table": df_table,
        }
    return out


def simulate_tournament_many(
    df_teams,
    team_ratings,
    simulate_group_once,
    simulate_match,
    n_sim=2000,
    seed=None,
):
    """
    Full Monte-Carlo: group stage + knockout, n_sim times.
    Returns DataFrame with per-team probabilities:
      champion %, final %, semi %, quarter %, r16 %, r32 %
    plus counts.
    """
    if seed is not None:
        np.random.seed(seed)

    codes = df_teams["team_code"].tolist()
    name_by_code = dict(zip(df_teams["team_code"], df_teams["team_name_ko"]))
    group_by_code = dict(zip(df_teams["team_code"], df_teams["group_letter"]))

    champ = {c: 0 for c in codes}
    final = {c: 0 for c in codes}
    semi = {c: 0 for c in codes}
    quarter = {c: 0 for c in codes}
    r16 = {c: 0 for c in codes}
    r32 = {c: 0 for c in codes}

    for _ in range(n_sim):
        gr = _group_results_from_tables(df_teams, team_ratings, simulate_group_once)
        res = simulate_knockout_once(gr, df_teams, team_ratings, simulate_match)
        winners = res["winners"]

        # R32 participants
        for m in res["r32"]:
            r32[m["home_code"]] += 1
            r32[m["away_code"]] += 1
        # R16 participants = winners of R32 (matches 73-88)
        for no in range(73, 89):
            r16[winners[no]] += 1
        # QF participants = winners of R16 (89-96)
        for no in range(89, 97):
            quarter[winners[no]] += 1
        # SF participants = winners of QF (97-100)
        for no in range(97, 101):
            semi[winners[no]] += 1
        # Final participants = winners of SF (101,102)
        final[winners[101]] += 1
        final[winners[102]] += 1
        # Champion
        champ[res["champion"]] += 1

    rows = []
    for c in codes:
        rows.append({
            "team_code": c,
            "team": name_by_code.get(c, c),
            "group": group_by_code.get(c, ""),
            "champion_pct": 100.0 * champ[c] / n_sim,
            "final_pct": 100.0 * final[c] / n_sim,
            "semi_pct": 100.0 * semi[c] / n_sim,
            "quarter_pct": 100.0 * quarter[c] / n_sim,
            "r16_pct": 100.0 * r16[c] / n_sim,
            "r32_pct": 100.0 * r32[c] / n_sim,
        })
    df = pd.DataFrame(rows).sort_values("champion_pct", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df


# =====================================================================
# Optimized fast path: precompute Elo per team once, then run group +
# knockout entirely on dict/array lookups (no per-match DataFrame ops).
# =====================================================================

def precompute_team_elo(df_teams, team_ratings, get_team_elo):
    """Return dict team_code -> elo (float), computed once."""
    elo = {}
    for _, row in df_teams.iterrows():
        e, _src = get_team_elo(row, team_ratings)
        elo[row["team_code"]] = float(e)
    return elo


# group_letter -> list of team_codes by group_pos (1..4)
def build_group_members(df_teams):
    members = {}
    for g in GROUP_LETTERS:
        sub = df_teams[df_teams["group_letter"] == g].sort_values("group_pos")
        members[g] = sub["team_code"].tolist()
    return members


# Group fixture template by position (matchday, posA, posB)
_GROUP_FIX = [(1, 1, 2), (1, 3, 4), (2, 1, 3), (2, 4, 2), (3, 4, 1), (3, 2, 3)]


def _fast_poisson_goals(lamA, lamB):
    return np.random.poisson(lamA), np.random.poisson(lamB)


def _fast_match_goals(eh, ea, base=2.6):
    diff = eh - ea
    pa = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
    return np.random.poisson(base * pa), np.random.poisson(base * (1.0 - pa))


def _fast_ko_winner(ch, ca, elo):
    eh, ea = elo[ch], elo[ca]
    gh, ga = _fast_match_goals(eh, ea)
    if gh > ga:
        return ch, ca
    if ga > gh:
        return ca, ch
    p = 1.0 / (1.0 + 10.0 ** (-(eh - ea) / 400.0))
    p = 0.5 + 0.6 * (p - 0.5)
    return (ch, ca) if np.random.random() < p else (ca, ch)


def _fast_group_once(members, elo):
    """Return group_results dict for all 12 groups, one simulation."""
    out = {}
    for g in GROUP_LETTERS:
        codes = members[g]  # [pos1,pos2,pos3,pos4]
        tab = {c: {"PTS": 0, "GF": 0, "GA": 0} for c in codes}
        for _md, pa, pb in _GROUP_FIX:
            ch, ca = codes[pa - 1], codes[pb - 1]
            gh, ga = _fast_match_goals(elo[ch], elo[ca])
            tab[ch]["GF"] += gh; tab[ch]["GA"] += ga
            tab[ca]["GF"] += ga; tab[ca]["GA"] += gh
            if gh > ga:
                tab[ch]["PTS"] += 3
            elif ga > gh:
                tab[ca]["PTS"] += 3
            else:
                tab[ch]["PTS"] += 1; tab[ca]["PTS"] += 1
        ranked = sorted(
            codes,
            key=lambda c: (
                tab[c]["PTS"],
                tab[c]["GF"] - tab[c]["GA"],
                tab[c]["GF"],
                -elo[c] * 0 + np.random.random(),  # random tiebreak (lots)
            ),
            reverse=True,
        )
        r3 = ranked[2]
        out[g] = {
            "1": ranked[0],
            "2": ranked[1],
            "3": {
                "team_code": r3,
                "group": g,
                "PTS": tab[r3]["PTS"],
                "GD": tab[r3]["GF"] - tab[r3]["GA"],
                "GF": tab[r3]["GF"],
            },
        }
    return out


def _fast_knockout_once(group_results, elo):
    best8, qgroups = rank_third_place_teams(group_results)
    r32, _assign = build_r32_matchups(group_results, best8, qgroups)
    winners = {}
    losers = {}
    for m in r32:
        w, l = _fast_ko_winner(m["home_code"], m["away_code"], elo)
        winners[m["match_no"]] = w; losers[m["match_no"]] = l
    for no, a, b in R16_EDGES + QF_EDGES + SF_EDGES:
        w, l = _fast_ko_winner(winners[a], winners[b], elo)
        winners[no] = w; losers[no] = l
    w3, l3 = _fast_ko_winner(losers[101], losers[102], elo)
    winners[THIRD_PLACE_NO] = w3; losers[THIRD_PLACE_NO] = l3
    champ, runup = _fast_ko_winner(winners[101], winners[102], elo)
    winners[FINAL_NO] = champ; losers[FINAL_NO] = runup
    return r32, winners, champ, runup, w3, l3


def simulate_tournament_fast(
    df_teams, team_ratings, get_team_elo, n_sim=2000, seed=None
):
    """Fast Monte-Carlo. Same output schema as simulate_tournament_many."""
    if seed is not None:
        np.random.seed(seed)
    elo = precompute_team_elo(df_teams, team_ratings, get_team_elo)
    members = build_group_members(df_teams)

    codes = df_teams["team_code"].tolist()
    name_by_code = dict(zip(df_teams["team_code"], df_teams["team_name_ko"]))
    group_by_code = dict(zip(df_teams["team_code"], df_teams["group_letter"]))

    champ = {c: 0 for c in codes}; final = {c: 0 for c in codes}
    semi = {c: 0 for c in codes}; quarter = {c: 0 for c in codes}
    r16 = {c: 0 for c in codes}; r32c = {c: 0 for c in codes}

    for _ in range(n_sim):
        gr = _fast_group_once(members, elo)
        r32, winners, ch, ru, w3, l3 = _fast_knockout_once(gr, elo)
        for m in r32:
            r32c[m["home_code"]] += 1; r32c[m["away_code"]] += 1
        for no in range(73, 89):
            r16[winners[no]] += 1
        for no in range(89, 97):
            quarter[winners[no]] += 1
        for no in range(97, 101):
            semi[winners[no]] += 1
        final[winners[101]] += 1; final[winners[102]] += 1
        champ[ch] += 1

    rows = []
    for c in codes:
        rows.append({
            "team_code": c, "team": name_by_code.get(c, c),
            "group": group_by_code.get(c, ""),
            "champion_pct": 100.0 * champ[c] / n_sim,
            "final_pct": 100.0 * final[c] / n_sim,
            "semi_pct": 100.0 * semi[c] / n_sim,
            "quarter_pct": 100.0 * quarter[c] / n_sim,
            "r16_pct": 100.0 * r16[c] / n_sim,
            "r32_pct": 100.0 * r32c[c] / n_sim,
        })
    df = pd.DataFrame(rows).sort_values("champion_pct", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df
