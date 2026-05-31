# app.py
import streamlit as st
import pandas as pd
import numpy as np


# =========================
# 데이터 로딩
# =========================

def assign_group_pos(group_letter: str, seeding_pot: int) -> int:
    """
    2026 월드컵 포트 규정을 반영해서
    각 팀의 조 내 포지션(1~4번 자리)을 계산한다.
    - 포트 1: 항상 1번 자리 (A1, B1, ..., L1)
    - 포트 2: A3 B4 C2 D3 E4 F2 G3 H4 I2 J3 K4 L2
    - 포트 3: A2 B3 C4 D2 E3 F4 G2 H3 I4 J2 K3 L4
    - 포트 4: A4 B2 C3 D4 E2 F3 G4 H2 I3 J4 K2 L3
    """
    g = str(group_letter).upper()
    p = int(seeding_pot)

    if p == 1:
        return 1  # 개최국/포트1은 무조건 1번 슬롯

    block1 = {"A", "D", "G", "J"}
    block2 = {"B", "E", "H", "K"}
    block3 = {"C", "F", "I", "L"}

    if p == 2:
        if g in block1:
            return 3
        if g in block2:
            return 4
        if g in block3:
            return 2
    elif p == 3:
        if g in block1:
            return 2
        if g in block2:
            return 3
        if g in block3:
            return 4
    elif p == 4:
        if g in block1:
            return 4
        if g in block2:
            return 2
        if g in block3:
            return 3

    raise ValueError(f"예상치 못한 group/pot 조합: {group_letter}/{seeding_pot}")


@st.cache_data
def load_teams():
    df = pd.read_csv("data/teams_2026.csv")

    # 조 내 포지션(1~4번) 계산
    df["group_pos"] = df.apply(
        lambda r: assign_group_pos(r["group_letter"], r["seeding_pot"]),
        axis=1,
    )

    # 보기 편하게 "A1, B3" 같은 슬롯 문자열도 추가
    df["slot"] = df["group_letter"] + df["group_pos"].astype(str)

    # 실제 슬롯 기준으로 정렬
    df = df.sort_values(["group_letter", "group_pos", "team_code"])
    return df


@st.cache_data
def load_players():
    try:
        df = pd.read_csv("data/players_2026.csv")
    except FileNotFoundError:
        return pd.DataFrame()

    if not df.empty:
        df = df.sort_values(
            ["team_code", "position", "is_starting", "player_name_en"],
            ascending=[True, True, False, True],
        )
    return df


# =========================
# 팀 레이팅 계산 로직
# =========================
def compute_player_overall(row: pd.Series) -> float:
    """
    선수 개별 종합 능력치(0~100)를 하나로 압축.
    - GK는 attack/defense 대신 gk 스탯 중심으로 평가
      (안 그러면 골키퍼의 낮은 공격력이 팀 평균을 왜곡함)
    """
    is_gk = str(row.get("position", "")).upper() == "GK" or int(row.get("gk", 0) or 0) > 0
    if is_gk:
        gk = float(row.get("gk", 0) or 0)
        return (
            gk * 0.55
            + row["defense"] * 0.15
            + row["passing"] * 0.10
            + row["physical"] * 0.10
            + row["mental"] * 0.10
        )
    return (
        row["attack"] * 0.35
        + row["defense"] * 0.30
        + row["passing"] * 0.15
        + row["physical"] * 0.10
        + row["mental"] * 0.10
    )


def build_team_ratings(df_players: pd.DataFrame, use_starting_only: bool = True):
    """
    players_2026.csv 기반 팀별 레이팅 계산
    - 기본: 선발 11명 평균으로 팀 능력치 산출
    - 선수 데이터 없는 팀은 결과 dict에 없음
    """
    ratings: dict[str, dict[str, float]] = {}

    if df_players.empty:
        return ratings

    df = df_players.copy()
    df["player_overall"] = df.apply(compute_player_overall, axis=1)

    for team_code, grp in df.groupby("team_code"):
        g = grp
        if use_starting_only:
            starters = g[g["is_starting"] == 1]
            if len(starters) >= 8:
                g = starters

        team_attack = g["attack"].mean()
        team_defense = g["defense"].mean()
        team_passing = g["passing"].mean()
        team_physical = g["physical"].mean()
        team_mental = g["mental"].mean()
        team_overall = g["player_overall"].mean()

        ratings[team_code] = {
            "overall": float(team_overall),
            "attack": float(team_attack),
            "defense": float(team_defense),
            "passing": float(team_passing),
            "physical": float(team_physical),
            "mental": float(team_mental),
        }

    return ratings


@st.cache_data
def load_team_elo() -> dict[str, dict]:
    """
    data/team_elo_2026.csv 에서 48팀 실제 Elo 레이팅을 로딩.
    - World Football Elo (2026-01 기준) + 추정치/플레이오프 평균
    - 반환: {team_code: {"elo": float, "elo_source": str}}
    """
    try:
        df = pd.read_csv("data/team_elo_2026.csv")
    except FileNotFoundError:
        return {}

    out: dict[str, dict] = {}
    for _, r in df.iterrows():
        out[str(r["team_code"])] = {
            "elo": float(r["elo"]),
            "elo_source": str(r.get("elo_source", "csv")),
        }
    return out


def pot_to_rating(pot: int) -> float:
    """
    포트 번호(1~4)를 Elo 비슷한 레이팅으로 변환
    - Elo 테이블에도 없는 팀용 최후 fallback
    """
    pot = int(pot)
    base = {
        1: 1850.0,
        2: 1780.0,
        3: 1700.0,
        4: 1620.0,
    }
    return base.get(pot, 1700.0)


def overall_to_elo(overall: float) -> float:
    """
    선수 평균 overall(0~100)을 Elo 스케일로 변환.
    이 데이터셋의 실제 squad overall 평균(약 73.5)을 1800에 앵커링하고
    1점당 약 45 Elo 기울기로 강팀/약팀 격차를 표현.
    - 73.5 → 1800, 78 → 2000, 70 → 1640 근처
    """
    return 1800.0 + (overall - 73.5) * 45.0


# 선수 데이터가 있는 팀에서 (선수기반 Elo) vs (실측 Elo) 블렌딩 비율
# 0.0 = 실측 Elo 100% / 1.0 = 선수 기반 100%
PLAYER_ELO_BLEND = 0.45

# 실측 Elo 테이블 전역 캐시 (main()에서 채움). 시뮬 함수 시그니처를
# 바꾸지 않고도 get_team_elo가 접근할 수 있게 한다.
_TEAM_ELO_CACHE: dict[str, dict] = {}


def get_team_elo(
    row_team: pd.Series,
    team_ratings: dict,
    team_elo: dict | None = None,
) -> tuple[float, str]:
    """
    팀 최종 Elo 레이팅 + 소스.
    우선순위:
      1) 실측 Elo 테이블에 있고 선수 데이터도 있으면 → 블렌딩
      2) 실측 Elo만 있으면 → 실측 Elo
      3) 선수 데이터만 있으면 → 선수 기반
      4) 둘 다 없으면 → 포트 fallback
    """
    code = row_team["team_code"]
    pot = row_team["seeding_pot"]
    team_elo = team_elo or _TEAM_ELO_CACHE

    base_elo = team_elo.get(code, {}).get("elo") if code in team_elo else None
    has_players = code in team_ratings

    if base_elo is not None and has_players:
        player_elo = overall_to_elo(team_ratings[code]["overall"])
        elo = (1 - PLAYER_ELO_BLEND) * base_elo + PLAYER_ELO_BLEND * player_elo
        source = "elo+players"
    elif base_elo is not None:
        elo = base_elo
        source = "elo_csv"
    elif has_players:
        elo = overall_to_elo(team_ratings[code]["overall"])
        source = "players_csv"
    else:
        elo = pot_to_rating(pot)
        source = "seeding_pot"

    return float(elo), source


# =========================
# 경기 시뮬레이션
# =========================
def expected_goals_from_elo(eA: float, eB: float, base_goals: float = 2.6):
    """
    Elo 레이팅으로 기대 득점 λA, λB 계산
    """
    diff = eA - eB
    pA = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
    pB = 1.0 - pA
    lamA = base_goals * pA
    lamB = base_goals * pB
    return float(lamA), float(lamB)


def simulate_match(
    home_row: pd.Series,
    away_row: pd.Series,
    team_ratings: dict,
    seed: int | None = None,
):
    """
    한 경기 시뮬레이션
    """
    if seed is not None:
        np.random.seed(seed)

    elo_home, src_home = get_team_elo(home_row, team_ratings)
    elo_away, src_away = get_team_elo(away_row, team_ratings)

    lam_home, lam_away = expected_goals_from_elo(elo_home, elo_away)

    goals_home = np.random.poisson(lam_home)
    goals_away = np.random.poisson(lam_away)

    meta = {
        "elo_home": elo_home,
        "elo_away": elo_away,
        "lam_home": lam_home,
        "lam_away": lam_away,
        "src_home": src_home,
        "src_away": src_away,
    }

    return int(goals_home), int(goals_away), meta


def simulate_many(
    home_row: pd.Series,
    away_row: pd.Series,
    team_ratings: dict,
    n_sim: int = 1000,
    seed: int | None = None,
):
    """
    같은 매치를 여러 번 시뮬레이션
    - 승/무/패 횟수
    - 평균 득점/실점
    - 스코어라인 분포
    """
    if seed is not None:
        np.random.seed(seed)

    home_wins = 0
    draws = 0
    away_wins = 0
    total_home_goals = 0
    total_away_goals = 0
    score_counts: dict[tuple[int, int], int] = {}

    # 예시 meta (설명용)
    _, _, meta_example = simulate_match(home_row, away_row, team_ratings)

    for _ in range(n_sim):
        gh, ga, _ = simulate_match(home_row, away_row, team_ratings)
        total_home_goals += gh
        total_away_goals += ga

        if gh > ga:
            home_wins += 1
        elif gh == ga:
            draws += 1
        else:
            away_wins += 1

        key = (gh, ga)
        score_counts[key] = score_counts.get(key, 0) + 1

    summary = {
        "n_sim": n_sim,
        "home_wins": home_wins,
        "draws": draws,
        "away_wins": away_wins,
        "avg_home_goals": total_home_goals / n_sim,
        "avg_away_goals": total_away_goals / n_sim,
        "score_counts": score_counts,
        "meta_example": meta_example,
    }
    return summary


# 조별리그 일정 템플릿
GROUP_FIXTURE_TEMPLATE = [
    # (matchday, home_pos, away_pos)
    (1, 1, 2),  # MD1: 1 vs 2
    (1, 3, 4),  # MD1: 3 vs 4
    (2, 1, 3),  # MD2: 1 vs 3
    (2, 4, 2),  # MD2: 4 vs 2
    (3, 4, 1),  # MD3: 4 vs 1
    (3, 2, 3),  # MD3: 2 vs 3
]


def build_group_fixtures_from_df(df_group: pd.DataFrame):
    """
    그룹 내 4개 팀을 기준 일정으로 변환
    - group_pos(1~4)에 따라 팀을 매핑한 뒤
      템플릿에 따라 6경기(3라운드)를 만든다.
    """
    grp = df_group.copy()

    # group_pos가 없으면 (혹시 모를 호환성) 옛 방식으로 fallback
    if "group_pos" not in grp.columns:
        grp_sorted = grp.sort_values(["seeding_pot", "team_code"])
        teams = grp_sorted["team_code"].tolist()
        if len(teams) != 4:
            return []
        t1, t2, t3, t4 = teams
        fixtures = [
            (1, t1, t2),
            (1, t3, t4),
            (2, t1, t3),
            (2, t2, t4),
            (3, t1, t4),
            (3, t2, t3),
        ]
        return fixtures

    # group_pos를 이용한 편성
    mapping = {int(row["group_pos"]): row["team_code"] for _, row in grp.iterrows()}
    if set(mapping.keys()) != {1, 2, 3, 4}:
        return []

    fixtures = []
    for md, hp, ap in GROUP_FIXTURE_TEMPLATE:
        home_code = mapping[hp]
        away_code = mapping[ap]
        fixtures.append((md, home_code, away_code))

    return fixtures


def simulate_group_once(
    group_letter: str,
    df_teams: pd.DataFrame,
    team_ratings: dict,
    seed: int | None = None,
):
    """
    특정 그룹(A~L) 한 번 시뮬레이션
    - 6경기 모두 돌려서 최종 순위표 + 경기 결과 반환
    """
    df_group = df_teams[df_teams["group_letter"] == group_letter].copy()
    if df_group.empty:
        return pd.DataFrame(), pd.DataFrame()

    fixtures = build_group_fixtures_from_df(df_group)
    if not fixtures:
        return pd.DataFrame(), pd.DataFrame()

    # 초기 테이블
    table = {}
    for _, row in df_group.iterrows():
        code = row["team_code"]
        table[code] = {
            "team_code": code,
            "team_name_ko": row["team_name_ko"],
            "P": 0,
            "W": 0,
            "D": 0,
            "L": 0,
            "GF": 0,
            "GA": 0,
            "GD": 0,
            "PTS": 0,
        }

    if seed is not None:
        np.random.seed(seed)

    match_rows = []

    for md, home_code, away_code in fixtures:
        home_row = df_group[df_group["team_code"] == home_code].iloc[0]
        away_row = df_group[df_group["team_code"] == away_code].iloc[0]

        gh, ga, _ = simulate_match(home_row, away_row, team_ratings)

        th = table[home_code]
        ta = table[away_code]

        th["P"] += 1
        ta["P"] += 1

        th["GF"] += gh
        th["GA"] += ga
        ta["GF"] += ga
        ta["GA"] += gh

        if gh > ga:
            th["W"] += 1
            ta["L"] += 1
            th["PTS"] += 3
        elif gh < ga:
            ta["W"] += 1
            th["L"] += 1
            ta["PTS"] += 3
        else:
            th["D"] += 1
            ta["D"] += 1
            th["PTS"] += 1
            ta["PTS"] += 1

        match_rows.append(
            {
                "matchday": md,
                "home_team": home_row["team_name_ko"],
                "home_code": home_code,
                "away_team": away_row["team_name_ko"],
                "away_code": away_code,
                "home_goals": gh,
                "away_goals": ga,
                "score": f"{gh}-{ga}",
            }
        )

    # GD 계산
    for rec in table.values():
        rec["GD"] = rec["GF"] - rec["GA"]

    df_table = pd.DataFrame(table.values())
    df_table = df_table.sort_values(
        ["PTS", "GD", "GF"], ascending=[False, False, False]
    ).reset_index(drop=True)
    df_table.insert(0, "Rank", df_table.index + 1)

    df_matches = pd.DataFrame(match_rows).sort_values(["matchday", "home_team"])

    return df_table, df_matches


def simulate_group_many(
    group_letter: str,
    df_teams: pd.DataFrame,
    team_ratings: dict,
    n_sim: int = 1000,
    seed: int | None = None,
):
    """
    특정 그룹(A~L)을 여러 번 시뮬레이션하여
    - 각 팀의 1위/2위/3위/4위 확률
    - 평균 승점, 평균 득실차, 평균 득점
    을 계산한다.
    """
    df_group = df_teams[df_teams["group_letter"] == group_letter].copy()
    if df_group.empty:
        return pd.DataFrame()

    team_codes = df_group["team_code"].tolist()

    # 통계 초기화
    stats = {
        code: {
            "team_code": code,
            "team_name_ko": df_group[df_group["team_code"] == code]["team_name_ko"].iloc[0],
            "cnt_rank1": 0,
            "cnt_rank2": 0,
            "cnt_rank3": 0,
            "cnt_rank4": 0,
            "sum_pts": 0.0,
            "sum_gd": 0.0,
            "sum_gf": 0.0,
        }
        for code in team_codes
    }

    if seed is not None:
        np.random.seed(seed)

    for _ in range(n_sim):
        # seed=None으로 넘겨서 매번 다른 난수 사용
        df_table, _ = simulate_group_once(group_letter, df_teams, team_ratings, seed=None)
        if df_table.empty:
            continue

        for _, row in df_table.iterrows():
            code = row["team_code"]
            rec = stats[code]

            rank = int(row["Rank"])
            if rank == 1:
                rec["cnt_rank1"] += 1
            elif rank == 2:
                rec["cnt_rank2"] += 1
            elif rank == 3:
                rec["cnt_rank3"] += 1
            elif rank == 4:
                rec["cnt_rank4"] += 1

            rec["sum_pts"] += float(row["PTS"])
            rec["sum_gd"] += float(row["GD"])
            rec["sum_gf"] += float(row["GF"])

    # 결과 DataFrame으로 변환
    rows = []
    for code, rec in stats.items():
        rows.append(
            {
                "team_code": code,
                "team_name_ko": rec["team_name_ko"],
                "P1(1위%)": rec["cnt_rank1"] / n_sim * 100,
                "P2(2위%)": rec["cnt_rank2"] / n_sim * 100,
                "P3(3위%)": rec["cnt_rank3"] / n_sim * 100,
                "P4(4위%)": rec["cnt_rank4"] / n_sim * 100,
                "avg_PTS": rec["sum_pts"] / n_sim,
                "avg_GD": rec["sum_gd"] / n_sim,
                "avg_GF": rec["sum_gf"] / n_sim,
            }
        )

    df_stats = pd.DataFrame(rows)

    # 1위 확률 → 2위 확률 → 평균 승점 순으로 정렬
    df_stats = df_stats.sort_values(
        ["P1(1위%)", "P2(2위%)", "avg_PTS"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return df_stats


def simulate_all_groups_many(
    df_teams: pd.DataFrame,
    team_ratings: dict,
    n_sim: int = 1000,
    seed: int | None = None,
):
    """
    전체 조별리그(A~L)를 동시에 여러 번 시뮬레이션하여
    - 각 팀의 1위/2위/3위/4위 확률
    - 평균 승점, 평균 득실차, 평균 득점
    - 1~2위 진출 확률(P_qual)
    을 한 번에 모아서 반환한다.
    """
    group_letters = sorted(df_teams["group_letter"].unique().tolist())
    all_stats = []

    if seed is not None:
        np.random.seed(seed)

    for gl in group_letters:
        df_stats = simulate_group_many(
            gl,
            df_teams,
            team_ratings,
            n_sim=n_sim,
            seed=None,  # 각 그룹은 독립적으로 랜덤
        )
        if df_stats.empty:
            continue

        # 어느 그룹인지 표시용
        df_stats.insert(0, "group_letter", gl)
        all_stats.append(df_stats)

    if not all_stats:
        return pd.DataFrame()

    df_all = pd.concat(all_stats, ignore_index=True)

    # 1~2위 합산 진출 확률 컬럼
    df_all["P_qual(1~2위%)"] = df_all["P1(1위%)"] + df_all["P2(2위%)"]

    return df_all


# =========================
# =========================
# Streamlit UI  (redesigned: tabbed layout + clean theme)
# =========================
import altair as alt


CONFED_LABEL = {
    "AFC": "아시아",
    "CAF": "아프리카",
    "CONCACAF": "북중미",
    "CONMEBOL": "남미",
    "OFC": "오세아니아",
    "UEFA": "유럽",
}


def _fmt_pct(v):
    return f"{v:.1f}"


def _elo_table(df_teams, team_ratings):
    """팀별 최종 Elo + 소스 테이블 (정렬용)."""
    rows = []
    for _, r in df_teams.iterrows():
        elo, src = get_team_elo(r, team_ratings)
        rows.append({
            "team_code": r["team_code"],
            "team": r["team_name_ko"],
            "group": r["group_letter"],
            "confed": r["confed"],
            "elo": round(elo, 0),
            "source": src,
        })
    return pd.DataFrame(rows).sort_values("elo", ascending=False).reset_index(drop=True)


def _hbar(df, x, y, title, color="#e0aa3e", fmt=".1f", height=None):
    """확률순으로 정렬된 가로 막대 (Altair)."""
    n = len(df)
    h = height or max(220, n * 30)
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=4, color=color)
        .encode(
            x=alt.X(f"{x}:Q", title=None, axis=alt.Axis(grid=True, format="~s")),
            y=alt.Y(f"{y}:N", sort="-x", title=None),
            tooltip=[
                alt.Tooltip(f"{y}:N", title="팀"),
                alt.Tooltip(f"{x}:Q", title=title, format=fmt),
            ],
        )
        .properties(height=h)
    )
    text = chart.mark_text(align="left", dx=4, color="#f2f2f2").encode(
        text=alt.Text(f"{x}:Q", format=fmt)
    )
    return chart + text


# -----------------------------------------------------------------
def render_overview(df_teams, team_ratings):
    st.subheader("대회 한눈에 보기")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("참가국", f"{len(df_teams)}")
    c2.metric("조", f"{df_teams['group_letter'].nunique()}")
    n_players = df_teams["team_code"].isin(team_ratings.keys()).sum()
    c3.metric("선수데이터 보유국", f"{n_players}")
    c4.metric("기간", "6/11 – 7/19")

    st.divider()
    st.caption("팀 강도 (블렌딩 Elo 기준, 상위 20개국)")
    elo_df = _elo_table(df_teams, team_ratings)
    st.altair_chart(
        _hbar(elo_df.head(20), "elo", "team", "Elo", color="#5b8def", fmt=".0f"),
        use_container_width=True,
    )

    with st.expander("전체 팀 Elo 표 보기"):
        show = elo_df.rename(columns={
            "team": "팀", "group": "조", "confed": "대륙",
            "elo": "Elo", "source": "산출방식",
        })
        show["대륙"] = show["대륙"].map(lambda c: CONFED_LABEL.get(c, c))
        st.dataframe(show.drop(columns=["team_code"]), use_container_width=True, hide_index=True)


# -----------------------------------------------------------------
def render_title_prediction(df_teams, team_ratings):
    st.subheader("🏆 우승 예측")
    st.caption(
        "조별리그(72경기) → 32강 → 16강 → 8강 → 4강 → 결승까지 대회 전체를 "
        "수천 번 시뮬레이션합니다. 32강 대진은 FIFA 공식 규정 + 베스트 3위 8팀 "
        "배정표(Annex C 495 시나리오)를 그대로 적용합니다."
    )

    col_a, col_b = st.columns([3, 1])
    n_tour = col_a.slider(
        "시뮬레이션 횟수", 1000, 50000, 10000, step=1000, key="n_tour_new"
    )
    col_b.write("")
    col_b.write("")
    run = col_b.button("🎲 시뮬레이션 실행", use_container_width=True, type="primary")

    if run:
        try:
            import tournament as TNM
        except Exception as e:
            st.error(f"tournament 모듈을 불러오지 못했습니다: {e}")
            return
        with st.spinner(f"{n_tour:,}회 대회 시뮬레이션 중..."):
            df_tour = TNM.simulate_tournament_fast(
                df_teams, team_ratings, get_team_elo, n_sim=n_tour
            )
        st.session_state["df_tour"] = df_tour
        st.session_state["n_tour_done"] = n_tour

    df_tour = st.session_state.get("df_tour")
    if df_tour is None:
        st.info("위 버튼을 눌러 우승 확률을 계산하세요. (1만 회 기준 약 2~3초)")
        return

    # ── 우승후보 TOP 3 카드 ──
    top3 = df_tour.head(3).reset_index(drop=True)
    medals = ["🥇", "🥈", "🥉"]
    cols = st.columns(3)
    for i, (_, row) in enumerate(top3.iterrows()):
        cols[i].metric(
            f"{medals[i]} {row['team']}",
            f"{row['champion_pct']:.1f}%",
            help=f"결승 {row['final_pct']:.1f}% · 4강 {row['semi_pct']:.1f}%",
        )

    st.divider()

    # ── 우승확률 가로 막대 (TOP 12, 정렬됨) ──
    left, right = st.columns([1, 1])
    with left:
        st.markdown("**우승 확률 TOP 12**")
        st.altair_chart(
            _hbar(df_tour.head(12)[["team", "champion_pct"]], "champion_pct", "team",
                  "우승%", color="#e0aa3e", fmt=".1f"),
            use_container_width=True,
        )
    with right:
        st.markdown("**결승 진출 확률 TOP 12**")
        st.altair_chart(
            _hbar(df_tour.head(12)[["team", "final_pct"]], "final_pct", "team",
                  "결승%", color="#9b6dff", fmt=".1f"),
            use_container_width=True,
        )

    # ── 전체 순위표 ──
    st.markdown("**단계별 진출 확률 (전체)**")
    ren = {
        "rank": "순위", "team": "팀", "group": "조",
        "champion_pct": "우승%", "final_pct": "결승%", "semi_pct": "4강%",
        "quarter_pct": "8강%", "r16_pct": "16강%", "r32_pct": "32강%",
    }
    cols_order = ["rank", "team", "group", "champion_pct", "final_pct",
                  "semi_pct", "quarter_pct", "r16_pct", "r32_pct"]
    show = df_tour[cols_order].rename(columns=ren)
    champ_max = max(float(show["우승%"].max()), 1.0)
    st.dataframe(
        show, use_container_width=True, hide_index=True, height=460,
        column_config={
            "순위": st.column_config.NumberColumn("순위", width="small"),
            "우승%": st.column_config.ProgressColumn(
                "우승%", format="%.1f", min_value=0.0, max_value=champ_max),
            "결승%": st.column_config.NumberColumn("결승%", format="%.1f"),
            "4강%": st.column_config.NumberColumn("4강%", format="%.1f"),
            "8강%": st.column_config.NumberColumn("8강%", format="%.1f"),
            "16강%": st.column_config.NumberColumn("16강%", format="%.1f"),
            "32강%": st.column_config.NumberColumn("32강%", format="%.1f"),
        },
    )

    # ── 한국/일본 ──
    kj = df_tour[df_tour["team_code"].isin(["KOR", "JPN"])]
    if not kj.empty:
        st.divider()
        st.markdown("**🇰🇷 한국 · 🇯🇵 일본**")
        kcols = st.columns(len(kj) * 3)
        idx = 0
        for _, row in kj.iterrows():
            kcols[idx].metric(f"{row['team']} 우승", f"{row['champion_pct']:.2f}%")
            kcols[idx + 1].metric("16강 진출", f"{row['r16_pct']:.1f}%")
            kcols[idx + 2].metric("32강 진출", f"{row['r32_pct']:.1f}%")
            idx += 3

    st.caption(
        f"{st.session_state.get('n_tour_done', 0):,}회 몬테카를로 결과. "
        "우승% 합=100, 결승% 합=200. 녹아웃 무승부는 Elo 가중 확률(연장+승부차기 대용)로 처리. "
        "선수 데이터가 없는 팀은 실측 Elo로 평가됩니다."
    )


# -----------------------------------------------------------------
def render_group_stage(df_teams, team_ratings):
    st.subheader("📊 조별리그 시뮬레이션")

    groups = sorted(df_teams["group_letter"].unique())
    c1, c2, c3 = st.columns([1, 1, 1])
    grp = c1.selectbox("조 선택", groups, key="grp_new")
    n_sim = c2.slider("시뮬 횟수", 100, 5000, 2000, step=100, key="grp_n_new")
    c3.write("")
    c3.write("")
    run = c3.button("실행", use_container_width=True, type="primary", key="grp_run")

    # 조 구성 표시
    members = df_teams[df_teams["group_letter"] == grp].sort_values("group_pos")
    chips = "  ".join(
        f"`{r['team_name_ko']}`" for _, r in members.iterrows()
    )
    st.markdown(f"**{grp}조:** {chips}")

    if run:
        with st.spinner(f"{grp}조 {n_sim:,}회 시뮬레이션 중..."):
            df_stats = simulate_group_many(grp, df_teams, team_ratings, n_sim=n_sim)
        st.session_state[f"grp_stats_{grp}"] = df_stats

    df_stats = st.session_state.get(f"grp_stats_{grp}")
    if df_stats is None or df_stats.empty:
        st.info("조를 선택하고 실행하세요.")
        return

    # 진출확률(1~2위) 막대
    df_stats = df_stats.copy()
    df_stats["진출%"] = df_stats["P1(1위%)"] + df_stats["P2(2위%)"]
    st.markdown("**조 통과(1~2위) 확률**")
    st.altair_chart(
        _hbar(df_stats[["team_name_ko", "진출%"]].rename(columns={"team_name_ko": "team"}),
              "진출%", "team", "진출%", color="#3ec97a", fmt=".1f", height=200),
        use_container_width=True,
    )

    ren = {
        "team_name_ko": "팀",
        "P1(1위%)": "1위%", "P2(2위%)": "2위%", "P3(3위%)": "3위%", "P4(4위%)": "4위%",
        "avg_PTS": "평균승점", "avg_GD": "평균득실", "avg_GF": "평균득점",
    }
    show = df_stats.drop(columns=["team_code", "진출%"], errors="ignore").rename(columns=ren)
    st.dataframe(
        show, use_container_width=True, hide_index=True,
        column_config={
            "1위%": st.column_config.ProgressColumn(
                "1위%", format="%.1f", min_value=0.0, max_value=100.0),
            "2위%": st.column_config.NumberColumn("2위%", format="%.1f"),
            "3위%": st.column_config.NumberColumn("3위%", format="%.1f"),
            "4위%": st.column_config.NumberColumn("4위%", format="%.1f"),
            "평균승점": st.column_config.NumberColumn("평균승점", format="%.2f"),
            "평균득실": st.column_config.NumberColumn("평균득실", format="%+.2f"),
            "평균득점": st.column_config.NumberColumn("평균득점", format="%.2f"),
        },
    )


# -----------------------------------------------------------------
def render_match_sim(df_teams, team_ratings):
    st.subheader("⚔️ 단일 경기 시뮬레이션")

    def label(code):
        r = df_teams[df_teams["team_code"] == code].iloc[0]
        return f"{r['team_name_ko']} ({code}) · {r['group_letter']}조"

    codes = df_teams["team_code"].tolist()
    c1, c2 = st.columns(2)
    home = c1.selectbox("홈 팀", codes, format_func=label, key="home_new", index=0)
    away_default = 1 if len(codes) > 1 else 0
    away = c2.selectbox("원정 팀", codes, format_func=label, key="away_new", index=away_default)

    n_sim = st.slider("시뮬 횟수", 100, 5000, 2000, step=100, key="match_n_new")
    run = st.button("⚔️ 맞대결 시뮬레이션", type="primary", key="match_run")

    if home == away:
        st.warning("서로 다른 두 팀을 선택하세요.")
        return

    if run:
        hr = df_teams[df_teams["team_code"] == home].iloc[0]
        ar = df_teams[df_teams["team_code"] == away].iloc[0]
        summary = simulate_many(hr, ar, team_ratings, n_sim=n_sim)
        eh, _ = get_team_elo(hr, team_ratings)
        ea, _ = get_team_elo(ar, team_ratings)

        hn = hr["team_name_ko"]; an = ar["team_name_ko"]
        hw = summary["home_wins"] / n_sim * 100
        dr = summary["draws"] / n_sim * 100
        aw = summary["away_wins"] / n_sim * 100

        c1, c2, c3 = st.columns(3)
        c1.metric(f"{hn} 승", f"{hw:.1f}%", help=f"Elo {eh:.0f}")
        c2.metric("무승부", f"{dr:.1f}%")
        c3.metric(f"{an} 승", f"{aw:.1f}%", help=f"Elo {ea:.0f}")

        st.caption(
            f"평균 스코어  {hn} {summary['avg_home_goals']:.2f} – "
            f"{summary['avg_away_goals']:.2f} {an}"
        )

        # 승무패 분포 막대
        wld = pd.DataFrame({
            "결과": [f"{hn} 승", "무승부", f"{an} 승"],
            "확률": [hw, dr, aw],
        })
        bar = (
            alt.Chart(wld).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X("확률:Q", title=None),
                y=alt.Y("결과:N", sort=None, title=None),
                color=alt.Color("결과:N", scale=alt.Scale(
                    range=["#5b8def", "#9aa0a6", "#e0563e"]), legend=None),
                tooltip=["결과", alt.Tooltip("확률:Q", format=".1f")],
            ).properties(height=140)
        )
        st.altair_chart(bar, use_container_width=True)

        # 자주 나오는 스코어 TOP 6
        sc = summary["score_counts"]
        sc_rows = sorted(sc.items(), key=lambda kv: kv[1], reverse=True)[:6]
        sc_df = pd.DataFrame([
            {"스코어": f"{h}-{a}", "확률%": cnt / n_sim * 100} for (h, a), cnt in sc_rows
        ])
        st.markdown("**자주 나오는 스코어**")
        st.dataframe(
            sc_df.style.format({"확률%": "{:.1f}"}),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("두 팀을 고르고 맞대결을 시뮬레이션하세요.")


# -----------------------------------------------------------------
def render_teams(df_teams, df_players, team_ratings):
    st.subheader("👥 팀 정보")

    fc1, fc2 = st.columns(2)
    groups = ["전체"] + sorted(df_teams["group_letter"].unique())
    confeds = ["전체"] + sorted(df_teams["confed"].unique())
    fg = fc1.selectbox("조 필터", groups, key="tf_grp")
    fconf = fc2.selectbox(
        "대륙 필터", confeds,
        format_func=lambda c: c if c == "전체" else f"{CONFED_LABEL.get(c, c)} ({c})",
        key="tf_conf",
    )

    view = df_teams.copy()
    if fg != "전체":
        view = view[view["group_letter"] == fg]
    if fconf != "전체":
        view = view[view["confed"] == fconf]

    # Elo 붙이기
    elos = []
    for _, r in view.iterrows():
        e, _s = get_team_elo(r, team_ratings)
        elos.append(round(e, 0))
    view = view.assign(Elo=elos).sort_values("Elo", ascending=False)

    show = view[["team_name_ko", "team_code", "group_letter", "confed", "Elo"]].rename(
        columns={"team_name_ko": "팀", "team_code": "코드",
                 "group_letter": "조", "confed": "대륙"}
    )
    show["대륙"] = show["대륙"].map(lambda c: CONFED_LABEL.get(c, c))
    st.dataframe(show, use_container_width=True, hide_index=True)

    # 선수 명단 조회
    st.divider()
    st.markdown("**선수 명단 조회**")
    code_options = view["team_code"].tolist()
    if not code_options:
        st.info("필터에 해당하는 팀이 없습니다.")
        return

    def plabel(code):
        r = df_teams[df_teams["team_code"] == code].iloc[0]
        return f"{r['team_name_ko']} ({code})"

    sel = st.selectbox("팀 선택", code_options, format_func=plabel, key="pl_team")
    if df_players.empty:
        st.warning("선수 데이터 파일이 없습니다.")
        return
    pl = df_players[df_players["team_code"] == sel].copy()
    if pl.empty:
        st.info(f"{plabel(sel)} 는 선수 데이터가 없어 실측 Elo로 평가됩니다.")
        return

    pos_order = {"GK": 0, "DF": 1, "MF": 2, "FW": 3}
    pl["_po"] = pl["position"].map(pos_order).fillna(9)
    pl = pl.sort_values(["_po", "is_starting", "player_name_en"],
                        ascending=[True, False, True])
    pl["주전"] = pl["is_starting"].map(lambda x: "●" if x == 1 else "")
    show_pl = pl[["player_name_ko", "position", "주전",
                  "attack", "defense", "passing", "physical", "mental", "gk"]].rename(
        columns={"player_name_ko": "이름", "position": "포지션",
                 "attack": "공격", "defense": "수비", "passing": "패스",
                 "physical": "피지컬", "mental": "멘탈", "gk": "GK"}
    )
    st.dataframe(show_pl, use_container_width=True, hide_index=True, height=480)


# -----------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="World Cup 2026 Simulator",
        page_icon="🏆",
        layout="wide",
    )

    # 데이터 로드
    df_teams = load_teams()
    df_players = load_players()
    team_ratings = build_team_ratings(df_players) if not df_players.empty else {}
    global _TEAM_ELO_CACHE
    _TEAM_ELO_CACHE = load_team_elo()

    # 헤더
    st.title("🏆 2026 월드컵 시뮬레이터")
    st.caption(
        "선수 능력치 + 실측 Elo 기반 몬테카를로 시뮬레이션 · "
        "캐나다 / 멕시코 / 미국 · 2026.6.11 – 7.19"
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏆 우승 예측",
        "📊 조별리그",
        "⚔️ 경기 시뮬",
        "🌍 개요",
        "👥 팀 정보",
    ])

    with tab1:
        render_title_prediction(df_teams, team_ratings)
    with tab2:
        render_group_stage(df_teams, team_ratings)
    with tab3:
        render_match_sim(df_teams, team_ratings)
    with tab4:
        render_overview(df_teams, team_ratings)
    with tab5:
        render_teams(df_teams, df_players, team_ratings)


if __name__ == "__main__":
    main()
