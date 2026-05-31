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
# Streamlit UI
# =========================
def main():
    st.set_page_config(page_title="World Cup 2026 – KOR/JPN Prototype", layout="wide")
    st.title("🏆 World Cup 2026 – Team & Match Simulator (KOR/JPN Prototype)")

    df_teams = load_teams()
    df_players = load_players()
    team_ratings = build_team_ratings(df_players)

    # 실측 Elo 테이블 로딩 + 전역 캐시에 주입
    global _TEAM_ELO_CACHE
    team_elo = load_team_elo()
    _TEAM_ELO_CACHE = team_elo

    # -------------------------
    # 1) 팀 마스터 + 간단 통계
    # -------------------------
    st.sidebar.header("필터")

    group_options = ["ALL"] + sorted(df_teams["group_letter"].unique().tolist())
    selected_group = st.sidebar.selectbox("그룹 선택", group_options)

    confed_all = sorted(df_teams["confed"].unique().tolist())
    selected_confed = st.sidebar.multiselect(
        "컨페더레이션 선택",
        options=confed_all,
        default=confed_all,
    )

    df_view = df_teams[df_teams["confed"].isin(selected_confed)]
    if selected_group != "ALL":
        df_view = df_view[df_view["group_letter"] == selected_group]

    st.subheader("팀 리스트")
    cols = [
        "group_letter",
        "slot",          # A1, B3 ...
        "team_code",
        "team_name_ko",
        "confed",
        "seeding_pot",
        "is_host",
        "notes",
    ]
    cols_existing = [c for c in cols if c in df_view.columns]

    st.dataframe(
        df_view[cols_existing],
        use_container_width=True,
        hide_index=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("그룹별 팀 수")
        group_counts = df_teams.groupby("group_letter")["team_code"].count()
        st.bar_chart(group_counts)

    with col2:
        st.subheader("컨페더레이션별 팀 수")
        confed_counts = df_teams.groupby("confed")["team_code"].count()
        st.bar_chart(confed_counts)

    st.markdown("---")

    # -------------------------
    # 2) 단일 경기 + 선수 미리보기
    # -------------------------
    st.header("⚽ 단일 경기 시뮬레이션")

    team_codes = df_teams["team_code"].tolist()

    def format_team_label(code: str) -> str:
        row = df_teams[df_teams["team_code"] == code].iloc[0]
        return (
            f"{row['team_name_ko']} ({code}) "
            f"- 그룹 {row['group_letter']} / 슬롯 {row['slot']} / 포트 {row['seeding_pot']}"
        )

    # 기본값: 아직 시뮬 불가 상태
    home_row = None
    away_row = None

    colA, colB = st.columns(2)

    with colA:
        home_code = st.selectbox(
            "홈 팀 선택",
            options=team_codes,
            format_func=format_team_label,
        )

    with colB:
        default_idx = 1 if len(team_codes) > 1 else 0
        away_code = st.selectbox(
            "원정 팀 선택",
            options=team_codes,
            index=default_idx,
            format_func=format_team_label,
        )

    if home_code == away_code:
        st.warning("홈 팀과 원정 팀을 다르게 선택해 주세요.")
    else:
        home_row = df_teams[df_teams["team_code"] == home_code].iloc[0]
        away_row = df_teams[df_teams["team_code"] == away_code].iloc[0]

        st.subheader("선수 데이터 미리 보기")

        colP1, colP2 = st.columns(2)

        with colP1:
            home_players = df_players[df_players["team_code"] == home_code]
            if home_players.empty:
                st.caption(f"🔍 {home_row['team_name_ko']} 선수 데이터가 아직 없습니다.")
            else:
                st.markdown(f"**{home_row['team_name_ko']} ({home_code}) 선수 목록**")
                st.dataframe(
                    home_players[
                        [
                            "player_name_ko",
                            "position",
                            "is_starting",
                            "attack",
                            "defense",
                            "passing",
                            "physical",
                            "mental",
                            "gk",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

        with colP2:
            away_players = df_players[df_players["team_code"] == away_code]
            if away_players.empty:
                st.caption(f"🔍 {away_row['team_name_ko']} 선수 데이터가 아직 없습니다.")
            else:
                st.markdown(f"**{away_row['team_name_ko']} ({away_code}) 선수 목록**")
                st.dataframe(
                    away_players[
                        [
                            "player_name_ko",
                            "position",
                            "is_starting",
                            "attack",
                            "defense",
                            "passing",
                            "physical",
                            "mental",
                            "gk",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

        st.markdown("---")

        # 단일 경기 버튼
        if st.button("🧮 한 경기 시뮬레이션 돌리기"):
            goals_home, goals_away, meta = simulate_match(home_row, away_row, team_ratings)

            st.subheader("단일 경기 결과")
            st.markdown(
                f"### **{home_row['team_name_ko']} {goals_home} - {goals_away} {away_row['team_name_ko']}**"
            )

            src_map = {
                "players_csv": "선수 능력치 기반",
                "seeding_pot": "포트 기반 (임시)",
            }

            st.caption(
                f"홈 Elo: {meta['elo_home']:.1f} "
                f"({src_map.get(meta['src_home'], meta['src_home'])})  |  "
                f"원정 Elo: {meta['elo_away']:.1f} "
                f"({src_map.get(meta['src_away'], meta['src_away'])})"
            )
            st.caption(
                f"기대 득점 λ  홈: {meta['lam_home']:.2f}  /  원정: {meta['lam_away']:.2f}"
            )

        st.markdown("---")

    # -------------------------
    # 3) 다중 시뮬레이션 (승/무/패 확률)
    # -------------------------
    st.header("📊 다중 시뮬레이션 – 승/무/패 확률")

    n_sim = st.slider(
        "시뮬레이션 횟수",
        min_value=100,
        max_value=5000,
        step=100,
        value=1000,
    )

    if st.button("🔁 다중 시뮬레이션 돌리기"):
        if home_row is None or away_row is None:
            st.warning("홈/원정 팀을 서로 다르게 선택한 뒤에만 다중 시뮬레이션을 실행할 수 있습니다.")
        else:
            summary = simulate_many(home_row, away_row, team_ratings, n_sim=n_sim)

            home_name = home_row["team_name_ko"]
            away_name = away_row["team_name_ko"]

            home_wins = summary["home_wins"]
            draws = summary["draws"]
            away_wins = summary["away_wins"]

            p_home = home_wins / n_sim * 100
            p_draw = draws / n_sim * 100
            p_away = away_wins / n_sim * 100

            avg_home_goals = summary["avg_home_goals"]
            avg_away_goals = summary["avg_away_goals"]

            meta_example = summary["meta_example"]

            st.subheader("요약")

            c1, c2, c3 = st.columns(3)
            c1.metric(f"{home_name} 승", f"{p_home:.1f}%", f"{home_wins} / {n_sim}")
            c2.metric("무승부", f"{p_draw:.1f}%", f"{draws} / {n_sim}")
            c3.metric(f"{away_name} 승", f"{p_away:.1f}%", f"{away_wins} / {n_sim}")

            st.caption(
                f"평균 스코어: {home_name} {avg_home_goals:.2f} - {avg_away_goals:.2f} {away_name}"
            )

            src_map = {
                "players_csv": "선수 능력치 기반",
                "seeding_pot": "포트 기반 (임시)",
            }

            st.caption(
                f"Elo(예시)  홈: {meta_example['elo_home']:.1f} "
                f"({src_map.get(meta_example['src_home'], meta_example['src_home'])})  |  "
                f"원정: {meta_example['elo_away']:.1f} "
                f"({src_map.get(meta_example['src_away'], meta_example['src_away'])})"
            )
            st.caption(
                f"기대 득점 λ(예시)  홈: {meta_example['lam_home']:.2f}  /  "
                f"원정: {meta_example['lam_away']:.2f}"
            )

            score_counts = summary["score_counts"]
            rows = [
                {
                    "home_goals": gh,
                    "away_goals": ga,
                    "count": cnt,
                    "prob_%": cnt / n_sim * 100,
                }
                for (gh, ga), cnt in score_counts.items()
            ]
            rows_sorted = sorted(rows, key=lambda x: x["count"], reverse=True)[:5]

            if rows_sorted:
                df_scores = pd.DataFrame(rows_sorted)
                df_scores = df_scores.rename(
                    columns={
                        "home_goals": f"{home_name} 골",
                        "away_goals": f"{away_name} 골",
                        "count": "횟수",
                        "prob_%": "확률(%)",
                    }
                )
                st.table(df_scores)
            else:
                st.caption("스코어 데이터가 없습니다.")

            st.info(
                f"{n_sim}번의 시뮬레이션 결과입니다. "
                "KOR / JPN은 players_2026.csv의 선수 능력치를 기반으로 팀 레이팅을 계산하고, "
                "다른 팀은 포트(seeding_pot) 기반 레이팅을 사용합니다."
            )

    st.markdown("---")

    # -------------------------
    # 4) 조별리그 – 그룹 단일 시뮬레이션
    # -------------------------
    st.header("🧮 조별리그 단일 시뮬레이션 (그룹별)")

    group_for_sim = st.selectbox(
        "조별리그에서 시뮬레이션할 그룹을 선택하세요",
        sorted(df_teams["group_letter"].unique().tolist()),
        index=0,
    )

    if st.button("🎯 선택한 그룹 한 번 시뮬레이션"):
        df_table, df_matches = simulate_group_once(
            group_for_sim, df_teams, team_ratings
        )

        if df_table.empty:
            st.warning("해당 그룹에 팀 데이터가 없습니다.")
        else:
            st.subheader(f"그룹 {group_for_sim} 최종 순위표")
            st.dataframe(
                df_table[
                    [
                        "Rank",
                        "team_name_ko",
                        "team_code",
                        "P",
                        "W",
                        "D",
                        "L",
                        "GF",
                        "GA",
                        "GD",
                        "PTS",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

            st.subheader(f"그룹 {group_for_sim} 경기 결과")
            st.table(
                df_matches[
                    [
                        "matchday",
                        "home_team",
                        "away_team",
                        "score",
                        "home_goals",
                        "away_goals",
                    ]
                ]
            )

            st.caption(
                "일정은 그룹 내 팀의 슬롯(A1~L4, group_pos)을 기준으로 "
                "총 3라운드(각 팀 3경기) 라운드 로빈 형태로 자동 생성됩니다."
            )

    st.markdown("---")

    # -------------------------
    # 5) 조별리그 – 다중 시뮬레이션 (그룹별 순위 확률)
    # -------------------------
    st.header("📈 조별리그 다중 시뮬레이션 (순위 확률)")

    group_for_mc = st.selectbox(
        "다중 시뮬레이션할 그룹을 선택하세요",
        sorted(df_teams["group_letter"].unique().tolist()),
        index=0,
        key="group_for_mc",
    )

    n_group_sim = st.slider(
        "그룹 시뮬레이션 횟수",
        min_value=100,
        max_value=5000,
        step=100,
        value=1000,
        key="n_group_sim",
    )

    if st.button("📈 선택한 그룹 다중 시뮬레이션 돌리기"):
        df_stats = simulate_group_many(
            group_for_mc,
            df_teams,
            team_ratings,
            n_sim=n_group_sim,
        )

        if df_stats.empty:
            st.warning("해당 그룹에 팀 데이터가 없습니다.")
        else:
            st.subheader(f"그룹 {group_for_mc} 순위 확률 요약")
            st.dataframe(
                df_stats,
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                f"{n_group_sim}번 조별리그를 돌린 결과입니다. "
                "각 팀의 1위·2위·3위·4위 확률과 평균 승점/득실차/득점을 보여줍니다."
            )

    st.markdown("---")

    # -------------------------
    # 6) 전체 조별리그 – 다중 시뮬레이션 (전 팀 요약)
    # -------------------------
    st.header("🌍 전체 조별리그 다중 시뮬레이션 (전 팀 요약)")

    n_all_sim = st.slider(
        "전체 조별리그 시뮬레이션 횟수",
        min_value=100,
        max_value=5000,
        step=100,
        value=1000,
        key="n_all_sim",
    )

    if st.button("🌍 전체 그룹 한 번에 시뮬레이션"):
        df_all_stats = simulate_all_groups_many(
            df_teams,
            team_ratings,
            n_sim=n_all_sim,
        )

        if df_all_stats.empty:
            st.warning("조별리그 통계를 계산할 수 있는 팀 데이터가 없습니다.")
        else:
            show_cols = [
                "group_letter",
                "team_name_ko",
                "team_code",
                "P1(1위%)",
                "P2(2위%)",
                "P3(3위%)",
                "P4(4위%)",
                "P_qual(1~2위%)",
                "avg_PTS",
                "avg_GD",
                "avg_GF",
            ]
            show_cols = [c for c in show_cols if c in df_all_stats.columns]

            st.subheader("전체 팀 순위 확률 및 진출 확률 요약")
            st.dataframe(
                df_all_stats[show_cols],
                use_container_width=True,
                hide_index=True,
            )

            # KOR / JPN만 별도 표시
            mask_kor_jpn = df_all_stats["team_code"].isin(["KOR", "JPN"])
            df_kor_jpn = df_all_stats[mask_kor_jpn]

            if not df_kor_jpn.empty:
                st.subheader("🇰🇷 KOR / 🇯🇵 JPN 요약")
                st.dataframe(
                    df_kor_jpn[show_cols],
                    use_container_width=True,
                    hide_index=True,
                )

            st.caption(
                f"{n_all_sim}번 전체 조별리그를 돌린 결과입니다. "
                "각 팀의 1~4위 확률과 평균 승점/득실차/득점을 한 번에 볼 수 있으며, "
                "P_qual(1~2위%)는 해당 팀이 조 1~2위로 올라갈 확률을 의미합니다."
            )

    # =====================================================================
    # 🏆 우승 예측 (전체 토너먼트 몬테카를로)
    # =====================================================================
    st.header("🏆 우승 예측 – 전체 토너먼트 시뮬레이션")
    st.caption(
        "조별리그(72경기) → 32강 → 16강 → 8강 → 4강 → 결승까지 "
        "대회 전체를 수천 번 돌려 각 팀의 단계별 진출 확률과 우승 확률을 계산합니다. "
        "32강 대진은 FIFA 공식 규정(매치 73~88) + 베스트 3위 8팀 배정표(Annex C, 495개 시나리오)를 그대로 적용합니다."
    )

    n_tour = st.slider(
        "토너먼트 시뮬레이션 횟수",
        min_value=1000,
        max_value=50000,
        step=1000,
        value=10000,
        key="n_tour",
    )

    if st.button("🏆 우승 확률 계산"):
        try:
            import tournament as TNM
        except Exception as e:
            st.error(f"tournament 모듈을 불러오지 못했습니다: {e}")
        else:
            with st.spinner(f"{n_tour:,}회 대회 시뮬레이션 중..."):
                df_tour = TNM.simulate_tournament_fast(
                    df_teams,
                    team_ratings,
                    get_team_elo,
                    n_sim=n_tour,
                )

            st.subheader("🥇 우승 확률 순위")
            show = df_tour.copy()
            ren = {
                "rank": "순위",
                "team": "팀",
                "group": "조",
                "champion_pct": "우승%",
                "final_pct": "결승%",
                "semi_pct": "4강%",
                "quarter_pct": "8강%",
                "r16_pct": "16강%",
                "r32_pct": "32강%",
            }
            cols = ["rank", "team", "group", "champion_pct", "final_pct",
                    "semi_pct", "quarter_pct", "r16_pct", "r32_pct"]
            show = show[cols].rename(columns=ren)
            for c in ["우승%", "결승%", "4강%", "8강%", "16강%", "32강%"]:
                show[c] = show[c].map(lambda v: f"{v:.1f}")

            st.dataframe(show, use_container_width=True, hide_index=True)

            # 상위 12팀 우승확률 바차트
            top = df_tour.head(12).set_index("team")["champion_pct"]
            st.subheader("우승 확률 TOP 12")
            st.bar_chart(top)

            # KOR/JPN 별도 하이라이트
            kj = df_tour[df_tour["team_code"].isin(["KOR", "JPN"])]
            if not kj.empty:
                st.subheader("🇰🇷 KOR / 🇯🇵 JPN")
                kjs = kj[cols].rename(columns=ren)
                for c in ["우승%", "결승%", "4강%", "8강%", "16강%", "32강%"]:
                    kjs[c] = kjs[c].map(lambda v: f"{v:.1f}")
                st.dataframe(kjs, use_container_width=True, hide_index=True)

            st.caption(
                f"{n_tour:,}회 몬테카를로 결과입니다. 우승%의 합은 100%, 결승%의 합은 200%가 됩니다. "
                "녹아웃 무승부는 Elo 가중 확률(연장+승부차기 대용)로 처리합니다. "
                "선수 데이터가 없는 팀은 실측 Elo로 대체 평가됩니다."
            )


if __name__ == "__main__":
    main()
