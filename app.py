# app.py
import streamlit as st
import pandas as pd
import numpy as np


# =========================
# 데이터 로딩
# =========================
@st.cache_data
def load_teams():
    df = pd.read_csv("data/teams_2026.csv")
    df = df.sort_values(["group_letter", "seeding_pot", "team_code"])
    return df


@st.cache_data
def load_players():
    try:
        df = pd.read_csv("data/players_2026.csv")
    except FileNotFoundError:
        return pd.DataFrame()
    # 기본 정렬: 팀 → 포지션 → 선발 여부
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
    선수 개별 종합 능력치(0~100)를 하나로 압축
    - 공격 비중 조금 높게
    - 수비/패스는 중간
    - 피지컬/멘탈은 보조
    """
    return (
        row["attack"] * 0.35
        + row["defense"] * 0.30
        + row["passing"] * 0.15
        + row["physical"] * 0.10
        + row["mental"] * 0.10
    )


def build_team_ratings(df_players: pd.DataFrame, use_starting_only: bool = True):
    """
    players_2026.csv를 기반으로 팀별 레이팅 계산
    - 우선 선발 11명 평균으로 팀 능력치 산출
    - 해당 팀에 선수 데이터가 없으면 이 dict에 안 들어감
    반환 형식:
    {
      "KOR": {
          "overall": 83.2,
          "attack": 82.1,
          "defense": 78.3,
          ...
      },
      ...
    }
    """
    ratings: dict[str, dict[str, float]] = {}

    if df_players.empty:
        return ratings

    # player_overall 컬럼 추가
    df = df_players.copy()
    df["player_overall"] = df.apply(compute_player_overall, axis=1)

    for team_code, grp in df.groupby("team_code"):
        g = grp
        if use_starting_only:
            starters = g[g["is_starting"] == 1]
            if len(starters) >= 8:  # 선발이 어느 정도 있으면 선발 기준
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


def pot_to_rating(pot: int) -> float:
    """
    포트 번호(1~4)를 간단 Elo-비슷한 레이팅으로 변환
    - 선수 데이터 없는 팀용 fallback
    """
    pot = int(pot)
    base = {
        1: 1850.0,
        2: 1800.0,
        3: 1750.0,
        4: 1700.0,
    }
    return base.get(pot, 1775.0)


def overall_to_elo(overall: float) -> float:
    """
    선수 평균 overall(0~100)을 Elo 비슷한 스케일로 변환
    - 75를 1800 정도, 90을 1950 근처로 맞추는 느낌
    """
    return 1800.0 + (overall - 75.0) * 10.0


def get_team_elo(row_team: pd.Series, team_ratings: dict) -> tuple[float, str]:
    """
    해당 팀의 최종 Elo 레이팅과, 어떤 소스를 썼는지 설명 문자열 반환
    - players_2026에 데이터 있으면: 선수 기반
    - 없으면: seeding_pot 기반
    """
    code = row_team["team_code"]
    pot = row_team["seeding_pot"]

    if code in team_ratings:
        overall = team_ratings[code]["overall"]
        elo = overall_to_elo(overall)
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
    두 팀 Elo 레이팅으로부터 각 팀 기대 득점 λA, λB 계산
    - Elo 차이 → 승률 → 기대 득점 분배
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
    - 팀 Elo(선수 기반 or 포트 기반) → 기대 득점 → 포아송 랜덤 골수
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


# =========================
# Streamlit UI
# =========================
def main():
    st.set_page_config(page_title="World Cup 2026 – KOR/JPN Prototype", layout="wide")
    st.title("🏆 World Cup 2026 – Team & Match Simulator (KOR/JPN Prototype)")

    df_teams = load_teams()
    df_players = load_players()
    team_ratings = build_team_ratings(df_players)

    # -------------------------
    # 1) 팀 마스터 + 간단 통계
    # -------------------------
    st.sidebar.header("필터")

    # 그룹 필터
    group_options = ["ALL"] + sorted(df_teams["group_letter"].unique().tolist())
    selected_group = st.sidebar.selectbox("그룹 선택", group_options)

    # 컨페더레이션 필터
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
    st.dataframe(
        df_view,
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
    # 2) 단일 경기 시뮬레이션
    # -------------------------
    st.header("⚽ 단일 경기 시뮬레이션")

    team_codes = df_teams["team_code"].tolist()

    def format_team_label(code: str) -> str:
        row = df_teams[df_teams["team_code"] == code].iloc[0]
        return f"{row['team_name_ko']} ({code}) - 그룹 {row['group_letter']} 포트 {row['seeding_pot']}"

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
        return

    home_row = df_teams[df_teams["team_code"] == home_code].iloc[0]
    away_row = df_teams[df_teams["team_code"] == away_code].iloc[0]

    # 선수 테이블 미리 보기 (KOR/JPN만 데이터 존재)
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

    if st.button("🧮 한 경기 시뮬레이션 돌리기"):
        goals_home, goals_away, meta = simulate_match(home_row, away_row, team_ratings)

        st.subheader("결과")
        st.markdown(
            f"### **{home_row['team_name_ko']} {goals_home} - {goals_away} {away_row['team_name_ko']}**"
        )

        # 레이팅/λ 설명
        src_map = {
            "players_csv": "선수 능력치 기반",
            "seeding_pot": "포트 기반 (임시)",
        }

        st.caption(
            f"홈 Elo: {meta['elo_home']:.1f} ({src_map.get(meta['src_home'], meta['src_home'])})  |  "
            f"원정 Elo: {meta['elo_away']:.1f} ({src_map.get(meta['src_away'], meta['src_away'])})"
        )
        st.caption(
            f"기대 득점 λ  홈: {meta['lam_home']:.2f}  /  원정: {meta['lam_away']:.2f}"
        )

        st.info(
            "KOR / JPN은 players_2026.csv에 있는 선수 능력치 평균으로 팀 레이팅을 계산합니다. "
            "다른 팀은 아직 선수 데이터가 없어서 포트(seeding_pot) 기반 레이팅을 사용 중입니다."
        )


if __name__ == "__main__":
    main()
