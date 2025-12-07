# app.py
import streamlit as st
import pandas as pd
import numpy as np


@st.cache_data
def load_teams():
    df = pd.read_csv("data/teams_2026.csv")
    # 그룹, 포트 순으로 정렬
    df = df.sort_values(["group_letter", "seeding_pot", "team_code"])
    return df


def pot_to_rating(pot: int) -> float:
    """
    포트 번호(1~4)를 간단한 팀 레이팅으로 변환
    나중에 FIFA 랭킹 / 선수 능력치로 대체할 예정
    """
    pot = int(pot)
    base = {
        1: 1900.0,
        2: 1800.0,
        3: 1700.0,
        4: 1600.0,
    }
    return base.get(pot, 1750.0)


def expected_goals_from_ratings(rA: float, rB: float):
    """
    두 팀 레이팅으로부터 각 팀 기대 득점 λA, λB 계산
    - 레이팅 차이 → 승률 비슷한 확률
    - 총 기대 득점은 대략 2.6 골 근처로 고정
    """
    diff = rA - rB
    # Elo 스타일 승률
    pA = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
    total_goals = 2.6
    lamA = total_goals * pA
    lamB = total_goals * (1.0 - pA)
    return float(lamA), float(lamB)


def simulate_match(rowA: pd.Series, rowB: pd.Series, seed: int | None = None):
    """
    포트 기반 간단 매치 시뮬레이션
    - 팀 A, B 행(row)을 받아서
    - 포트 → 레이팅 → 기대 득점 λ → 포아송 샘플링
    """
    if seed is not None:
        np.random.seed(seed)

    potA = rowA["seeding_pot"]
    potB = rowB["seeding_pot"]

    ratingA = pot_to_rating(potA)
    ratingB = pot_to_rating(potB)

    lamA, lamB = expected_goals_from_ratings(ratingA, ratingB)

    goalsA = np.random.poisson(lamA)
    goalsB = np.random.poisson(lamB)

    return int(goalsA), int(goalsB), lamA, lamB, ratingA, ratingB


def main():
    st.set_page_config(page_title="World Cup 2026 – Team Master", layout="wide")
    st.title("🏆 World Cup 2026 – Team Master")

    df = load_teams()

    # =========================
    # 1) 팀 마스터 보기 + 필터
    # =========================
    st.sidebar.header("필터")

    # 그룹 필터
    group_options = ["ALL"] + sorted(df["group_letter"].unique().tolist())
    selected_group = st.sidebar.selectbox("그룹 선택", group_options)

    # 컨페더레이션 필터
    confed_all = sorted(df["confed"].unique().tolist())
    selected_confed = st.sidebar.multiselect(
        "컨페더레이션 선택",
        options=confed_all,
        default=confed_all,
    )

    df_view = df[df["confed"].isin(selected_confed)]
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
        group_counts = df.groupby("group_letter")["team_code"].count()
        st.bar_chart(group_counts)

    with col2:
        st.subheader("컨페더레이션별 팀 수")
        confed_counts = df.groupby("confed")["team_code"].count()
        st.bar_chart(confed_counts)

    st.markdown("---")

    # =========================
    # 2) 단일 경기 시뮬레이션
    # =========================
    st.header("⚽ 단일 경기 시뮬레이션 (포트 기반 1차 버전)")

    team_codes = df["team_code"].tolist()

    def label_func(code: str) -> str:
        row = df[df["team_code"] == code].iloc[0]
        return f"{row['team_name_ko']} ({code}) - 그룹 {row['group_letter']} 포트 {row['seeding_pot']}"

    colA, colB = st.columns(2)

    with colA:
        home_code = st.selectbox(
            "홈 팀 선택",
            options=team_codes,
            format_func=label_func,
        )

    with colB:
        # 기본값은 두 번째 팀으로 (홈과 다르게 보이도록)
        default_index = 1 if len(team_codes) > 1 else 0
        away_code = st.selectbox(
            "원정 팀 선택",
            options=team_codes,
            index=default_index,
            format_func=label_func,
        )

    if home_code == away_code:
        st.warning("홈 팀과 원정 팀을 다르게 선택해 주세요.")
        return

    home_row = df[df["team_code"] == home_code].iloc[0]
    away_row = df[df["team_code"] == away_code].iloc[0]

    if st.button("🧮 한 경기 시뮬레이션 돌리기"):
        goalsA, goalsB, lamA, lamB, ratingA, ratingB = simulate_match(home_row, away_row)

        st.subheader("결과")

        st.markdown(
            f"### **{home_row['team_name_ko']} {goalsA} - {goalsB} {away_row['team_name_ko']}**"
        )

        st.caption(
            f"레이팅(임시): 홈 {ratingA:.0f} vs 원정 {ratingB:.0f}  |  "
            f"기대 득점 λ: 홈 {lamA:.2f}, 원정 {lamB:.2f}"
        )

        st.info("지금은 포트(시드)만 반영한 1차 버전입니다. 나중에 선수 능력치 / 전술이 여기로 들어갈 예정.")


if __name__ == "__main__":
    main()
