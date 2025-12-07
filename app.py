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


def pot_to_rating(pot: int) -> float:
    """
    포트 번호(1~4)를 Elo 비슷한 레이팅으로 변환
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
    - 75 → 1800 근처, 90 → 1950 근처
    """
    return 1800.0 + (overall - 75.0) * 10.0


def get_team_elo(row_team: pd.Series, team_ratings: dict) -> tuple[float, str]:
    """
    팀 최종 Elo 레이팅 + 소스
    - players_2026에 있으면 선수 기반
    - 없으면 포트 기반
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

    # 한 번 메타만 뽑아두고(설명용) 실제 확률 계산엔 안 씀
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
    # 2) 단일 경기 + 선수 미리보기
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
            f"홈 Elo: {meta['elo_home']:.1f} ({src_map.get(meta['src_home'], meta['src_home'])})  |  "
            f"원정 Elo: {meta['elo_away']:.1f} ({src_map.get(meta['src_away'], meta['src_away'])})"
        )
        st.caption(
            f"기대 득점 λ  홈: {meta['lam_home']:.2f}  /  원정: {meta['lam_away']:.2f}"
        )

    st.markdown("---")

    # -------------------------
    # 3) 다중 시뮬레이션 (승/무/패 확률)
    # -------------------------
    st.header("📊 다중 시뮬레이션 – 승/무/패 확률")

    n_sim = st.slider("시뮬레이션 횟수", min_value=100, max_value=5000, step=100, value=1000)

    if st.button("🔁 다중 시뮬레이션 돌리기"):
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
            f"Elo(예시)  홈: {meta_example['elo_home']:.1f} ({src_map.get(meta_example['src_home'], meta_example['src_home'])})  |  "
            f"원정: {meta_example['elo_away']:.1f} ({src_map.get(meta_example['src_away'], meta_example['src_away'])})"
        )
        st.caption(
            f"기대 득점 λ(예시)  홈: {meta_example['lam_home']:.2f}  /  원정: {meta_example['lam_away']:.2f}"
        )

        # 스코어 분포 상위 N개
        st.subheader("자주 나오는 스코어 TOP 5")

        score_counts = summary["score_counts"]
        # (gh, ga, count) 리스트로 변환 후 정렬
        rows = [
            {"home_goals": gh, "away_goals": ga, "count": cnt, "prob_%": cnt / n_sim * 100}
            for (gh, ga), cnt in score_counts.items()
        ]
        rows_sorted = sorted(rows, key=lambda x: x["count"], reverse=True)[:5]

        if rows_sorted:
            df_scores = pd.DataFrame(rows_sorted)
            # 보기 좋게 컬럼 이름 변경
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


if __name__ == "__main__":
    main()
