# app.py
import streamlit as st
import pandas as pd

@st.cache_data
def load_teams():
    df = pd.read_csv("data/teams_2026.csv")

    # 기본 정렬: group → seeding_pot
    df = df.sort_values(["group_letter", "seeding_pot"])
    return df


def main():
    st.set_page_config(page_title="World Cup 2026 – Team Master", layout="wide")
    st.title("🏆 World Cup 2026 – Team Master")

    df = load_teams()

    # --- 사이드바 필터 ---
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

    # --- 메인 화면 ---
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


if __name__ == "__main__":
    main()

