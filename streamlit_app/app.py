import streamlit as st

st.set_page_config(
    page_title="FanPulse",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚾ FanPulse")
st.caption("Bayesian Marketing Mix Model — MLB Fan Demand Analysis")

st.markdown("""
This dashboard explores what drives MLB game attendance across 5 seasons (2019, 2021-2024),
covering 11,972 home games across all 30 teams.

A Bayesian MMM was fit to decompose attendance into:
- **Game factors** — opening day, rival matchups, promotional nights, playoff race
- **Team performance** — rolling win percentage, run differential, win streaks
- **Marketing channels** — paid social, email, broadcast, out-of-home spend
- **Market context** — city size, day of week, holiday weeks

Use the sidebar to navigate between pages.
""")

st.markdown("### Quick Stats")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Games Analyzed", "11,972")
col2.metric("Seasons", "5 (2019, 2021-24)")
col3.metric("Teams", "30")
col4.metric("Avg Attendance", "26,572")

st.markdown("### Model Performance")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Holdout R²", "0.81")
col2.metric("MAPE", "8.3%")
col3.metric("Divergences", "0")
col4.metric("Max R-hat", "1.005")

st.markdown("### Key Findings")
st.markdown("""
- Opening Day drives a **+42.7% attendance premium** over an average home game
- Teams in playoff races see **+11.3% lift** in September attendance
- A 10pp improvement in rolling win rate adds roughly **+2,600 fans per game**
- Large market teams average **+36.6% more attendance** than small market teams
- Paid social has the highest marketing ROI at **$4.20 per $1 spent**
- Rebalancing 18% of broadcast budget to email and paid social projects **+11.4% attendance lift**
""")