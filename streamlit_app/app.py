import json
import os

import streamlit as st

st.set_page_config(
    page_title="FanPulse",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

METRICS_PATH = "backend/data/processed/validation_metrics.json"


@st.cache_data
def load_validation_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return None


st.title("⚾ FanPulse")
st.caption("Bayesian Marketing Mix Model — MLB Fan Demand Analysis")

st.warning(
    "**Note:** Marketing spend, channel ROI, and promo-night flags are **synthetic** "
    "proxies built to demonstrate MMM methodology. Attendance and game results are real. "
    "See the **Methodology** page for the full data-provenance breakdown."
)

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
metrics = load_validation_metrics()
if metrics is not None:
    test_season = metrics.get("test_season", 2024)
    st.caption(
        f"Time-based holdout: trained on {metrics['train_seasons']}, "
        f"evaluated on held-out {test_season} ({metrics['n_test']:,} games). "
        "Reproduce with `python -m backend.models.validate`."
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"Holdout R² ({test_season})", f"{metrics['holdout']['r2']:.2f}")
    col2.metric(f"Holdout MAPE ({test_season})", f"{metrics['holdout']['mape']:.1f}%")
    col3.metric("Divergences", f"{metrics['convergence']['divergences']}")
    col4.metric("Max R-hat", f"{metrics['convergence']['max_rhat']:.3f}")
else:
    st.caption(
        "Run `python -m backend.models.validate` to generate reproducible "
        "out-of-sample metrics here."
    )

st.markdown("### Key Findings")
st.markdown("""
**Estimated against real attendance:**
- Opening Day drives a large **attendance premium** over an average home game
- Teams in playoff races see a **September lift** in attendance
- Improvement in rolling win rate adds **incremental fans per game**
- Large market teams average **materially higher attendance** than small market teams

**Illustrative (synthetic spend / ROI — methodology demo, not empirical):**
- Email shows the highest modeled marketing ROI, paid social second
- Rebalancing budget away from broadcast toward email and paid social projects a positive attendance lift
""")
st.caption(
    "Marketing-channel findings use synthetic spend data and illustrate the MMM "
    "approach rather than real MLB media performance. See the Model Results and "
    "Methodology pages for the underlying posteriors."
)