import streamlit as st
import pandas as pd

st.set_page_config(page_title="Methodology — FanPulse", layout="wide")
st.title("Methodology & Data Provenance")
st.caption("What is real, what is simulated, and how the model is built.")

st.warning(
    "**Important:** Marketing spend, channel ROI, and promotional-night flags in "
    "this project are **synthetic** — generated as plausible proxies calibrated to "
    "industry benchmarks, not measured from a real media-planning system. "
    "Attendance and on-field results are real. See the table below for the full breakdown."
)

st.markdown("### What's real vs. simulated")
provenance = pd.DataFrame([
    {"Field": "Game attendance", "Source": "Baseball Reference (via pybaseball)", "Status": "Real"},
    {"Field": "Game results / runs", "Source": "Baseball Reference (via pybaseball)", "Status": "Real"},
    {"Field": "Schedule, opponent, date", "Source": "Baseball Reference (via pybaseball)", "Status": "Real"},
    {"Field": "Rolling win pct / run diff / streak", "Source": "Derived from real results", "Status": "Real (derived)"},
    {"Field": "Opening day / rival / weekend / holiday", "Source": "Derived from real schedule", "Status": "Real (derived)"},
    {"Field": "Market size tier", "Source": "Manual classification of teams", "Status": "Assumption"},
    {"Field": "Promo nights / fireworks", "Source": "Random draw (~15/team/season)", "Status": "Synthetic"},
    {"Field": "Marketing spend (4 channels)", "Source": "Generated proxy vs. event calendar", "Status": "Synthetic"},
    {"Field": "Channel ROI ($/$ spent)", "Source": "Illustrative industry-benchmark figures", "Status": "Synthetic"},
])
st.dataframe(provenance, use_container_width=True, hide_index=True)

st.info(
    "Because spend and ROI are synthetic, the channel-level conclusions (e.g. "
    "\"email returns \\$7.80 per \\$1\") demonstrate the MMM *methodology* — they are "
    "not empirical claims about real MLB marketing. The game-factor and team-performance "
    "effects, by contrast, are estimated against real attendance."
)

st.markdown("### Model structure")
st.markdown(r"""
A Bayesian Marketing Mix Model fit with PyMC (NUTS). Attendance is modeled on the
log scale, decomposed into a baseline, additive covariate effects, and saturated
marketing contributions:

$$\log(\text{attendance}_i) \sim \mathcal{N}(\mu_i, \sigma)$$

$$\mu_i = \alpha + u_{\text{team}[i]} + \sum_m \beta^{\text{month}}_m \cdot \text{month}_{im}
+ \sum_k \beta_k x_{ik} + \sum_c \beta_c \cdot f_c(\text{spend}_{ic})$$

where:
- $u_{\text{team}}$ is a **hierarchical team intercept** (partial pooling across 30 teams)
- $x_{ik}$ are game-factor and team-performance covariates (opening day, rival, win pct, etc.)
- $f_c(\cdot)$ is the **adstock + Hill saturation** transform applied to channel $c$'s spend
""")

st.markdown("### Adstock & saturation transforms")
st.markdown(r"""
Each marketing channel's raw spend is passed through two transforms before entering
the linear predictor:

1. **Geometric adstock** — captures carry-over: spend in prior weeks still drives
   demand, decaying geometrically. $\text{adstock}_t = \sum_{l=0}^{L} \text{decay}^l \cdot \text{spend}_{t-l}$.
   Fast-decaying channels (paid social, decay≈0.3) fade quickly; broadcast (decay≈0.7)
   lingers 6-8 weeks.
2. **Hill saturation** — captures diminishing returns: doubling spend less than doubles
   impact. $\text{sat}(x) = \frac{x^\alpha}{x^\alpha + \gamma^\alpha}$.

Channel priors (`decay`, `alpha`, `gamma`) are set from MMM literature for the
entertainment / sports category. See `backend/models/adstock.py`.
""")

st.markdown("### Validation")
st.markdown("""
Model fit is reported on a **time-based holdout**: trained on the 2019, 2021-2023
seasons and evaluated on the held-out **2024** season. This is generated reproducibly by:

```bash
python -m backend.models.validate
```

which writes `backend/data/processed/validation_metrics.json`. The home page reads
these numbers directly, so the headline metrics reflect genuine out-of-sample
performance rather than asserted figures.
""")

st.markdown("### Known limitations")
st.markdown("""
- **Synthetic spend** is the biggest caveat — real media-planning data would sharpen
  channel estimates considerably.
- **No weather data** — temperature and precipitation are known attendance drivers
  not yet included.
- **Probabilistic promo schedule** — a real promo calendar would let us identify
  effects by promotion type.
- **League-wide pooling** — a single team's local marketing data would enable
  team-specific budget guidance.
""")
