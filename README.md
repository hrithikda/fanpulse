# FanPulse

Bayesian Marketing Mix Model for MLB fan demand. Models what actually drives
attendance across a season — promotional events, opponent strength, playoff
race intensity, and hypothetical marketing channel spend.

Built as a learning project to understand MMM methodology in a sports context.

> **Data honesty:** Attendance and game results are **real** (Baseball Reference).
> Marketing spend, channel ROI, and promo-night flags are **synthetic** proxies
> used to demonstrate MMM methodology — they are not measured MLB marketing data.
> See the **Methodology** page in the dashboard for the full provenance breakdown.

## What it does

- Pulls game-level attendance data for 5 MLB seasons (2019, 2021-2024) using pybaseball
- Engineers covariates: opening day, bobblehead nights, rival games, playoff race
- Generates synthetic marketing spend proxies (paid social, email, broadcast, OOH)
- Fits a Bayesian MMM using PyMC with geometric adstock and Hill saturation transforms
- A **fully Bayesian variant** learns the adstock/saturation parameters themselves
  (per-channel `decay`, `alpha`, `gamma`) via a reset-aware `pytensor.scan`
- **Time-based holdout validation** (train 2019-2023, test 2024) writes reproducible
  out-of-sample metrics that the dashboard reads directly
- Produces posterior distributions over channel ROI and covariate effects
- Streamlit app with EDA, model results, **model diagnostics** (predicted-vs-actual,
  residuals, posterior predictive checks, per-team/month error), a **budget optimizer**
  (scipy SLSQP under diminishing returns), and a **methodology / data-provenance** page

## Dashboard pages

| Page | What it shows |
|------|---------------|
| Home | Project overview + reproducible holdout metrics |
| EDA | Attendance distributions and raw factor effects |
| Model Results | Posterior coefficients, channel ROI, learned adstock/saturation |
| Diagnostics | Predicted vs. actual, residuals, PPC, per-team/month error |
| Scenario Planner | Manual reallocation + one-click optimal allocation |
| Methodology | What's real vs. synthetic, model structure, limitations |

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
# fetch and process data
python backend/utils/data_loader.py

# fit the model (takes ~15-20 min first time)
python -m backend.models.mmm

# (optional) reproduce the out-of-sample holdout metrics shown on the home page
python -m backend.models.validate

# (optional) fit the fully-Bayesian adstock/saturation variant
python -m backend.models.mmm_bayesian

# launch the dashboard
streamlit run streamlit_app/app.py
```

The repository ships with the processed dataset, fitted trace, and validation
metrics committed, so the dashboard runs immediately after `pip install`.

## Data sources

- Game logs and attendance: pybaseball (Baseball Reference wrapper)
- Team performance: pybaseball rolling standings
- Marketing spend: synthetic proxies calibrated to industry benchmarks
- Social engagement proxy: Google Trends via pytrends

## Stack

Python, PyMC, pandas, matplotlib, seaborn, Streamlit
