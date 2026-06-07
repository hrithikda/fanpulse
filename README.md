# FanPulse — Bayesian Marketing Mix Model for MLB Fan Demand

A Bayesian Marketing Mix Model (MMM) that decomposes Major League Baseball game
attendance into the factors that actually drive it — game context, team
performance, market structure, and marketing spend — and turns that
decomposition into an interactive budget-planning dashboard.

> **Data honesty.** Attendance and game results are **real**, scraped from
> Baseball Reference. Marketing spend, channel ROI, and promotional-night flags
> are **synthetic** proxies generated to demonstrate MMM methodology — they are
> not measured MLB marketing data. The dashboard's *Methodology* page documents
> exactly what is real vs. simulated. This project demonstrates the modeling
> approach end-to-end; the marketing-channel conclusions are illustrative.

---

## Overview

Marketing Mix Modeling is the industry-standard technique for attributing
outcomes (sales, sign-ups, attendance) across many simultaneous drivers and for
answering "if I move budget from channel A to channel B, what happens?" FanPulse
applies a full Bayesian MMM to **11,972 real MLB home games** across five
seasons (2019, 2021–2024) and all 30 teams.

The model is fit with PyMC using NUTS (Hamiltonian Monte Carlo). It includes a
hierarchical team intercept (partial pooling across the 30 teams), a set of
game-context and team-performance covariates, and four marketing channels passed
through **geometric adstock** (carry-over) and **Hill saturation** (diminishing
returns) transforms. A fully Bayesian variant goes a step further and *learns*
those transform parameters instead of fixing them.

Everything is wrapped in a six-page Streamlit dashboard for exploration,
diagnostics, and budget scenario planning.

---

## Problem Statement

What actually drives MLB game attendance, and how should a team allocate a fixed
seasonal marketing budget to maximize attendance (and therefore gate revenue)?

This is hard because the drivers are correlated and operate on different scales:
opening day, a pennant race, a winning streak, market size, and marketing spend
all move attendance at once, and marketing has carry-over and saturation effects
that a naïve regression cannot capture. A Bayesian MMM addresses this by
modeling the full generative structure and returning **posterior distributions**
(uncertainty included) over every effect.

---

## Key Features

- **Bayesian MMM in PyMC** — log-attendance decomposed into a baseline,
  hierarchical team intercepts, game-context and team-performance covariates, and
  saturated marketing contributions, fit with NUTS.
- **Geometric adstock + Hill saturation** — carry-over and diminishing-returns
  transforms for each marketing channel.
- **Fully Bayesian transform learning** — an alternative model
  (`mmm_bayesian.py`) places priors on and *learns* each channel's adstock
  `decay`, Hill `alpha`, and `gamma`, using a **reset-aware `pytensor.scan`** so
  carry-over does not leak across team-season boundaries.
- **Reproducible time-based holdout validation** — trains on 2019–2023, tests on
  the held-out 2024 season with leakage-safe scaling, and writes the metrics to
  JSON that the dashboard reads (no hardcoded performance claims).
- **Model diagnostics** — predicted-vs-actual, residual plots, a posterior
  predictive check, and per-team / per-month error breakdowns.
- **Budget optimizer** — a constrained `scipy` SLSQP solver finds the
  attendance-maximizing channel allocation under diminishing returns and budget
  constraints; one click pre-fills the scenario sliders.
- **Transparent data provenance** — an in-app methodology page and banners
  clearly separate real data from synthetic proxies.

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Modeling | PyMC, PyTensor, ArviZ (NUTS / HMC, MCMC diagnostics) |
| Optimization | SciPy (`optimize.minimize`, SLSQP) |
| Data | pybaseball (Baseball Reference), pandas, NumPy, scikit-learn |
| Visualization / App | Streamlit, Matplotlib, Seaborn |
| Language | Python 3.x |

---

## Project Architecture / Workflow

```
 Baseball Reference (pybaseball)
            │  data_loader.py
            ▼
   raw game logs + attendance
            │  feature_engineering.py   (win%, run diff, streaks, opening day,
            │                            rivals, playoff race, market size, ...)
            │  synthetic_spend.py        (synthetic 4-channel marketing spend)
            ▼
     games_features.csv
            │  mmm.py: preprocess()      (scale covariates, adstock + Hill on spend)
            ▼
   ┌─────────────────────────────┬──────────────────────────────┐
   │  mmm.py (fixed transforms)  │  mmm_bayesian.py (learned     │
   │  → mmm_trace.pkl            │  decay/alpha/gamma via scan)  │
   └─────────────────────────────┴──────────────────────────────┘
            │  predict.py (reconstruct μ from posterior)
            │  validate.py (2019-23 train / 2024 holdout → metrics JSON)
            ▼
        Streamlit dashboard (app.py + pages/)
        Home · EDA · Model Results · Diagnostics · Scenario Planner · Methodology
```

**Dashboard pages**

| Page | What it shows |
|------|---------------|
| Home | Overview + reproducible 2024 holdout metrics |
| EDA | Attendance distributions and raw factor effects |
| Model Results | Posterior coefficients, channel ROI, learned adstock/saturation |
| Diagnostics | Predicted vs. actual, residuals, posterior predictive check, per-team/month error |
| Scenario Planner | Manual budget reallocation + one-click optimal allocation |
| Methodology | Real vs. synthetic data provenance, model structure, limitations |

---

## Folder Structure

```
fanpulse/
├── backend/
│   ├── data/
│   │   └── processed/              # committed dataset, fitted trace, scalers, metrics
│   ├── models/
│   │   ├── adstock.py              # geometric adstock + Hill saturation + channel priors
│   │   ├── mmm.py                  # baseline Bayesian MMM (fixed transforms) + preprocessing
│   │   ├── mmm_bayesian.py         # fully Bayesian variant (learns decay/alpha/gamma)
│   │   ├── predict.py              # posterior-mean prediction reconstruction + metrics
│   │   └── validate.py             # time-based holdout validation
│   └── utils/
│       ├── data_loader.py          # pybaseball ingestion + cleaning
│       ├── feature_engineering.py  # covariate construction
│       └── synthetic_spend.py      # synthetic marketing-spend generator
├── streamlit_app/
│   ├── app.py                      # home page
│   └── pages/
│       ├── eda.py
│       ├── model_results.py
│       ├── diagnostics.py
│       ├── scenario_planner.py
│       └── methodology.py
├── notebooks/
│   └── executive_brief.md          # written summary of findings
├── requirements.txt
└── README.md
```

---

## Installation

Requires Python 3.x. A virtual environment is recommended.

```bash
git clone https://github.com/hrithikda/fanpulse.git
cd fanpulse

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

> **Note:** PyMC relies on PyTensor, which compiles C code at runtime and needs a
> working C/C++ compiler. On macOS run `xcode-select --install`; on Debian/Ubuntu
> install `build-essential`.

---

## How to Run

The repository ships with the processed dataset, fitted trace, scalers, and
validation metrics committed, so **the dashboard runs immediately** without
re-fetching data or re-fitting the model:

```bash
streamlit run streamlit_app/app.py
```

Then open http://localhost:8501.

To reproduce the full pipeline from scratch:

```bash
# 1. fetch + clean game logs from Baseball Reference (creates games.csv)
python backend/utils/data_loader.py

# 2. build features and write games_features.csv
python backend/utils/feature_engineering.py
python backend/utils/synthetic_spend.py

# 3. fit the baseline MMM (~15-25 min; writes mmm_trace.pkl + scalers)
python -m backend.models.mmm

# 4. (optional) reproduce the out-of-sample holdout metrics shown on the home page
python -m backend.models.validate

# 5. (optional) fit the fully-Bayesian adstock/saturation variant
python -m backend.models.mmm_bayesian

# 6. launch the dashboard
streamlit run streamlit_app/app.py
```

> If `streamlit` on your PATH points to a different Python than the one where you
> installed the requirements, launch it explicitly with that interpreter:
> `python -m streamlit run streamlit_app/app.py`.

---

## Environment Variables

None. The project requires no API keys, tokens, or `.env` file — pybaseball
scrapes publicly available data from Baseball Reference, and all other inputs are
generated locally.

---

## Usage Example

Reconstruct predictions and compute fit metrics from the committed trace:

```python
import pickle
from backend.models import mmm
from backend.models.predict import predict_attendance, regression_metrics

df = mmm.load_data()
df, _ = mmm.preprocess(df)

with open("backend/data/processed/mmm_trace.pkl", "rb") as f:
    trace = pickle.load(f)

pred = predict_attendance(df, trace)
print(regression_metrics(df["attendance"].values, pred))
# {'r2': ..., 'mape': ..., 'rmse': ...}
```

In the **Scenario Planner**, adjust the four channel sliders (they must sum to
100%) to see projected attendance lift and incremental gate revenue, or click
**Find optimal allocation** to let the SLSQP optimizer solve for the
attendance-maximizing split under diminishing returns.

---

## Demo / Screenshots

Run `streamlit run streamlit_app/app.py` and browse the six pages described
above. Highlights:

- **Home** — reproducible 2024 holdout metrics (R² and MAPE read from JSON).
- **Diagnostics** — posterior predictive check plus per-team error, which
  surfaces explainable outliers (e.g. Toronto's 2021 home games played in
  Buffalo/Dunedin during COVID).
- **Scenario Planner** — interactive reallocation and one-click optimization.

*(Screenshots can be added to a `docs/` folder and linked here.)*

---

## Important Implementation Details

- **Hierarchical team intercepts.** Team baselines are modeled with partial
  pooling (`team_offset ~ Normal(0, team_sigma)`), so small-sample teams are
  shrunk toward the league mean rather than overfit.
- **Leakage-safe rolling features.** Win percentage, run differential, and
  streaks are all shifted by one game before rolling, so a game never "sees" its
  own result.
- **Reset-aware adstock.** In the Bayesian variant, geometric adstock is computed
  with `pytensor.scan` and a reset mask at each team-season boundary, so a team's
  September spend never carries into another team's April. The learnable `decay`
  is passed to `scan` as a `non_sequences` to keep it out of the log-probability
  graph (a common PyMC pitfall).
- **Prediction reconstruction.** `predict.py` rebuilds the posterior-mean linear
  predictor directly from the trace, kept in lockstep with the model's `mu`
  equation, so diagnostics and validation never need to re-sample.
- **Reproducible validation.** `validate.py` performs a strict time-based split,
  fits scalers on train only, applies them to test, and serializes metrics to
  `validation_metrics.json` that the home page reads.

---

## Challenges Solved

- **Unsupported performance claims.** The original dashboard asserted a holdout
  R² of 0.81 / MAPE 8.3% with no supporting code. By reconstructing predictions
  from the posterior and running a true holdout, the real numbers were measured
  (R² ≈ 0.33, MAPE ≈ 23.9%) and the dashboard was rewired to display verifiable,
  reproducible metrics.
- **Adstock leaking across teams.** Implementing carry-over over a single
  concatenated series would bleed spend across team-season boundaries; a
  reset-aware scan fixed this.
- **Random variables inside `scan`.** A learnable decay captured as a closure
  leaked into PyMC's logp graph and raised; passing it as a `non_sequences`
  resolved it.
- **Honest treatment of synthetic data.** Synthetic spend was clearly labeled
  throughout the app and docs so channel-level results are not mistaken for
  empirical findings.

---

## Future Improvements

- **Real marketing-spend data** to replace the synthetic proxies and make the
  channel-level estimates empirical rather than illustrative.
- **Weather covariates** (NOAA daily temperature/precipitation), which are known
  attendance drivers currently omitted.
- **Real promotional calendars** instead of the probabilistic promo schedule.
- **Team-specific models** with local marketing data for franchise-level budget
  guidance, rather than league-wide pooling.
- **Posterior-predictive prediction intervals** persisted directly in the trace,
  and a forward-looking forecasting page for a hypothetical upcoming schedule.
- **Interactive charts** (Plotly/Altair) and a hosted deployment with screenshots
  in the README.

---

## Author

**Hrithik Angadi** — [github.com/hrithikda](https://github.com/hrithikda)

Built as a portfolio project to demonstrate Bayesian Marketing Mix Modeling,
probabilistic programming, and end-to-end data-science delivery in a sports
analytics context.
