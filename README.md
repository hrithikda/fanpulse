# FanPulse

Bayesian Marketing Mix Model for MLB fan demand. Models what actually drives
attendance across a season — promotional events, opponent strength, playoff
race intensity, weather, and hypothetical marketing channel spend.

Built as a learning project to understand MMM methodology in a sports context.

## What it does

- Pulls game-level attendance data for 5 MLB seasons (2019, 2021-2024) using pybaseball
- Engineers covariates: opening day, bobblehead nights, rival games, playoff race, weather
- Generates synthetic marketing spend proxies (paid social, email, broadcast, OOH)
- Fits a Bayesian MMM using PyMC with geometric adstock and Hill saturation transforms
- Produces posterior distributions over channel ROI and covariate effects
- Streamlit app for budget scenario planning and reallocation analysis

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
# fetch and process data
python backend/utils/data_loader.py

# fit the model (takes ~15-20 min first time)
python backend/models/mmm.py

# launch the dashboard
streamlit run streamlit_app/app.py
```

## Data sources

- Game logs and attendance: pybaseball (Baseball Reference wrapper)
- Team performance: pybaseball rolling standings
- Marketing spend: synthetic proxies calibrated to industry benchmarks
- Social engagement proxy: Google Trends via pytrends

## Stack

Python, PyMC, pandas, matplotlib, seaborn, Streamlit
