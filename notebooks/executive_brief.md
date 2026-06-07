# FanPulse: MLB Fan Demand Analysis
## Executive Brief — Marketing Mix Model Findings

**Prepared by:** Hrithik Angadi
**Data:** 11,972 MLB home games, 2019 and 2021-2024, all 30 teams
**Model:** Bayesian MMM with geometric adstock and Hill saturation transforms

---

> **Data note:** Attendance and game results are real (Baseball Reference).
> Marketing spend, channel ROI, and promotional-night flags are **synthetic**
> proxies generated to demonstrate MMM methodology — they are not measured MLB
> marketing data. Findings below are split accordingly: game-factor and
> team-performance effects are estimated against real attendance, while
> channel-level results are illustrative.

---

## Business Question

What actually drives MLB game attendance, and how should a team allocate its
seasonal marketing budget to maximize gate revenue?

---

## Methodology

A Bayesian Marketing Mix Model was fit to game-level attendance data pulled
from Baseball Reference via pybaseball. The model decomposes log-attendance
into three layers:

**Game factors** — binary and continuous covariates capturing game context:
opening day, rival matchups, promotional nights, playoff race intensity, day of
week, and holiday weeks.

**Team performance** — rolling 20-game win percentage, 10-game run differential,
and current win/loss streak, all lagged to avoid data leakage.

**Marketing channels** — four synthetic spend channels (paid social, email,
broadcast, out-of-home) transformed through geometric adstock (carry-over decay)
and Hill saturation curves (diminishing returns) before entering the model.

All parameters were estimated using NUTS sampling in PyMC (2 chains, 1,000
draws each after 2,000 tuning steps). The production model achieved 0
divergences and max R-hat of 1.005, indicating clean convergence. A fully
Bayesian variant additionally learns the per-channel adstock/saturation
parameters rather than fixing them.

---

## Key Findings (estimated against real attendance)

### 1. Opening Day is the single strongest attendance driver

The model estimates a posterior mean coefficient of **0.427** on log-attendance,
which is roughly a **+53% lift** on the natural scale relative to an average home
game. Marketing investment concentrated around Opening Day has an outsized return
relative to any other single game.

### 2. Market size sets the ceiling

The market-size coefficient (**0.366**, ≈ **+44%** natural scale) is the largest
structural effect — stadium capacity, city population, and regional fan base
dominate any single marketing lever. Marketing is most efficiently spent on
marginal fans within an already-engaged base rather than fighting structural
limits.

### 3. Team quality compounds across a season

Rolling 20-game win percentage carries a coefficient of **0.257**. Sustained
winning meaningfully lifts attendance over the back half of a season — winning in
May fills seats in June. The effect is persistent and cumulative.

### 4. Playoff-race games drive a September surge

September/October contention games show a coefficient of **0.113** (≈ **+12%**
lift). This suggests September marketing budget is best held in reserve and
deployed contingent on playoff position.

### 5. Some "intuitive" effects are weak or absent in the data

Rival matchups show only a small effect (coefficient ≈ 0.03), and the
(synthetic) promo-night flag shows essentially no positive effect — an honest
artifact of the promo schedule being randomly generated rather than reflecting
real giveaway calendars. This is a useful reminder that a covariate only carries
signal if the underlying data does.

---

## Illustrative Channel Findings (synthetic spend — methodology demo)

> These results rest on synthetic spend data and demonstrate how an MMM would
> quantify channel performance. They are **not** empirical claims about real MLB
> marketing.

- Under the modeled efficiencies, **email and paid social** show the highest
  marginal response per dollar, while **broadcast** has the longest carry-over
  (adstock decay ≈ 0.7, ~6-8 week memory) but the lowest marginal return.
- A budget optimizer (constrained `scipy` SLSQP under diminishing returns)
  reallocates away from broadcast toward the more efficient channels, consistent
  with the baseline being over-weighted on broadcast (45%).

| Scenario | Paid Social | Email | Broadcast | OOH |
|---|---|---|---|---|
| Baseline | 25% | 5% | 45% | 25% |
| Optimizer-recommended | shifts toward paid social / email | up to cap | down | down |

The interactive Scenario Planner in the dashboard reports the projected
attendance lift and incremental gate revenue for any chosen allocation.

---

## Validation

Reported fit comes from a **time-based holdout**: trained on 2019/2021-2023 and
evaluated on the held-out 2024 season (generated reproducibly via
`python -m backend.models.validate`).

- Holdout R² (2024): **0.33**
- Holdout MAPE: **23.9%**
- Convergence: max R-hat 1.01, 0 divergences

These are honest out-of-sample numbers. Attendance is genuinely noisy at the
game level (weather, opponent, local events), so a mid-range R² is expected for
a parsimonious, interpretable model — the value here is decomposition and
inference, not point-prediction accuracy.

---

## Limitations and Next Steps

**Synthetic spend data.** Marketing channel spend was generated as a proxy
calibrated to industry benchmarks. Real spend data from a team's media-planning
system would make the channel-level estimates empirical rather than illustrative.

**No weather data.** Temperature and precipitation are known attendance drivers
excluded due to data-collection scope. Adding NOAA daily weather would likely
improve fit and reduce residual variance.

**Promo-night identification.** Promotional events were approximated
probabilistically. A complete promo calendar would allow precise identification
of promo effects by type.

**League-wide pooling.** The current model partially pools across all 30 teams.
A team-specific model with granular local marketing data would produce
actionable budget guidance for a specific franchise.

---

## Technical Notes

- Production model: PyMC, NUTS sampler, 2 chains x 1,000 draws, 2,000 tuning steps
- Convergence (production): 0 divergences, max R-hat 1.005
- Holdout (2024): R² 0.33, MAPE 23.9%, max R-hat 1.01, 0 divergences
- Code and data pipeline: github.com/hrithikda/fanpulse
