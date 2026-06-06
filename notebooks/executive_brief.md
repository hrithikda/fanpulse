# FanPulse: MLB Fan Demand Analysis
## Executive Brief — Marketing Mix Model Findings

**Prepared by:** Hrithik Angadi
**Data:** 11,972 MLB home games, 2019 and 2021-2024, all 30 teams
**Model:** Bayesian MMM with geometric adstock and Hill saturation transforms

---

## Business Question

What actually drives MLB game attendance, and how should a team allocate its
seasonal marketing budget to maximize gate revenue?

---

## Methodology

A Bayesian Marketing Mix Model was fit to game-level attendance data pulled
from Baseball Reference via pybaseball. The model decomposes attendance into
three layers:

**Game factors** — binary and continuous covariates capturing the game context:
opening day, rival matchups, promotional nights, playoff race intensity, day of
week, and holiday weeks.

**Team performance** — rolling 20-game win percentage, 10-game run differential,
and current win/loss streak, all lagged to avoid data leakage.

**Marketing channels** — four hypothetical spend channels (paid social, email,
broadcast, out-of-home) transformed through geometrik (carry-over decay)
and Hill saturation curves (diminishing returns) before entering the model.

All parameters were estimated using NUTS sampling in PyMC (2 chains, 1,000
draws each after 2,000 tuning steps). The model achieved 0 divergences and
max R-hat of 1.005, indicating clean convergence.

---

## Key Findings

### 1. Opening Day is the single strongest attendance driver

The model estimates a posterior mean coefficient of 0.427 on log-attendance,
translating to a **+42.7% attendance premium** over an average home game.
At mean attendance of 26,572, that is roughly 11,300 additional fans. Marketing
investment concentrated in the two weeks before Opening Day has an outsized
return relative to any other single game.

### 2. Team quality compounds across a season

Rolling 20-game win percentage carries a coefficient of 0.257, meaning a team
that improves from a .400 winning pace to a .600 pace can expect roughly
**+2,600 additional fans per home game** on average. This effect is persistent
and cumulative — winning in May fills seats in June.

### 3. Playoff race intensity drives a September surge

Games played when a team is in contention (within 5 games of a playoff spot
in September) show a **+11.3% attendance lift**. This is the second largest
game-factor effect after Opening Day and suggests that September marketing
budgets should be held in reserve and deployed contingent on playoff position.

### 4. Market size explains more variance than any single marketing channel

The market size coefficient (0.366) dwarfs individual channel betas, confirming
that structural factors — stadium capacity, city population, regional fan base —
set a ceiling that marketing cannot overcome on its own. Marketing dollars are
most efficiently spent on marginal fans within an already-engaged base.

### 5. Email delivers the highest ROI despite the smallest budget allocation

At an estimated $7.80 return per $1 spent, email outperforms every other
channel on a marginal basis. The current baseline allocation gives email only
5% of the budget ($105K/season). Doubling the email allocation to 10% while
trimming broadcast by 5 points projects a net positive attendance impact at
significantly lost.

### 6. Broadcast has the longest carry-over but the lowest marginal return

The broadcast adstock decay parameter of 0.7 means a TV campaign runs for
roughly 6-8 weeks in the audience's memory — the longest of any channel.
However, the marginal fan impact per $1K of spend is the lowest (0.2 fans/$1K
vs 1.9 for paid social). Broadcast is best used for broad awareness at season
launch, not for driving incremental attendance game-by-game.

---

## Budget Reallocation Recommendation

The baseline allocation (45% broadcast, 25% paid social, 25% OOH, 5% email)
over-weights the lowest-ROI channel. A reallocation scenario shifting 18% of
broadcast spend toward email and paid social projects:

| Scenario | Paid Social | Email | Broadcast | OOH | Projected Lift |
|---|---|---|---|---|---|
| Baseline | 25% | 5% | 45% | 25% | — |
| Optimized | 35% | 15% | 27% | 23% | +11.4% |

At mean attendance of 26,572 and an average ticket price of $35, an 11.4% lift
across 81 home games represents approximately **$3.2M in incntal gate
revenue** per season per team.

---

## Limitations and Next Steps

**Synthetic spend data.** Marketing channel spend was generated as a proxy
calibrated to industry benchmarks. Real spend data from a team's media planning
system would sharpen channel-level estimates considerably.

**No weather data.** Temperature and precipitation are known attendance drivers
that were excluded due to data collection scope. Adding NOAA daily weather
would likely improve model fit and reduce residual variance.

**Promo night identification.** Promotional events (bobblehead nights, giveaways)
were approximated probabilistically. A complete promo calendar from a team's
CRM would allow precise identification of promo effects by type.

**Single-team application.** The current model pools across all 30 teams. A
team-specific model with more granular local marketing data would produce
actionable budget guidance for a specific franchise rather than league-wide
estimates.

---

## Technical Notes

- Model fit: PyMC 6.0, NUTS sampler, 2 chains x 1,000 draws, 2,000 tuning steps
- Convergence: 0 divergences, max R-hat 1.005, min ESS bulk 800+
- Holdout R² (2024 season): 0.81
- MAPE: 8.3%
- Code and data pipeline: github.com/hrithikda/fanpulse
