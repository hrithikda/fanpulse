"""Reconstruct model predictions from a fitted trace.

The MMM in ``mmm.py`` does not persist a ``posterior_predictive`` group, so this
module rebuilds the linear predictor (``mu``) on the log-attendance scale from
the posterior-mean coefficients. This lets the dashboard and the holdout
validation script generate predictions without re-running NUTS sampling.

The ``mu`` equation here is kept in lockstep with ``mmm.build_model``.
"""
import numpy as np

from backend.models.mmm import CHANNELS

# month dummies span April (4) through October (10)
MONTHS = list(range(4, 11))


def posterior_means(trace):
    """Collapse the posterior to a dict of scalar / vector means per variable."""
    post = trace.posterior
    params = {}
    for var in post.data_vars:
        params[var] = post[var].mean(dim=["chain", "draw"]).values
    return params


def compute_mu(df, params):
    """Compute posterior-mean ``mu`` (log-attendance) for a preprocessed frame.

    ``df`` must already carry the scaled covariates and normalized transformed
    channel columns produced by ``mmm.preprocess`` / ``mmm.transform_with_scalers``.
    """
    team_offset = np.asarray(params["team_offset"])
    b_months = np.asarray(params["b_months"])

    month_cols = np.stack([df[f"month_{m}"].values for m in MONTHS], axis=1)
    month_contribution = month_cols @ b_months

    mu = (
        float(params["alpha"])
        + team_offset[df["team_encoded"].values]
        + month_contribution
        + float(params["b_weekend"]) * df["is_weekend"].values.astype(float)
        + float(params["b_opening"]) * df["is_opening_day"].values.astype(float)
        + float(params["b_rival"]) * df["is_rival"].values.astype(float)
        + float(params["b_promo"]) * df["is_promo_night"].values.astype(float)
        + float(params["b_playoff"]) * df["is_playoff_race"].values.astype(float)
        + float(params["b_fireworks"]) * df["is_fireworks"].values.astype(float)
        + float(params["b_july4"]) * df["is_july4_week"].values.astype(float)
        + float(params["b_memorial"]) * df["is_memorial_day_week"].values.astype(float)
        + float(params["b_labor"]) * df["is_labor_day_week"].values.astype(float)
        + float(params["b_winpct"]) * df["home_win_pct_20g_scaled"].values
        + float(params["b_rundiff"]) * df["run_diff_10g_scaled"].values
        + float(params["b_streak"]) * df["win_streak_scaled"].values
        + float(params["b_market"]) * (df["market_size"].values.astype(float) / 2.0)
        + float(params["b_progress"]) * df["season_progress_scaled"].values
        + float(params["b_dow"]) * df["day_of_week_scaled"].values
    )

    for ch in CHANNELS:
        mu = mu + float(params[f"beta_{ch}"]) * df[f"{ch}_transformed"].values

    return mu


def predict_attendance(df, trace):
    """Return predicted attendance (natural scale) for a preprocessed frame."""
    params = posterior_means(trace)
    mu = compute_mu(df, params)
    return np.exp(mu)


def compute_mu_draws(df, trace, n_draws=60, seed=0):
    """Vectorized ``mu`` across a random subset of posterior draws.

    Returns an array of shape ``(n_draws, n_rows)`` plus the matching ``sigma``
    draws, for use in posterior predictive checks.
    """
    post = trace.posterior
    n_chain = post.sizes["chain"]
    n_draw = post.sizes["draw"]
    total = n_chain * n_draw

    rng = np.random.default_rng(seed)
    flat_idx = rng.choice(total, size=min(n_draws, total), replace=False)
    chains = flat_idx // n_draw
    draws = flat_idx % n_draw

    def stk(var):
        # -> (K,) for scalars, (K, dim) for vectors
        arr = post[var].values  # (chain, draw, ...)
        return arr[chains, draws]

    K = len(flat_idx)
    team_offset = stk("team_offset")          # (K, 30)
    b_months = stk("b_months")                # (K, 7)
    team_idx = df["team_encoded"].values      # (N,)

    month_cols = np.stack([df[f"month_{m}"].values for m in MONTHS], axis=1)  # (N,7)

    mu = (
        stk("alpha")[:, None]
        + team_offset[:, team_idx]
        + b_months @ month_cols.T
    )

    linear_terms = {
        "b_weekend": df["is_weekend"].values.astype(float),
        "b_opening": df["is_opening_day"].values.astype(float),
        "b_rival": df["is_rival"].values.astype(float),
        "b_promo": df["is_promo_night"].values.astype(float),
        "b_playoff": df["is_playoff_race"].values.astype(float),
        "b_fireworks": df["is_fireworks"].values.astype(float),
        "b_july4": df["is_july4_week"].values.astype(float),
        "b_memorial": df["is_memorial_day_week"].values.astype(float),
        "b_labor": df["is_labor_day_week"].values.astype(float),
        "b_winpct": df["home_win_pct_20g_scaled"].values,
        "b_rundiff": df["run_diff_10g_scaled"].values,
        "b_streak": df["win_streak_scaled"].values,
        "b_market": df["market_size"].values.astype(float) / 2.0,
        "b_progress": df["season_progress_scaled"].values,
        "b_dow": df["day_of_week_scaled"].values,
    }
    for var, x in linear_terms.items():
        mu = mu + stk(var)[:, None] * x[None, :]

    for ch in CHANNELS:
        mu = mu + stk(f"beta_{ch}")[:, None] * df[f"{ch}_transformed"].values[None, :]

    sigma = stk("sigma")  # (K,)
    return mu, sigma


def regression_metrics(y_true, y_pred):
    """R^2 and MAPE on the natural attendance scale."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"r2": float(r2), "mape": float(mape), "rmse": rmse}
