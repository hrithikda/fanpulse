"""Fully Bayesian MMM: adstock and saturation parameters are *learned*.

The baseline model in ``mmm.py`` applies geometric adstock and Hill saturation
with fixed point estimates (``decay_mu``, ``alpha_mu``, ``gamma_mu``) before
sampling. That throws away uncertainty in the transform itself. This module
instead places priors on the per-channel ``decay``, ``alpha``, and ``gamma`` and
learns them jointly with everything else inside PyMC — which is what makes it a
"real" MMM.

Key implementation detail: geometric adstock is computed with a reset-aware
``pytensor.scan`` so carry-over does not bleed across team-season boundaries.

Run with:
    python -m backend.models.mmm_bayesian
"""
import os
import pickle

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import arviz as az

from backend.models import mmm
from backend.models.adstock import CHANNEL_PRIORS

CHANNELS = mmm.CHANNELS
TRACE_PATH = "backend/data/processed/mmm_trace_bayesian.pkl"


def _reset_mask(df):
    """1.0 at the first home game of each (team, season), else 0.0."""
    reset = np.zeros(len(df), dtype=float)
    starts = df.groupby(["team", "season"]).head(1).index
    reset[df.index.isin(starts)] = 1.0
    return reset


def prepare(df):
    """Scaled covariates + raw (max-scaled) spend + reset mask, all aligned."""
    df = df.copy()
    df = df.sort_values(["team", "season", "date"]).reset_index(drop=True)
    df = df.dropna(subset=["attendance"])
    df["log_attendance"] = np.log(df["attendance"])

    for col in ["home_win_pct_20g", "run_diff_10g", "win_streak",
                "days_since_last_home", "season_progress", "game_number"]:
        mn, mx = df[col].min(), df[col].max()
        df[col + "_scaled"] = (df[col] - mn) / (mx - mn + 1e-8)

    df["day_of_week_scaled"] = df["day_of_week"] / 6.0
    for m in range(4, 11):
        df[f"month_{m}"] = (df["month"] == m).astype(float)

    spend_scales = {}
    for ch in CHANNELS:
        col = f"spend_{ch}"
        scale = df[col].max()
        df[f"{ch}_raw_scaled"] = df[col] / (scale + 1e-8)
        spend_scales[ch] = float(scale)

    return df, spend_scales


def _adstock_scan(x, reset, decay):
    """Reset-aware geometric adstock: a_t = x_t + decay * a_{t-1} * (1 - reset_t).

    ``decay`` is a random variable, so it must be passed to ``scan`` as a
    non-sequence rather than captured as a closure (otherwise it leaks into the
    logp graph and PyMC raises).
    """
    def step(x_t, reset_t, prev, decay_):
        return x_t + decay_ * prev * (1.0 - reset_t)

    out, _ = pytensor.scan(
        fn=step,
        sequences=[x, reset],
        outputs_info=[pt.zeros(())],
        non_sequences=[decay],
    )
    return out


def build_model(df, spend_scales):
    y = df["log_attendance"].values
    reset = _reset_mask(df)

    X_weekend   = df["is_weekend"].values.astype(float)
    X_opening   = df["is_opening_day"].values.astype(float)
    X_rival     = df["is_rival"].values.astype(float)
    X_promo     = df["is_promo_night"].values.astype(float)
    X_playoff   = df["is_playoff_race"].values.astype(float)
    X_fireworks = df["is_fireworks"].values.astype(float)
    X_july4     = df["is_july4_week"].values.astype(float)
    X_memorial  = df["is_memorial_day_week"].values.astype(float)
    X_labor     = df["is_labor_day_week"].values.astype(float)
    X_winpct    = df["home_win_pct_20g_scaled"].values
    X_rundiff   = df["run_diff_10g_scaled"].values
    X_streak    = df["win_streak_scaled"].values
    X_market    = df["market_size"].values.astype(float) / 2.0
    X_progress  = df["season_progress_scaled"].values
    X_dow       = df["day_of_week_scaled"].values
    X_team      = df["team_encoded"].values

    month_cols = np.stack([df[f"month_{m}"].values for m in range(4, 11)], axis=1)
    raw_spend = {ch: df[f"{ch}_raw_scaled"].values for ch in CHANNELS}
    n_teams = df["team_encoded"].nunique()

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=10.0, sigma=1.0)
        team_sigma = pm.HalfNormal("team_sigma", sigma=0.5)
        team_offset = pm.Normal("team_offset", mu=0.0, sigma=team_sigma, shape=n_teams)
        b_months = pm.Normal("b_months", mu=0.0, sigma=0.2, shape=7)

        b_weekend   = pm.Normal("b_weekend",   mu=0.08, sigma=0.04)
        b_opening   = pm.Normal("b_opening",   mu=0.32, sigma=0.08)
        b_rival     = pm.Normal("b_rival",     mu=0.12, sigma=0.05)
        b_promo     = pm.Normal("b_promo",     mu=0.15, sigma=0.05)
        b_playoff   = pm.Normal("b_playoff",   mu=0.20, sigma=0.06)
        b_fireworks = pm.Normal("b_fireworks", mu=0.08, sigma=0.04)
        b_july4     = pm.Normal("b_july4",     mu=0.10, sigma=0.04)
        b_memorial  = pm.Normal("b_memorial",  mu=0.06, sigma=0.03)
        b_labor     = pm.Normal("b_labor",     mu=0.06, sigma=0.03)
        b_winpct    = pm.Normal("b_winpct",    mu=0.22, sigma=0.07)
        b_rundiff   = pm.Normal("b_rundiff",   mu=0.08, sigma=0.04)
        b_streak    = pm.Normal("b_streak",    mu=0.05, sigma=0.03)
        b_market    = pm.Normal("b_market",    mu=0.25, sigma=0.08)
        b_progress  = pm.Normal("b_progress",  mu=0.05, sigma=0.05)
        b_dow       = pm.Normal("b_dow",       mu=0.05, sigma=0.03)

        month_contribution = pm.math.dot(month_cols, b_months)

        mu = (
            alpha
            + team_offset[X_team]
            + month_contribution
            + b_weekend * X_weekend
            + b_opening * X_opening
            + b_rival * X_rival
            + b_promo * X_promo
            + b_playoff * X_playoff
            + b_fireworks * X_fireworks
            + b_july4 * X_july4
            + b_memorial * X_memorial
            + b_labor * X_labor
            + b_winpct * X_winpct
            + b_rundiff * X_rundiff
            + b_streak * X_streak
            + b_market * X_market
            + b_progress * X_progress
            + b_dow * X_dow
        )

        reset_t = pt.as_tensor_variable(reset)
        for ch in CHANNELS:
            p = CHANNEL_PRIORS[ch]
            # decay in (0,1): Beta centered near the literature prior mean
            decay = pm.Beta(f"decay_{ch}", alpha=p["decay_mu"] * 8, beta=(1 - p["decay_mu"]) * 8)
            # Hill steepness > 0
            hill_alpha = pm.Gamma(f"alpha_{ch}", mu=p["alpha_mu"], sigma=p["alpha_sigma"])
            # half-saturation point in (0,1)
            gamma = pm.Beta(f"gamma_{ch}", alpha=p["gamma_mu"] * 8, beta=(1 - p["gamma_mu"]) * 8)
            beta = pm.HalfNormal(f"beta_{ch}", sigma=p["beta_sigma"] * 3)

            x = pt.as_tensor_variable(raw_spend[ch])
            adstocked = _adstock_scan(x, reset_t, decay)
            # normalize adstock to (0,1] so gamma is comparable across channels
            ad_norm = adstocked / (pt.max(adstocked) + 1e-8)
            saturated = ad_norm ** hill_alpha / (ad_norm ** hill_alpha + gamma ** hill_alpha)
            mu = mu + beta * saturated

        sigma = pm.HalfNormal("sigma", sigma=0.3)
        pm.Normal("attendance", mu=mu, sigma=sigma, observed=y)

    return model


def fit(draws=500, tune=1000, chains=2):
    print("loading + preparing data...")
    df = mmm.load_data()
    df, spend_scales = prepare(df)
    print(f"building Bayesian-adstock model on {len(df)} games...")
    model = build_model(df, spend_scales)
    with model:
        print("sampling (scan-based adstock is slow; expect a long run)...")
        trace = pm.sample(
            draws=draws, tune=tune, chains=chains, cores=1,
            target_accept=0.95, return_inferencedata=True, progressbar=True,
        )
    os.makedirs(os.path.dirname(TRACE_PATH), exist_ok=True)
    with open(TRACE_PATH, "wb") as f:
        pickle.dump(trace, f)
    print(f"saved Bayesian trace to {TRACE_PATH}")

    learned = [f"decay_{c}" for c in CHANNELS] + \
              [f"alpha_{c}" for c in CHANNELS] + \
              [f"gamma_{c}" for c in CHANNELS]
    print(az.summary(trace, var_names=learned, round_to=3))
    return trace


if __name__ == "__main__":
    fit()
