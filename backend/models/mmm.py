import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from backend.utils.feature_engineering import build_features
from backend.utils.synthetic_spend import add_spend_features
from backend.models.adstock import apply_adstock_saturation, CHANNEL_PRIORS

CHANNELS = ['paid_social', 'email', 'broadcast', 'ooh']
MODEL_PATH = 'backend/data/processed/mmm_trace.pkl'
SCALER_PATH = 'backend/data/processed/mmm_scalers.pkl'

def load_data():
    path = 'backend/data/processed/games_features.csv'
    if not os.path.exists(path):
        raise FileNotFoundError("run feature_engineering.py first")
    df = pd.read_csv(path, parse_dates=['date'])
    df = add_spend_features(df)
    return df

def preprocess(df):
    df = df.copy()
    df = df.sort_values(['team', 'season', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['attendance'])

    df['log_attendance'] = np.log(df['attendance'])

    scalers = {}
    for col in ['home_win_pct_20g', 'run_diff_10g', 'win_streak',
                'days_since_last_home', 'season_progress', 'game_number']:
        mn, mx = df[col].min(), df[col].max()
        df[col + '_scaled'] = (df[col] - mn) / (mx - mn + 1e-8)
        scalers[col] = (mn, mx)

    df['day_of_week_scaled'] = df['day_of_week'] / 6.0

    # month dummies to soak up seasonal attendance patterns
    # this prevents playoff_race and promo from picking up seasonal noise
    for m in range(4, 11):
        df[f'month_{m}'] = (df['month'] == m).astype(float)

    adstocked_channels = {ch: [] for ch in CHANNELS}
    for (team, season), grp in df.groupby(['team', 'season']):
        idx = grp.index
        for ch in CHANNELS:
            p = CHANNEL_PRIORS[ch]
            transformed = apply_adstock_saturation(
                grp[f'spend_{ch}'].values,
                decay=p['decay_mu'],
                alpha=p['alpha_mu'],
                gamma=p['gamma_mu']
            )
            adstocked_channels[ch].extend(list(zip(idx, transformed)))

    for ch in CHANNELS:
        mapping = dict(adstocked_channels[ch])
        df[f'{ch}_transformed'] = df.index.map(mapping)

    # normalize transformed channels
    for ch in CHANNELS:
        col = f'{ch}_transformed'
        mn, mx = df[col].min(), df[col].max()
        df[col] = (df[col] - mn) / (mx - mn + 1e-8)
        scalers[f'{ch}_transformed'] = (mn, mx)

    return df, scalers

def build_model(df):
    y = df['log_attendance'].values
    n = len(y)

    X_weekend   = df['is_weekend'].values.astype(float)
    X_opening   = df['is_opening_day'].values.astype(float)
    X_rival     = df['is_rival'].values.astype(float)
    X_promo     = df['is_promo_night'].values.astype(float)
    X_playoff   = df['is_playoff_race'].values.astype(float)
    X_fireworks = df['is_fireworks'].values.astype(float)
    X_july4     = df['is_july4_week'].values.astype(float)
    X_memorial  = df['is_memorial_day_week'].values.astype(float)
    X_labor     = df['is_labor_day_week'].values.astype(float)
    X_winpct    = df['home_win_pct_20g_scaled'].values
    X_rundiff   = df['run_diff_10g_scaled'].values
    X_streak    = df['win_streak_scaled'].values
    X_market    = df['market_size'].values.astype(float) / 2.0
    X_progress  = df['season_progress_scaled'].values
    X_dow       = df['day_of_week_scaled'].values
    X_team      = df['team_encoded'].values

    month_cols = np.stack([df[f'month_{m}'].values for m in range(4, 11)], axis=1)

    channel_data = {ch: df[f'{ch}_transformed'].values for ch in CHANNELS}
    n_teams = df['team_encoded'].nunique()
    n_months = 7

    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=10.0, sigma=1.0)

        team_sigma  = pm.HalfNormal('team_sigma', sigma=0.5)
        team_offset = pm.Normal('team_offset', mu=0.0, sigma=team_sigma, shape=n_teams)

        # month coefficients soak up baseline seasonal variation
        # so promo/playoff effects are identified cleanly
        b_months = pm.Normal('b_months', mu=0.0, sigma=0.2, shape=n_months)

        b_weekend   = pm.Normal('b_weekend',   mu=0.08,  sigma=0.04)
        b_opening   = pm.Normal('b_opening',   mu=0.32,  sigma=0.08)
        b_rival     = pm.Normal('b_rival',     mu=0.12,  sigma=0.05)
        b_promo     = pm.Normal('b_promo',     mu=0.15,  sigma=0.05)
        b_playoff   = pm.Normal('b_playoff',   mu=0.20,  sigma=0.06)
        b_fireworks = pm.Normal('b_fireworks', mu=0.08,  sigma=0.04)
        b_july4     = pm.Normal('b_july4',     mu=0.10,  sigma=0.04)
        b_memorial  = pm.Normal('b_memorial',  mu=0.06,  sigma=0.03)
        b_labor     = pm.Normal('b_labor',     mu=0.06,  sigma=0.03)
        b_winpct    = pm.Normal('b_winpct',    mu=0.22,  sigma=0.07)
        b_rundiff   = pm.Normal('b_rundiff',   mu=0.08,  sigma=0.04)
        b_streak    = pm.Normal('b_streak',    mu=0.05,  sigma=0.03)
        b_market    = pm.Normal('b_market',    mu=0.25,  sigma=0.08)
        b_progress  = pm.Normal('b_progress',  mu=0.05,  sigma=0.05)
        b_dow       = pm.Normal('b_dow',       mu=0.05,  sigma=0.03)

        beta_paid_social = pm.HalfNormal('beta_paid_social', sigma=0.15)
        beta_email       = pm.HalfNormal('beta_email',       sigma=0.20)
        beta_broadcast   = pm.HalfNormal('beta_broadcast',   sigma=0.10)
        beta_ooh         = pm.HalfNormal('beta_ooh',         sigma=0.08)

        month_contribution = pm.math.dot(month_cols, b_months)

        mu = (
            alpha +
            team_offset[X_team] +
            month_contribution +
            b_weekend   * X_weekend +
            b_opening   * X_opening +
            b_rival     * X_rival +
            b_promo     * X_promo +
            b_playoff   * X_playoff +
            b_fireworks * X_fireworks +
            b_july4     * X_july4 +
            b_memorial  * X_memorial +
            b_labor     * X_labor +
            b_winpct    * X_winpct +
            b_rundiff   * X_rundiff +
            b_streak    * X_streak +
            b_market    * X_market +
            b_progress  * X_progress +
            b_dow       * X_dow +
            beta_paid_social * channel_data['paid_social'] +
            beta_email       * channel_data['email'] +
            beta_broadcast   * channel_data['broadcast'] +
            beta_ooh         * channel_data['ooh']
        )

        sigma = pm.HalfNormal('sigma', sigma=0.3)
        pm.Normal('attendance', mu=mu, sigma=sigma, observed=y)

    return model

def fit_model(df, draws=1000, tune=2000, chains=2):
    model = build_model(df)
    with model:
        print("sampling - this will take 20-25 minutes...")
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=1,
            target_accept=0.92,
            return_inferencedata=True,
            progressbar=True,
        )
    return model, trace

def save_artifacts(trace, scalers):
    os.makedirs('backend/data/processed', exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(trace, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scalers, f)
    print(f"saved trace to {MODEL_PATH}")

def load_artifacts():
    with open(MODEL_PATH, 'rb') as f:
        trace = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scalers = pickle.load(f)
    return trace, scalers

def print_summary(trace):
    print("\nmodel summary (key parameters):")
    params = [
        'b_opening', 'b_rival', 'b_promo', 'b_playoff',
        'b_winpct', 'b_market',
        'beta_paid_social', 'beta_email', 'beta_broadcast', 'beta_ooh'
    ]
    summary = az.summary(trace, var_names=params, round_to=3)
    print(summary)

    print("\nchannel betas (posterior mean):")
    for ch in CHANNELS:
        samples = trace.posterior[f'beta_{ch}'].values.flatten()
        print(f"  {ch}: {samples.mean():.3f} +/- {samples.std():.3f}")

if __name__ == '__main__':
    print("loading data...")
    df = load_data()
    print("preprocessing...")
    df, scalers = preprocess(df)
    print(f"fitting model on {len(df)} games...")
    model, trace = fit_model(df, draws=1000, tune=2000, chains=2)
    save_artifacts(trace, scalers)
    print_summary(trace)