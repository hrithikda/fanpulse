import pandas as pd
import numpy as np

# spend figures calibrated to MLB team marketing budget benchmarks
# mid-market team spends roughly $8-12M/season on marketing
# channel splits based on sports marketing industry reports

ANNUAL_BUDGET = 10_000_000
CHANNEL_SPLITS = {
    'paid_social': 0.25,
    'email': 0.05,
    'broadcast': 0.45,
    'out_of_home': 0.25,
}

def generate_paid_social(df):
    # paid social spikes around promos, opening day, weekends
    base = (ANNUAL_BUDGET * CHANNEL_SPLITS['paid_social']) / len(df)
    spend = np.full(len(df), base)

    spend += df['is_promo_night'].values * base * 3.0
    spend += df['is_opening_day'].values * base * 8.0
    spend += df['is_weekend'].values * base * 1.5
    spend += df['is_rival'].values * base * 2.0
    spend += df['is_playoff_race'].values * base * 2.5
    spend += df['is_july4_week'].values * base * 2.0

    noise = np.random.RandomState(7).normal(0, base * 0.2, len(df))
    spend = np.clip(spend + noise, base * 0.3, base * 15)
    return spend

def generate_email(df):
    # email volume is steady with spikes, low cost channel
    base = (ANNUAL_BUDGET * CHANNEL_SPLITS['email']) / len(df)
    spend = np.full(len(df), base)

    spend += df['is_promo_night'].values * base * 4.0
    spend += df['is_opening_day'].values * base * 6.0
    spend += df['is_weekend'].values * base * 0.8
    spend += df['is_playoff_race'].values * base * 3.0

    noise = np.random.RandomState(13).normal(0, base * 0.15, len(df))
    spend = np.clip(spend + noise, base * 0.2, base * 10)
    return spend

def generate_broadcast(df):
    # broadcast is heavy early season and around big series, slow adstock decay
    base = (ANNUAL_BUDGET * CHANNEL_SPLITS['broadcast']) / len(df)
    spend = np.full(len(df), base)

    # heavier in april/may for season launch
    early_season = (df['month'].values <= 5).astype(float)
    spend += early_season * base * 1.8

    spend += df['is_opening_day'].values * base * 10.0
    spend += df['is_rival'].values * base * 1.5
    spend += df['is_playoff_race'].values * base * 2.0
    spend += df['is_july4_week'].values * base * 1.5

    noise = np.random.RandomState(21).normal(0, base * 0.1, len(df))
    spend = np.clip(spend + noise, base * 0.5, base * 12)
    return spend

def generate_ooh(df):
    # out of home is steady base, bumps around promos and holidays
    base = (ANNUAL_BUDGET * CHANNEL_SPLITS['out_of_home']) / len(df)
    spend = np.full(len(df), base)

    spend += df['is_promo_night'].values * base * 1.5
    spend += df['is_july4_week'].values * base * 2.0
    spend += df['is_memorial_day_week'].values * base * 1.8
    spend += df['is_labor_day_week'].values * base * 1.5
    spend += df['is_opening_day'].values * base * 4.0

    noise = np.random.RandomState(33).normal(0, base * 0.12, len(df))
    spend = np.clip(spend + noise, base * 0.4, base * 8)
    return spend

def add_spend_features(df):
    df = df.copy()
    np.random.seed(42)

    df['spend_paid_social'] = generate_paid_social(df)
    df['spend_email'] = generate_email(df)
    df['spend_broadcast'] = generate_broadcast(df)
    df['spend_ooh'] = generate_ooh(df)

    df['spend_total'] = (
        df['spend_paid_social'] +
        df['spend_email'] +
        df['spend_broadcast'] +
        df['spend_ooh']
    )

    return df

if __name__ == '__main__':
    df = pd.read_csv('backend/data/processed/games_features.csv', parse_dates=['date'])
    df = add_spend_features(df)
    df.to_csv('backend/data/processed/games_features.csv', index=False)

    print(f"spend features added: {df.shape}")
    print("\nannual spend by channel (mean per game x 81 home games x 30 teams):")
    for ch in ['paid_social', 'email', 'broadcast', 'ooh']:
        col = f'spend_{ch}'
        ann = df[col].mean() * 81
        print(f"  {ch}: ${ann:,.0f}/season per team")

    print(f"\ntotal spend range: ${df['spend_total'].min():,.0f} - ${df['spend_total'].max():,.0f} per game")
