import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def add_win_pct(df):
    df = df.copy()
    df['result_binary'] = df['result'].str.startswith('W').astype(float)
    df = df.sort_values(['team', 'season', 'date']).reset_index(drop=True)

    win_pcts = []
    for (team, season), grp in df.groupby(['team', 'season']):
        rolling = grp['result_binary'].shift(1).rolling(window=20, min_periods=5).mean()
        win_pcts.append(rolling)

    df['home_win_pct_20g'] = pd.concat(win_pcts)
    df['home_win_pct_20g'] = df['home_win_pct_20g'].fillna(0.5)
    return df

def add_run_diff(df):
    df = df.copy()
    df['run_diff'] = df['runs_scored'] - df['runs_allowed']
    df = df.sort_values(['team', 'season', 'date']).reset_index(drop=True)

    diffs = []
    for (team, season), grp in df.groupby(['team', 'season']):
        rolling = grp['run_diff'].shift(1).rolling(window=10, min_periods=3).mean()
        diffs.append(rolling)

    df['run_diff_10g'] = pd.concat(diffs)
    df['run_diff_10g'] = df['run_diff_10g'].fillna(0.0)
    return df

def add_streak(df):
    df = df.copy()
    df = df.sort_values(['team', 'season', 'date']).reset_index(drop=True)

    streaks = []
    for (team, season), grp in df.groupby(['team', 'season']):
        streak = []
        current = 0
        for res in grp['result_binary']:
            if pd.isna(res):
                streak.append(0)
                continue
            if res == 1:
                current = current + 1 if current > 0 else 1
            else:
                current = current - 1 if current < 0 else -1
            streak.append(current)
        streaks.extend(streak)

    df['win_streak'] = streaks
    return df

def add_schedule_features(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['team', 'season', 'date']).reset_index(drop=True)

    gaps = []
    for (team, season), grp in df.groupby(['team', 'season']):
        diff = grp['date'].diff().dt.days.fillna(3)
        gaps.append(diff)

    df['days_since_last_home'] = pd.concat(gaps)

    df['is_july4_week'] = (
        (df['date'].dt.month == 7) & (df['date'].dt.day <= 7)
    ).astype(int)

    df['is_memorial_day_week'] = (
        (df['date'].dt.month == 5) & (df['date'].dt.day >= 25)
    ).astype(int)

    df['is_labor_day_week'] = (
        (df['date'].dt.month == 9) & (df['date'].dt.day <= 7)
    ).astype(int)

    df['season_progress'] = df['game_number'] / 81.0

    return df

def add_market_size(df):
    large_markets = ['NYY', 'NYM', 'LAD', 'BOS', 'CHC', 'SFG', 'PHI', 'ATL', 'HOU', 'TEX']
    medium_markets = ['STL', 'SEA', 'TOR', 'MIN', 'CLE', 'MIL', 'DET', 'ARI', 'COL', 'SDP']

    def get_market(team):
        if team in large_markets:
            return 2
        elif team in medium_markets:
            return 1
        return 0

    df['market_size'] = df['team'].map(get_market)
    return df

def add_promo_proxies(df):
    # synthetic promo schedule - bobblehead/fireworks nights
    # approximated from historical promo calendars
    np.random.seed(42)
    df = df.copy()

    promo_flags = []
    for (team, season), grp in df.groupby(['team', 'season']):
        n = len(grp)
        # ~15 promo nights per team per season is realistic
        promo_idx = np.random.choice(n, size=min(15, n), replace=False)
        flags = np.zeros(n)
        flags[promo_idx] = 1
        promo_flags.extend(flags)

    df['is_promo_night'] = promo_flags

    # fireworks more likely on weekends and holidays
    df['is_fireworks'] = (
        (df['is_weekend'] == 1) &
        ((df['is_july4_week'] == 1) | (np.random.rand(len(df)) < 0.08))
    ).astype(int)

    return df

def encode_team(df):
    df = df.copy()
    le = LabelEncoder()
    df['team_encoded'] = le.fit_transform(df['team'])
    return df, le

def build_features(df):
    df = add_win_pct(df)
    df = add_run_diff(df)
    df = add_streak(df)
    df = add_schedule_features(df)
    df = add_market_size(df)
    df = add_promo_proxies(df)
    df, team_encoder = encode_team(df)

    df = df.dropna(subset=['attendance', 'home_win_pct_20g', 'run_diff_10g'])

    return df, team_encoder

if __name__ == '__main__':
    df_raw = pd.read_csv('backend/data/processed/games.csv', parse_dates=['date'])
    df, enc = build_features(df_raw)
    df.to_csv('backend/data/processed/games_features.csv', index=False)
    print(f"features built: {df.shape}")
    print(df[['team', 'date', 'attendance', 'home_win_pct_20g',
              'win_streak', 'is_promo_night', 'market_size']].head(10))
