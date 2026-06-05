import pandas as pd
import numpy as np
from pybaseball import schedule_and_record
from tqdm import tqdm
import os
import time

TEAMS = [
    'NYY', 'BOS', 'TBR', 'TOR', 'BAL',
    'CLE', 'CHW', 'DET', 'KCR', 'MIN',
    'HOU', 'LAA', 'OAK', 'SEA', 'TEX',
    'ATL', 'MIA', 'NYM', 'PHI', 'WSN',
    'CHC', 'CIN', 'MIL', 'PIT', 'STL',
    'ARI', 'COL', 'LAD', 'SDP', 'SFG'
]

SEASONS = [2019, 2021, 2022, 2023, 2024]

RIVALS = {
    'NYY': ['BOS'], 'BOS': ['NYY'],
    'CHC': ['STL', 'MIL'], 'STL': ['CHC'],
    'LAD': ['SFG', 'SDP'], 'SFG': ['LAD'],
    'HOU': ['TEX'], 'TEX': ['HOU'],
    'NYM': ['NYY', 'PHI'], 'PHI': ['NYM'],
    'CLE': ['CHW'], 'CHW': ['CLE'],
}

def fetch_season(team, year):
    try:
        df = schedule_and_record(year, team)
        time.sleep(0.5)
        return df
    except Exception as e:
        print(f"failed {team} {year}: {e}")
        return None

def clean_game_log(df, team, year):
    df = df.copy()
    df['team'] = team
    df['season'] = year

    home_games = df[df['Home_Away'] == 'Home'].copy()

    home_games = home_games.rename(columns={
        'Date': 'date',
        'Opp': 'opponent',
        'Attendance': 'attendance',
        'W/L': 'result',
        'R': 'runs_scored',
        'RA': 'runs_allowed',
        'Win': 'winning_pitcher',
        'Loss': 'losing_pitcher',
    })

    keep = ['date', 'team', 'season', 'opponent', 'attendance',
            'result', 'runs_scored', 'runs_allowed']
    home_games = home_games[[c for c in keep if c in home_games.columns]]

    home_games['attendance'] = pd.to_numeric(
        home_games['attendance'].astype(str).str.replace(',', ''), errors='coerce'
    )
    home_games = home_games[home_games['attendance'] > 0].dropna(subset=['attendance'])

    home_games['date'] = pd.to_datetime(
        home_games['date'].astype(str).str.extract(r'([A-Za-z]+ \d+)')[0] + f' {year}',
        format='%b %d %Y',
        errors='coerce'
    )
    home_games = home_games.dropna(subset=['date'])

    return home_games

def add_basic_features(df):
    df = df.sort_values(['team', 'date']).reset_index(drop=True)

    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([4, 5, 6]).astype(int)

    df['is_opening_day'] = 0
    for (team, season), grp in df.groupby(['team', 'season']):
        first_idx = grp.index[0]
        df.loc[first_idx, 'is_opening_day'] = 1

    df['is_rival'] = df.apply(
        lambda r: 1 if r['opponent'] in RIVALS.get(r['team'], []) else 0, axis=1
    )

    df['is_playoff_race'] = (
        (df['month'] == 9) | (df['month'] == 10)
    ).astype(int)

    df['game_number'] = df.groupby(['team', 'season']).cumcount() + 1

    return df

def load_all_seasons(force_refresh=False):
    out_path = 'backend/data/processed/games.csv'

    if os.path.exists(out_path) and not force_refresh:
        print(f"loading cached data from {out_path}")
        return pd.read_csv(out_path, parse_dates=['date'])

    all_dfs = []
    for year in SEASONS:
        print(f"\nfetching {year}...")
        for team in tqdm(TEAMS):
            raw = fetch_season(team, year)
            if raw is not None and len(raw) > 0:
                cleaned = clean_game_log(raw, team, year)
                if len(cleaned) > 0:
                    all_dfs.append(cleaned)

    df = pd.concat(all_dfs, ignore_index=True)
    df = add_basic_features(df)

    os.makedirs('backend/data/processed', exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nsaved {len(df)} games to {out_path}")

    return df

if __name__ == '__main__':
    df = load_all_seasons(force_refresh=True)
    print(df.head())
    print(f"\nshape: {df.shape}")
    print(f"seasons: {sorted(df['season'].unique())}")
    print(f"teams: {df['team'].nunique()}")
    print(f"avg attendance: {df['attendance'].mean():,.0f}")