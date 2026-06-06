import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="EDA — FanPulse", layout="wide")
st.title("Exploratory Data Analysis")

@st.cache_data
def load_data():
    df = pd.read_csv('backend/data/processed/games_features.csv', parse_dates=['date'])
    return df

df = load_data()

st.markdown(f"**{len(df):,} home games** across {df['season'].nunique()} seasons, {df['team'].nunique()} teams.")

st.markdown("### Attendance Distribution by Season")
fig, ax = plt.subplots(figsize=(10, 4))
for season in sorted(df['season'].unique()):
    vals = df[df['season'] == season]['attendance'].dropna()
    ax.hist(vals, bins=40, alpha=0.5, label=str(season))
ax.set_xlabel("Attendance")
ax.set_ylabel("Games")
ax.legend()
ax.set_title("Attendance distribution by season")
sns.despine()
st.pyplot(fig)
plt.close()

st.markdown("### Average Attendance by Team (all seasons)")
team_avg = df.groupby('team')['attendance'].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(14, 4))
ax.bar(team_avg.index, team_avg.values, color='steelblue', edgecolor='white')
ax.set_ylabel("Avg Attendance")
ax.set_title("Average home attendance by team")
ax.tick_params(axis='x', rotation=45)
sns.despine()
st.pyplot(fig)
plt.close()

st.markdown("### Attendance by Day of Week")
dow_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
df['dow_label'] = df['day_of_week'].map(dow_map)
dow_avg = df.groupby('day_of_week')['attendance'].mean()
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar([dow_map[d] for d in dow_avg.index], dow_avg.values, color='steelblue')
ax.set_ylabel("Avg Attendance")
ax.set_title("Attendance by day of week")
sns.despine()
st.pyplot(fig)
plt.close()

st.markdown("### Attendance by Month")
month_map = {4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct'}
month_avg = df.groupby('month')['attendance'].mean()
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar([month_map.get(m, str(m)) for m in month_avg.index], month_avg.values, color='steelblue')
ax.set_ylabel("Avg Attendance")
ax.set_title("Attendance by month")
sns.despine()
st.pyplot(fig)
plt.close()

st.markdown("### Game Factor Effects (raw averages)")
factors = {
    'Opening Day': df[df['is_opening_day'] == 1]['attendance'].mean(),
    'Regular Game': df[df['is_opening_day'] == 0]['attendance'].mean(),
    'Rival Game': df[df['is_rival'] == 1]['attendance'].mean(),
    'Non-Rival': df[df['is_rival'] == 0]['attendance'].mean(),
    'Promo Night': df[df['is_promo_night'] == 1]['attendance'].mean(),
    'No Promo': df[df['is_promo_night'] == 0]['attendance'].mean(),
    'Weekend': df[df['is_weekend'] == 1]['attendance'].mean(),
    'Weekday': df[df['is_weekend'] == 0]['attendance'].mean(),
    'Playoff Race': df[df['is_playoff_race'] == 1]['attendance'].mean(),
    'Not In Race': df[df['is_playoff_race'] == 0]['attendance'].mean(),
}
fig, ax = plt.subplots(figsize=(12, 4))
colors = ['#2196F3' if i % 2 == 0 else '#90CAF9' for i in range(len(factors))]
ax.bar(factors.keys(), factors.values(), color=colors)
ax.set_ylabel("Avg Attendance")
ax.set_title("Raw attendance by game factor")
ax.tick_params(axis='x', rotation=30)
sns.despine()
st.pyplot(fig)
plt.close()

st.markdown("### Win Percentage vs Attendance")
fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(df['home_win_pct_20g'], df['attendance'], alpha=0.15, s=8, color='steelblue')
m, b = np.polyfit(df['home_win_pct_20g'].fillna(0.5), df['attendance'].fillna(0), 1)
x_line = np.linspace(0.2, 0.8, 100)
ax.plot(x_line, m * x_line + b, color='red', linewidth=1.5, label=f'trend')
ax.set_xlabel("Rolling 20-game win pct")
ax.set_ylabel("Attendance")
ax.set_title("Team performance vs attendance")
ax.legend()
sns.despine()
st.pyplot(fig)
plt.close()