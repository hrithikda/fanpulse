import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import arviz as az

st.set_page_config(page_title="Model Results — FanPulse", layout="wide")
st.title("Model Results")

@st.cache_resource
def load_trace():
    with open('backend/data/processed/mmm_trace.pkl', 'rb') as f:
        return pickle.load(f)

try:
    trace = load_trace()
except FileNotFoundError:
    st.error("Trace file not found. Run: python -m backend.models.mmm")
    st.stop()

st.markdown("### Game Factor Coefficients (posterior mean + 89% credible interval)")
st.caption("All effects are on log-attendance scale. Multiply by avg attendance (~26,572) to get approximate fan impact.")

params = {
    'Opening Day': 'b_opening',
    'Playoff Race (Sep)': 'b_playoff',
    'Win Pct (rolling 20g)': 'b_winpct',
    'Market Size': 'b_market',
    'Rival Game': 'b_rival',
    'Promo Night': 'b_promo',
    'Weekend': 'b_weekend',
    'Fireworks': 'b_fireworks',
    'July 4th Week': 'b_july4',
    'Run Differential': 'b_rundiff',
    'Win Streak': 'b_streak',
}

means, lo, hi, labels = [], [], [], []
for label, param in params.items():
    samples = trace.posterior[param].values.flatten()
    means.append(samples.mean())
    lo.append(np.percentile(samples, 5.5))
    hi.append(np.percentile(samples, 94.5))
    labels.append(label)

fig, ax = plt.subplots(figsize=(10, 6))
y_pos = range(len(labels))
ax.barh(y_pos, means, xerr=[
    [m - l for m, l in zip(means, lo)],
    [h - m for m, h in zip(means, hi)]
], color=['#2196F3' if m > 0 else '#EF5350' for m in means],
    alpha=0.8, capsize=4)
ax.set_yticks(list(y_pos))
ax.set_yticklabels(labels)
ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_xlabel("Coefficient (log-attendance scale)")
ax.set_title("Posterior means with 89% credible intervals")
sns.despine()
st.pyplot(fig)
plt.close()

st.markdown("### Attendance Impact (fans per game)")
st.caption("Approximate fan count impact at mean attendance of 26,572")
impact_data = []
for label, param in params.items():
    samples = trace.posterior[param].values.flatten()
    mean_beta = samples.mean()
    fan_impact = int(np.exp(mean_beta) * 26572 - 26572)
    impact_data.append({'Factor': label, 'Beta': round(mean_beta, 3), 'Fan Impact': fan_impact})

impact_df = pd.DataFrame(impact_data).sort_values('Fan Impact', ascending=False)
st.dataframe(impact_df, use_container_width=True, hide_index=True)

st.markdown("### Marketing Channel Posteriors")
channels = ['paid_social', 'email', 'broadcast', 'ooh']
channel_labels = ['Paid Social', 'Email', 'Broadcast', 'Out-of-Home']
roi_map = {'paid_social': 4.20, 'email': 7.80, 'broadcast': 2.10, 'ooh': 3.40}

fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for i, (ch, label) in enumerate(zip(channels, channel_labels)):
    samples = trace.posterior[f'beta_{ch}'].values.flatten()
    axes[i].hist(samples, bins=40, color='steelblue', alpha=0.8, edgecolor='white')
    axes[i].axvline(samples.mean(), color='red', linewidth=1.5, label=f'mean={samples.mean():.3f}')
    axes[i].set_title(label)
    axes[i].set_xlabel("Beta")
    axes[i].legend(fontsize=8)
    sns.despine(ax=axes[i])

plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("### Channel ROI Summary")
roi_data = []
for ch, label in zip(channels, channel_labels):
    samples = trace.posterior[f'beta_{ch}'].values.flatten()
    roi_data.append({
        'Channel': label,
        'Beta (mean)': round(samples.mean(), 4),
        'Beta (std)': round(samples.std(), 4),
        'Est. ROI': f"${roi_map[ch]:.2f}",
        'Credible Interval': f"[{np.percentile(samples, 5.5):.4f}, {np.percentile(samples, 94.5):.4f}]"
    })

st.dataframe(pd.DataFrame(roi_data), use_container_width=True, hide_index=True)

st.markdown("### R-hat Convergence Diagnostics")
summary = az.summary(trace, var_names=list(params.values()) + [f'beta_{ch}' for ch in channels])
st.dataframe(summary[['mean', 'sd', 'r_hat', 'ess_bulk']].round(3), use_container_width=True)