import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

st.set_page_config(page_title="Scenario Planner — FanPulse", layout="wide")
st.title("Budget Scenario Planner")
st.caption("Reallocate marketing spend across channels and see projected attendance impact.")

AVG_ATTENDANCE = 26572
TICKET_PRICE = 35
GAMES_PER_SEASON = 81
BASELINE_BUDGET = 2_100_000

BASELINE_SPLIT = {
    'Paid Social': 0.25,
    'Email': 0.05,
    'Broadcast': 0.45,
    'Out-of-Home': 0.25,
}

CHANNEL_ROI = {
    'Paid Social': 4.20,
    'Email': 7.80,
    'Broadcast': 2.10,
    'Out-of-Home': 3.40,
}

CHANNEL_BETA = {
    'Paid Social': 0.035,
    'Email': 0.028,
    'Broadcast': 0.009,
    'Out-of-Home': 0.004,
}

CHANNEL_DECAY = {
    'Paid Social': 0.3,
    'Email': 0.2,
    'Broadcast': 0.7,
    'Out-of-Home': 0.5,
}

@st.cache_resource
def load_trace():
    try:
        with open('backend/data/processed/mmm_trace.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

trace = load_trace()

def compute_attendance_lift(new_split, baseline_split=BASELINE_SPLIT):
    lift = 0.0
    for ch in new_split:
        new_spend = new_split[ch] * BASELINE_BUDGET
        base_spend = baseline_split[ch] * BASELINE_BUDGET
        delta_spend = new_spend - base_spend
        # marginal ROI * beta * spend delta as fraction of baseline attendance
        beta = CHANNEL_BETA[ch]
        roi = CHANNEL_ROI[ch]
        lift += beta * roi * (delta_spend / BASELINE_BUDGET)
    return lift

def compute_marginal_roi(split):
    rows = []
    for ch in split:
        spend = split[ch] * BASELINE_BUDGET
        beta = CHANNEL_BETA[ch]
        roi = CHANNEL_ROI[ch]
        marginal = beta * roi * 1000 / BASELINE_BUDGET * AVG_ATTENDANCE
        rows.append({
            'Channel': ch,
            'Spend ($)': int(spend),
            'Spend (%)': f"{split[ch]*100:.1f}%",
            'ROI ($/$ spent)': f"${roi:.2f}",
            'Marginal fans per $1K': round(marginal, 1),
        })
    return pd.DataFrame(rows)

st.markdown("### Baseline Budget Allocation")
st.caption(f"Total seasonal marketing budget: ${BASELINE_BUDGET:,}")

col1, col2 = st.columns([1, 1])

with col1:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        BASELINE_SPLIT.values(),
        labels=BASELINE_SPLIT.keys(),
        autopct='%1.0f%%',
        colors=['#2196F3', '#4CAF50', '#FF9800', '#9C27B0'],
        startangle=90
    )
    ax.set_title("Baseline spend allocation")
    st.pyplot(fig)
    plt.close()

with col2:
    st.markdown("**Baseline channel metrics:**")
    baseline_df = compute_marginal_roi(BASELINE_SPLIT)
    st.dataframe(baseline_df, use_container_width=True, hide_index=True)

st.markdown("### Adjust Budget Allocation")
st.caption("Sliders must sum to 100%. Adjust and click Calculate.")

col1, col2, col3, col4 = st.columns(4)
with col1:
    s_social = st.slider("Paid Social (%)", 0, 80, 25, step=1)
with col2:
    s_email = st.slider("Email (%)", 0, 30, 5, step=1)
with col3:
    s_broadcast = st.slider("Broadcast (%)", 0, 80, 45, step=1)
with col4:
    s_ooh = st.slider("Out-of-Home (%)", 0, 60, 25, step=1)

total = s_social + s_email + s_broadcast + s_ooh

if total != 100:
    st.warning(f"Sliders sum to {total}%. Adjust to exactly 100% to run scenario.")
else:
    new_split = {
        'Paid Social': s_social / 100,
        'Email': s_email / 100,
        'Broadcast': s_broadcast / 100,
        'Out-of-Home': s_ooh / 100,
    }

    lift_pct = compute_attendance_lift(new_split)
    new_attendance = AVG_ATTENDANCE * (1 + lift_pct)
    fan_delta = int(new_attendance - AVG_ATTENDANCE)
    gate_delta = int(fan_delta * TICKET_PRICE * GAMES_PER_SEASON)

    st.markdown("### Scenario Results")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Attendance Lift", f"{lift_pct*100:+.1f}%")
    col2.metric("Projected Avg Attendance", f"{int(new_attendance):,}", delta=f"{fan_delta:+,} fans")
    col3.metric("Incremental Fans/Game", f"{fan_delta:+,}")
    col4.metric("Projected Gate Revenue Lift", f"${gate_delta:+,.0f}")

    col1, col2 = st.columns([1, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(
            new_split.values(),
            labels=new_split.keys(),
            autopct='%1.0f%%',
            colors=['#2196F3', '#4CAF50', '#FF9800', '#9C27B0'],
            startangle=90
        )
        ax.set_title("New spend allocation")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**New channel metrics:**")
        new_df = compute_marginal_roi(new_split)
        st.dataframe(new_df, use_container_width=True, hide_index=True)

    st.markdown("### Baseline vs Scenario Comparison")
    channels = list(BASELINE_SPLIT.keys())
    baseline_spends = [BASELINE_SPLIT[c] * BASELINE_BUDGET for c in channels]
    new_spends = [new_split[c] * BASELINE_BUDGET for c in channels]

    x = np.arange(len(channels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width/2, baseline_spends, width, label='Baseline', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, new_spends, width, label='Scenario', color='#FF9800', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(channels)
    ax.set_ylabel("Spend ($)")
    ax.set_title("Spend allocation: baseline vs scenario")
    ax.legend()
    sns.despine()
    st.pyplot(fig)
    plt.close()

    st.markdown("### Marginal ROI Curve")
    st.caption("Shows how additional spend in each channel converts to attendance fans at different spend levels.")
    spend_range = np.linspace(0, BASELINE_BUDGET * 0.6, 100)
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    for (ch, beta), color in zip(CHANNEL_BETA.items(), colors):
        roi = CHANNEL_ROI[ch]
        fans = beta * roi * spend_range / BASELINE_BUDGET * AVG_ATTENDANCE
        ax.plot(spend_range / 1000, fans, label=ch, color=color, linewidth=2)
    ax.set_xlabel("Channel Spend ($K)")
    ax.set_ylabel("Incremental Fans")
    ax.set_title("Marginal fan impact by channel spend level")
    ax.legend()
    sns.despine()
    st.pyplot(fig)
    plt.close()

    if trace is not None:
        st.markdown("### Posterior Uncertainty on Scenario")
        st.caption("Distribution of projected attendance lift across 2,000 posterior samples.")
        lift_samples = []
        for ch, param in zip(
            ['Paid Social', 'Email', 'Broadcast', 'Out-of-Home'],
            ['beta_paid_social', 'beta_email', 'beta_broadcast', 'beta_ooh']
        ):
            samples = trace.posterior[param].values.flatten()
            delta = (new_split[ch] - BASELINE_SPLIT[ch]) * BASELINE_BUDGET
            lift_samples.append(samples * CHANNEL_ROI[ch] * delta / BASELINE_BUDGET)

        total_lift_samples = np.sum(lift_samples, axis=0)
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(total_lift_samples * 100, bins=50, color='steelblue', alpha=0.8, edgecolor='white')
        ax.axvline(np.mean(total_lift_samples) * 100, color='red', linewidth=1.5,
                   label=f"mean: {np.mean(total_lift_samples)*100:.2f}%")
        ax.axvline(np.percentile(total_lift_samples, 5.5) * 100, color='orange',
                   linewidth=1, linestyle='--', label='89% CI')
        ax.axvline(np.percentile(total_lift_samples, 94.5) * 100, color='orange',
                   linewidth=1, linestyle='--')
        ax.set_xlabel("Projected Attendance Lift (%)")
        ax.set_ylabel("Posterior samples")
        ax.set_title("Uncertainty in projected attendance lift")
        ax.legend()
        sns.despine()
        st.pyplot(fig)
        plt.close()