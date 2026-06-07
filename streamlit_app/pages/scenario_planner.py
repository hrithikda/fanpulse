import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy.optimize import minimize

st.set_page_config(page_title="Scenario Planner — FanPulse", layout="wide")
st.title("Budget Scenario Planner")
st.caption("Reallocate marketing spend across channels and see projected attendance impact.")

st.warning(
    "Spend levels, channel ROI, and the budget figures below are **synthetic** proxies "
    "for demonstrating MMM-driven budget planning — not real MLB marketing data. "
    "See the Methodology page."
)

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

# saturation scale (as a fraction of total budget) controlling diminishing
# returns: small scale = saturates quickly (a little spend goes a long way but
# extra dollars do little), large scale = lots of headroom before saturating.
CHANNEL_SAT_SCALE = {
    'Paid Social': 0.35,
    'Email': 0.12,
    'Broadcast': 0.45,
    'Out-of-Home': 0.30,
}

# slider bounds (%) — optimizer respects the same ranges so its result can
# pre-fill the sliders cleanly
CHANNEL_BOUNDS = {
    'Paid Social': (0, 80),
    'Email': (0, 30),
    'Broadcast': (0, 80),
    'Out-of-Home': (0, 60),
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


CHANNELS_ORDER = ['Paid Social', 'Email', 'Broadcast', 'Out-of-Home']


def saturated_response(weights):
    """Total attendance response under diminishing returns.

    Unlike ``compute_attendance_lift`` (which is linear in spend and therefore
    always favors a single channel), this applies an exponential saturation per
    channel so that the optimizer finds a balanced interior allocation.
    """
    total = 0.0
    for ch, w in zip(CHANNELS_ORDER, weights):
        efficiency = CHANNEL_BETA[ch] * CHANNEL_ROI[ch]
        k = CHANNEL_SAT_SCALE[ch]
        total += efficiency * (1.0 - np.exp(-w / k))
    return total


def optimize_allocation():
    """Maximize saturated response s.t. weights sum to 1 and per-channel bounds."""
    bounds = [(lo / 100.0, hi / 100.0) for lo, hi in
              (CHANNEL_BOUNDS[ch] for ch in CHANNELS_ORDER)]
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    x0 = np.array([BASELINE_SPLIT[ch] for ch in CHANNELS_ORDER])

    result = minimize(
        lambda w: -saturated_response(w),
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-9, 'maxiter': 500},
    )
    w = np.clip(result.x, 0, None)
    w = w / w.sum()
    return {ch: float(w[i]) for i, ch in enumerate(CHANNELS_ORDER)}


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

st.markdown("### Optimize Allocation")
st.caption(
    "Solve for the budget split that maximizes projected attendance under "
    "diminishing returns (Hill-style saturation per channel), subject to the "
    "budget summing to 100% and per-channel caps."
)

SLIDER_DEFAULTS = {'sl_social': 25, 'sl_email': 5, 'sl_broadcast': 45, 'sl_ooh': 25}
for key, val in SLIDER_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

opt_col1, opt_col2 = st.columns([1, 3])
with opt_col1:
    run_opt = st.button("Find optimal allocation", type="primary")
if run_opt:
    optimal = optimize_allocation()
    st.session_state['sl_social'] = int(round(optimal['Paid Social'] * 100))
    st.session_state['sl_email'] = int(round(optimal['Email'] * 100))
    st.session_state['sl_broadcast'] = int(round(optimal['Broadcast'] * 100))
    st.session_state['sl_ooh'] = int(round(optimal['Out-of-Home'] * 100))
    # nudge to ensure the rounded integers still sum to 100
    drift = 100 - sum(st.session_state[k] for k in SLIDER_DEFAULTS)
    st.session_state['sl_broadcast'] = max(0, st.session_state['sl_broadcast'] + drift)
    st.session_state['_optimized'] = True
    st.rerun()

if st.session_state.get('_optimized'):
    st.success(
        "Sliders below are set to the optimizer's recommended allocation. "
        "Adjust them to explore around the optimum."
    )

st.markdown("### Adjust Budget Allocation")
st.caption("Sliders must sum to 100%. Adjust and click Calculate.")

col1, col2, col3, col4 = st.columns(4)
with col1:
    s_social = st.slider("Paid Social (%)", 0, 80, step=1, key='sl_social')
with col2:
    s_email = st.slider("Email (%)", 0, 30, step=1, key='sl_email')
with col3:
    s_broadcast = st.slider("Broadcast (%)", 0, 80, step=1, key='sl_broadcast')
with col4:
    s_ooh = st.slider("Out-of-Home (%)", 0, 60, step=1, key='sl_ooh')

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