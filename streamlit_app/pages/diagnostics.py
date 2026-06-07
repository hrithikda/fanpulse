import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from backend.models import mmm
from backend.models.predict import (
    posterior_means,
    compute_mu,
    compute_mu_draws,
    regression_metrics,
)

st.set_page_config(page_title="Diagnostics — FanPulse", layout="wide")
st.title("Model Diagnostics")
st.caption(
    "How well do predictions track reality? Predictions are reconstructed from the "
    "posterior on the fitted dataset."
)


@st.cache_resource
def load_trace():
    with open("backend/data/processed/mmm_trace.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_predictions():
    df = mmm.load_data()
    df, _ = mmm.preprocess(df)
    trace = load_trace()
    params = posterior_means(trace)
    df = df.reset_index(drop=True)
    df["pred"] = np.exp(compute_mu(df, params))
    df["actual"] = df["attendance"].values
    df["resid"] = df["actual"] - df["pred"]
    df["pct_err"] = np.abs(df["resid"]) / df["actual"] * 100.0
    return df


try:
    trace = load_trace()
except FileNotFoundError:
    st.error("Trace file not found. Run: python -m backend.models.mmm")
    st.stop()

df = load_predictions()
overall = regression_metrics(df["actual"].values, df["pred"].values)

st.markdown("### Overall fit (in-sample, full dataset)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("R²", f"{overall['r2']:.3f}")
c2.metric("MAPE", f"{overall['mape']:.1f}%")
c3.metric("RMSE", f"{overall['rmse']:,.0f}")
c4.metric("Games", f"{len(df):,}")
st.caption(
    "For genuine out-of-sample numbers see the home page (time-based 2024 holdout). "
    "These in-sample figures show how the posterior-mean model tracks the data it was fit on."
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Predicted vs. Actual")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["actual"], df["pred"], alpha=0.1, s=8, color="steelblue")
    lim = [df["actual"].min(), df["actual"].max()]
    ax.plot(lim, lim, color="red", linewidth=1.2, linestyle="--", label="perfect")
    ax.set_xlabel("Actual attendance")
    ax.set_ylabel("Predicted attendance")
    ax.set_title("Predicted vs. actual")
    ax.legend()
    sns.despine()
    st.pyplot(fig)
    plt.close()

with col2:
    st.markdown("#### Residuals vs. Fitted")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["pred"], df["resid"], alpha=0.1, s=8, color="steelblue")
    ax.axhline(0, color="red", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Predicted attendance")
    ax.set_ylabel("Residual (actual − predicted)")
    ax.set_title("Residuals vs. fitted")
    sns.despine()
    st.pyplot(fig)
    plt.close()

st.markdown("#### Residual Distribution")
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.hist(df["resid"], bins=60, color="steelblue", alpha=0.8, edgecolor="white")
ax.axvline(0, color="red", linewidth=1.2, linestyle="--")
ax.axvline(df["resid"].mean(), color="orange", linewidth=1.2,
           label=f"mean = {df['resid'].mean():,.0f}")
ax.set_xlabel("Residual (fans)")
ax.set_ylabel("Games")
ax.set_title("Residual distribution")
ax.legend()
sns.despine()
st.pyplot(fig)
plt.close()

st.markdown("### Posterior Predictive Check")
st.caption(
    "Replicated attendance datasets drawn from the posterior (parameter + observation "
    "uncertainty) overlaid on the observed distribution. Good fit = simulated curves "
    "envelope the observed curve."
)
mu_draws, sigma_draws = compute_mu_draws(df, trace, n_draws=40)
rng = np.random.default_rng(1)
y_rep = np.exp(mu_draws + rng.normal(0, 1, mu_draws.shape) * sigma_draws[:, None])

fig, ax = plt.subplots(figsize=(10, 4))
for k in range(min(30, y_rep.shape[0])):
    ax.hist(y_rep[k], bins=60, range=(0, df["actual"].max()), histtype="step",
            color="steelblue", alpha=0.15, density=True, linewidth=0.7)
ax.hist(df["actual"], bins=60, range=(0, df["actual"].max()), histtype="step",
        color="red", linewidth=2.0, density=True, label="observed")
ax.plot([], [], color="steelblue", alpha=0.6, label="posterior replicates")
ax.set_xlabel("Attendance")
ax.set_ylabel("Density")
ax.set_title("Posterior predictive vs. observed attendance")
ax.legend()
sns.despine()
st.pyplot(fig)
plt.close()

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Error by Team")
    team_err = (
        df.groupby("team")
        .agg(**{
            "MAPE (%)": ("pct_err", "mean"),
            "Mean Resid": ("resid", "mean"),
            "Games": ("team", "size"),
        })
        .reset_index()
        .rename(columns={"team": "Team"})
        .sort_values("MAPE (%)", ascending=False)
    )
    team_err["MAPE (%)"] = team_err["MAPE (%)"].round(1)
    team_err["Mean Resid"] = team_err["Mean Resid"].round(0).astype(int)
    st.dataframe(team_err, use_container_width=True, hide_index=True, height=420)

with col2:
    st.markdown("### Error by Month")
    month_map = {4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct"}
    month_err = (
        df.groupby("month")
        .agg(**{
            "MAPE (%)": ("pct_err", "mean"),
            "Mean Resid": ("resid", "mean"),
            "Games": ("month", "size"),
        })
        .reset_index()
    )
    month_err["Month"] = month_err["month"].map(month_map)
    month_err["MAPE (%)"] = month_err["MAPE (%)"].round(1)
    month_err["Mean Resid"] = month_err["Mean Resid"].round(0).astype(int)
    month_err = month_err[["Month", "MAPE (%)", "Mean Resid", "Games"]]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(month_err["Month"], month_err["MAPE (%)"], color="steelblue")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Error by month")
    sns.despine()
    st.pyplot(fig)
    plt.close()
    st.dataframe(month_err, use_container_width=True, hide_index=True)
