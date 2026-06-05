import numpy as np

def geometric_adstock(x, decay, L=8):
    # carry-over effect: spend from prior weeks still drives attendance
    # decay=0.3 means fast decay (paid social), decay=0.7 means slow (broadcast)
    x = np.array(x, dtype=float)
    out = np.zeros_like(x)
    for t in range(len(x)):
        for l in range(min(L, t + 1)):
            out[t] += x[t - l] * (decay ** l)
    return out

def hill_saturation(x, alpha, gamma):
    # diminishing returns curve - spend beyond saturation point adds little
    # alpha controls steepness, gamma controls inflection point
    x = np.array(x, dtype=float)
    x_norm = x / (np.max(x) + 1e-8)
    return (x_norm ** alpha) / (x_norm ** alpha + gamma ** alpha)

def apply_adstock_saturation(spend_series, decay, alpha, gamma, L=8):
    adstocked = geometric_adstock(spend_series, decay, L)
    saturated = hill_saturation(adstocked, alpha, gamma)
    return saturated

# default priors for each channel based on MMM literature for entertainment/sports
CHANNEL_PRIORS = {
    'paid_social': {
        'decay_mu': 0.3, 'decay_sigma': 0.1,
        'alpha_mu': 2.0, 'alpha_sigma': 0.5,
        'gamma_mu': 0.3, 'gamma_sigma': 0.1,
        'beta_mu': 0.15, 'beta_sigma': 0.05,
    },
    'email': {
        'decay_mu': 0.2, 'decay_sigma': 0.08,
        'alpha_mu': 1.5, 'alpha_sigma': 0.4,
        'gamma_mu': 0.2, 'gamma_sigma': 0.08,
        'beta_mu': 0.20, 'beta_sigma': 0.06,
    },
    'broadcast': {
        'decay_mu': 0.7, 'decay_sigma': 0.1,
        'alpha_mu': 1.8, 'alpha_sigma': 0.5,
        'gamma_mu': 0.5, 'gamma_sigma': 0.12,
        'beta_mu': 0.10, 'beta_sigma': 0.04,
    },
    'ooh': {
        'decay_mu': 0.5, 'decay_sigma': 0.1,
        'alpha_mu': 1.6, 'alpha_sigma': 0.4,
        'gamma_mu': 0.4, 'gamma_sigma': 0.1,
        'beta_mu': 0.08, 'beta_sigma': 0.03,
    },
}

if __name__ == '__main__':
    # quick sanity check
    np.random.seed(42)
    test_spend = np.random.exponential(1000, 81)

    adstocked = geometric_adstock(test_spend, decay=0.5)
    saturated = hill_saturation(adstocked, alpha=2.0, gamma=0.3)

    print(f"raw spend mean:      {test_spend.mean():.2f}")
    print(f"adstocked mean:      {adstocked.mean():.2f}")
    print(f"saturated mean:      {saturated.mean():.4f}")
    print(f"saturation range:    {saturated.min():.4f} - {saturated.max():.4f}")
    print("\nchannel priors loaded:")
    for ch, p in CHANNEL_PRIORS.items():
        print(f"  {ch}: decay~N({p['decay_mu']}, {p['decay_sigma']}), beta~N({p['beta_mu']}, {p['beta_sigma']})")
