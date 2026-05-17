import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def bs_price(S, K, T, r, sigma, opt_type='C'):
    """Black-Scholes option price."""
    if T <= 0 or sigma <= 0:
        return np.nan
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt_type == 'C':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_vol(mkt_price, S, K, T, r, opt_type='C'):
    """
    Implied volatility via Brent's method.
    Returns NaN when no-arbitrage bounds are violated or root not found.
    """
    if T <= 0 or mkt_price <= 0:
        return np.nan
    intrinsic = max(S - K, 0) if opt_type == 'C' else max(K - S, 0)
    if mkt_price < intrinsic * 0.999:
        return np.nan
    if opt_type == 'C' and mkt_price >= S:
        return np.nan
    try:
        return brentq(
            lambda s: bs_price(S, K, T, r, s, opt_type) - mkt_price,
            1e-6, 10.0, xtol=1e-6, maxiter=200
        )
    except (ValueError, RuntimeError):
        return np.nan
    

def compute_iv_surface(df, risk_free_rate):
    """Vectorised IV computation with progress reporting."""
    # Pre-filter to a sensible moneyness range (speeds up & improves quality)
    df = df[(df['log_m'] > -0.6) & (df['log_m'] < 0.4)].copy()
    print(f"Computing IV for {len(df):,} options...")

    ivs = np.full(len(df), np.nan)
    for i, (_, row) in enumerate(df.iterrows()):
        ivs[i] = implied_vol(row['mid'], row['spot'], row['strike'],
                             row['T'], risk_free_rate, row['type'])
        if (i + 1) % 10_000 == 0:
            pct = (i + 1) / len(df) * 100
            valid = np.sum(~np.isnan(ivs[:i+1]))
            print(f"  {i+1:>6,} / {len(df):,}  ({pct:.0f}%)  valid IV: {valid:,}")

    df['iv'] = ivs
    df = df[(df['iv'] > 0.01) & (df['iv'] < 2.0)].copy()
    print(f"✓ Valid IVs: {len(df):,}")
    return df