import numpy as np
import pandas as pd
import glob
import os

def load_and_clean(data_folder):
    """Load all daily CSVs, clean, and estimate spot via implicit discount factor
    (methodology: Brenner & Galai 1986, extended — regression of C-P on K)."""
    files = sorted(glob.glob(os.path.join(data_folder, '*.csv')))
    if not files:
        raise FileNotFoundError(f"No CSV files in {data_folder!r}")
    print(f"Loading {len(files)} files...")

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"  ⚠ {f}: {e}")

    df = pd.concat(dfs, ignore_index=True)
    df.columns = ['date','date2','ask','bid','strike','oi',
                  'strike2','high','low','open','instrument','exp_date','type']

    df['date']     = pd.to_datetime(df['date'])
    df['exp_date'] = pd.to_datetime(df['exp_date'])
    df['T']        = (df['exp_date'] - df['date']).dt.days / 365.0
    df['mid']      = (df['ask'] + df['bid']) / 2.0

    # Drop expiry noise & illiquid options
    df = df[df['T'] > 7/365].copy()
    df = df[df['mid'] > 0].copy()
    df['ba_rel'] = (df['ask'] - df['bid']) / df['mid']
    df = df[df['ba_rel'] < 0.5].copy()

    # ── Implicit discount factor via put-call parity regression ────────────────
    # C - P = B·F - B·K  →  slope = -B,  intercept = B·F
    # B = e^{-rT} (discount factor),  F = forward price
    calls = df[df['type']=='C'][['date','exp_date','strike','T','mid']].rename(columns={'mid':'C'})
    puts  = df[df['type']=='P'][['date','exp_date','strike','T','mid']].rename(columns={'mid':'P'})
    pairs = pd.merge(calls, puts, on=['date','exp_date','strike','T'])
    pairs['G'] = pairs['C'] - pairs['P']

    impl_rows = []
    for (date, exp), grp in pairs.groupby(['date', 'exp_date']):
        if len(grp) < 3:
            continue
        T   = grp['T'].iloc[0]
        coef = np.polyfit(grp['strike'], grp['G'], deg=1)
        B    = -coef[0]                          # implicit discount factor
        if B <= 0 or B > 1:                      # no-arbitrage filter
            continue
        F      = coef[1] / B                     # implicit forward price
        S_est  = F * B                           # spot = F · e^{-rT}
        r_impl = -np.log(B) / T if T > 0 else np.nan   # implicit rate
        impl_rows.append({
            'date'   : date,
            'exp_date': exp,
            'S_est'  : S_est,
            'F_impl' : F,
            'B_impl' : B,
            'r_impl' : r_impl,
        })

    impl_df = pd.DataFrame(impl_rows)

    # Daily spot: median across expiries
    spot_per_date = impl_df.groupby('date')['S_est'].median().rename('spot')

    # Merge spot and per-(date, expiry) implied quantities back into df
    df = df.merge(spot_per_date, on='date')
    df = df.merge(impl_df[['date','exp_date','F_impl','B_impl','r_impl']],
                  on=['date','exp_date'], how='left')

    # Forward log-moneyness: log(K/F) — symmetric around ATM under most models
    df['log_m'] = np.log(df['strike'] / df['F_impl'])

    # Use implicit r for discounting
    df['r'] = df['r_impl']

    print(f"✓ Rows after cleaning : {len(df):,}")
    print(f"  Trading days        : {df['date'].nunique()}")
    print(f"  Unique expiries     : {df['exp_date'].nunique()}")
    print(f"  Spot range          : {df['spot'].min():.0f} – {df['spot'].max():.0f}")
    print(f"  Implied r (mean)    : {impl_df['r_impl'].mean()*100:.2f}%  ")
    return df