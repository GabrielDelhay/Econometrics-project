import numpy as np
import pandas as pd

def _slice_metrics(t_df, wing_lo, wing_hi_p, wing_lo_c, wing_hi):
    """ATM IV, put-wing IV, call-wing IV for a single expiry slice."""
    if len(t_df) < 3:
        return np.nan, np.nan, np.nan
    atm_iv = t_df.loc[t_df['log_m'].abs().idxmin(), 'iv']
    p_wing = t_df[(t_df['log_m'] > wing_lo)  & (t_df['log_m'] < wing_hi_p) & (t_df['type']=='P')]
    c_wing = t_df[(t_df['log_m'] > wing_lo_c) & (t_df['log_m'] < wing_hi)  & (t_df['type']=='C')]
    iv_p   = p_wing['iv'].mean() if len(p_wing) > 0 else np.nan
    iv_c   = c_wing['iv'].mean() if len(c_wing) > 0 else np.nan
    return atm_iv, iv_p, iv_c


def _interp_var(iv1, iv2, T1, T2, tau):
    """Linear interpolation in variance space between tenors T1 < tau <= T2."""
    if T1 == T2:
        return iv1
    w1  = (T2 - tau) / (T2 - T1)
    w2  = (tau - T1) / (T2 - T1)
    var = w1 * iv1**2 + w2 * iv2**2
    return np.sqrt(max(var, 0.0))


def extract_daily_metrics(df, target_tenor,
                          wing_lo, wing_hi_p,
                          wing_lo_c, wing_hi):
    """
    Extract ATM IV, skew, convexity per trading day using constant-maturity
    interpolation: variance-weighted blend of the two expiries bracketing
    target_tenor days (same methodology as the CBOE VIX).
    """
    rows = []
    for date, day in df.groupby('date'):
        exps   = np.array(sorted(day['exp_date'].unique()))
        tenors = np.array([(pd.Timestamp(e) - date).days for e in exps])

        below = tenors[tenors <= target_tenor]
        above = tenors[tenors >  target_tenor]

        if len(below) == 0 or len(above) == 0:
            # Edge case: target is outside available range → use closest expiry
            best_exp = exps[np.argmin(np.abs(tenors - target_tenor))]
            t_df     = day[day['exp_date'] == best_exp].sort_values('log_m')
            atm_iv, iv_p, iv_c = _slice_metrics(t_df, wing_lo, wing_hi_p, wing_lo_c, wing_hi)
        else:
            T1   = int(below[-1])   # largest tenor <= target
            T2   = int(above[0])    # smallest tenor > target
            exp1 = exps[tenors == T1][0]
            exp2 = exps[tenors == T2][0]

            t_df1 = day[day['exp_date'] == exp1].sort_values('log_m')
            t_df2 = day[day['exp_date'] == exp2].sort_values('log_m')

            atm1, ivp1, ivc1 = _slice_metrics(t_df1, wing_lo, wing_hi_p, wing_lo_c, wing_hi)
            atm2, ivp2, ivc2 = _slice_metrics(t_df2, wing_lo, wing_hi_p, wing_lo_c, wing_hi)

            if np.isnan(atm1) or np.isnan(atm2):
                atm_iv = atm1 if not np.isnan(atm1) else atm2
                iv_p   = ivp1 if not np.isnan(ivp1) else ivp2
                iv_c   = ivc1 if not np.isnan(ivc1) else ivc2
            else:
                atm_iv = _interp_var(atm1, atm2, T1, T2, target_tenor)
                iv_p   = _interp_var(ivp1, ivp2, T1, T2, target_tenor) \
                         if not (np.isnan(ivp1) or np.isnan(ivp2)) else np.nan
                iv_c   = _interp_var(ivc1, ivc2, T1, T2, target_tenor) \
                         if not (np.isnan(ivc1) or np.isnan(ivc2)) else np.nan

        rows.append({
            'date'        : date,
            'atm_iv'      : atm_iv,
            'skew'        : iv_p - iv_c if not (np.isnan(iv_p) or np.isnan(iv_c)) else np.nan,
            'convexity'   : (iv_p + iv_c) / 2 - atm_iv
                            if not (np.isnan(iv_p) or np.isnan(iv_c)) else np.nan,
            'iv_put_wing' : iv_p,
            'iv_call_wing': iv_c,
        })

    metrics = pd.DataFrame(rows).set_index('date').sort_index()
    print(f"✓ Daily metrics: {len(metrics)} trading days  "
          f"(constant-maturity {target_tenor}d interpolation)")
    return metrics