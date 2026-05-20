import numpy as np
import pandas as pd
import statsmodels.api as sm


def within_estimator(panel, y_col, x_cols, entity_col):
    p = panel[[entity_col, y_col] + x_cols].dropna().copy()
 
    # ── Within (entity-demeaning) transformation ──────────────────────────────
    group_means = p.groupby(entity_col)[[y_col] + x_cols].transform('mean')
    p_dm = p[[y_col] + x_cols] - group_means          # demeaned DataFrame
 
    y_dm = p_dm[y_col].values
    X_dm = p_dm[x_cols].values                        # no constant added
 
    # ── OLS on demeaned data — no constant, heteroscedasticity-robust SEs ─────
    result = sm.OLS(y_dm, X_dm).fit(
        cov_type='HC1',          # White's heteroscedasticity-robust SEs
    )
 
    # Attach readable parameter names so .summary() is self-documenting
    result.model.exog_names[:] = x_cols
 
    return result

def build_surface_panel(df, reg, wing_lo, wing_hi_p, wing_lo_c, wing_hi):
    rows = []
    for (date, exp), g in df.groupby(['date', 'exp_date']):
        T_days = (exp - date).days
        if T_days < 7:
            continue
        atm_iv = g[g['log_m'].abs() < 0.03]['iv'].mean()
        p_wing = g[(g['log_m'] > wing_lo)  & (g['log_m'] < wing_hi_p) & (g['type']=='P')]['iv'].mean()
        c_wing = g[(g['log_m'] > wing_lo_c) & (g['log_m'] < wing_hi)  & (g['type']=='C')]['iv'].mean()
        skew   = (p_wing + c_wing)/2 - atm_iv if not (np.isnan(p_wing) or np.isnan(c_wing)) else np.nan
        if np.isnan(atm_iv):
            continue
        rows.append({'date': date, 'expiry': exp,
                     'T_days': T_days, 'atm_iv': atm_iv, 'skew': skew})

    panel = (pd.DataFrame(rows)
               .merge(reg[['rv_forward']].reset_index(), on='date', how='left')
               .dropna(subset=['rv_forward', 'atm_iv']))
    return panel