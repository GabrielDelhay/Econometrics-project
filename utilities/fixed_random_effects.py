import numpy as np
import pandas as pd
from statsmodels.api import OLS as ols

def within_estimator(panel, y_col, x_cols, entity_col):
    # demean then OLS on demeaned data
    p = panel.copy()
    
    # Compute entity means
    for col in [y_col] + x_cols:
        p[col + '_mean'] = p.groupby(entity_col)[col].transform('mean')
    
    # Demean
    p[y_col + '_dm'] = p[y_col] - p[y_col + '_mean']
    for col in x_cols:
        p[col + '_dm']  = p[col]  - p[col  + '_mean']
    
    # OLS on demeaned variables (no constant — absorbed by demeaning)
    y_dm = p[y_col + '_dm'].values
    X_dm = p[[col + '_dm' for col in x_cols]].values
    
    result = ols(y_dm, X_dm, add_const=False)
    return result

def build_surface_panel(df, reg):
    rows = []
    for (date, exp), g in df.groupby(['date', 'exp_date']):
        T_days = (exp - date).days
        if T_days < 7:
            continue
        atm_iv = g[g['log_m'].abs() < 0.03]['iv'].mean()
        p_wing = g[(g['log_m'] > WING_LOW)  & (g['log_m'] < WING_HIGH_PUT) & (g['type']=='P')]['iv'].mean()
        c_wing = g[(g['log_m'] > WING_LOW_CALL) & (g['log_m'] < WING_HIGH)  & (g['type']=='C')]['iv'].mean()
        conv   = (p_wing + c_wing)/2 - atm_iv if not (np.isnan(p_wing) or np.isnan(c_wing)) else np.nan
        if np.isnan(atm_iv):
            continue
        rows.append({'date': date, 'expiry': exp,
                     'T_days': T_days, 'atm_iv': atm_iv, 'convexity': conv})

    panel = (pd.DataFrame(rows)
               .merge(reg[['rv_forward']].reset_index(), on='date', how='left')
               .dropna(subset=['rv_forward', 'atm_iv']))
    return panel