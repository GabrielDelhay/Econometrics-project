from scipy.stats import t as t_dist
import numpy as np

def ols(y, X_df, add_const=True):
    """
    OLS estimator with full output.
    Parameters
    ----------
    y      : array-like, dependent variable
    X_df   : DataFrame or array, regressors (without constant)
    Returns dict with beta, se, t_stat, p_value, r2, adj_r2, resid
    """

    y = np.asarray(y, dtype=float)
    X = np.asarray(X_df, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if add_const:
        X = np.column_stack([np.ones(len(X)), X])

    # Drop NaN rows
    mask = ~(np.isnan(y) | np.any(np.isnan(X), axis=1))
    y, X = y[mask], X[mask]
    n, k = X.shape

    XtX_inv = np.linalg.inv(X.T @ X)
    beta    = XtX_inv @ X.T @ y
    y_hat   = X @ beta
    resid   = y - y_hat
    s2      = (resid @ resid) / (n - k)

    se      = np.sqrt(np.diag(s2 * XtX_inv))
    t_stat  = beta / se
    p_val   = 2 * (1 - t_dist.cdf(np.abs(t_stat), df=n - k))

    ss_res = resid @ resid
    ss_tot = ((y - y.mean()) @ (y - y.mean()))
    r2     = 1 - ss_res / ss_tot
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k)

    return dict(beta=beta, se=se, t_stat=t_stat, p_val=p_val,
                r2=r2, adj_r2=adj_r2, resid=resid, n=n, k=k)


def reg_table(result, names):
    """Pretty-print regression results."""
    print(f"\n{'─'*62}")
    print(f"{'Variable':<22} {'Coef':>9} {'SE':>9} {'t':>8}  {'p':>8}")
    print(f"{'─'*62}")
    for nm, b, se, t, p in zip(names, result['beta'], result['se'],
                                result['t_stat'], result['p_val']):
        stars = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
        print(f"{nm:<22} {b:>9.4f} {se:>9.4f} {t:>8.3f}  {p:>8.4f} {stars}")
    print(f"{'─'*62}")
    print(f"{'R²':<22} {result['r2']:>9.4f}")
    print(f"{'Adj. R²':<22} {result['adj_r2']:>9.4f}")
    print(f"{'N':<22} {result['n']:>9}")
    print(f"{'─'*62}")
    print("Significance: * p<0.05  ** p<0.01  *** p<0.001")