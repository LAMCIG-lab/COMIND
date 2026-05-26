import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

def _z(x):
    x = np.asarray(x, float)
    m = np.nanmean(x); s = np.nanstd(x)
    return (x - m) / (s if np.isfinite(s) and s > 0 else 1.0)

def build_severity_index(df):
    moca = _z(df["MCATOT"].to_numpy())        # higher is better
    td   = _z(df["TD_score"].to_numpy())
    pigd = _z(df["PIGD_score"].to_numpy())
    S = (-moca + td + pigd) / 3.0             # higher = worse / later
    return S

def fit_mixedlm_beta_from_clinical(df, ids, dt, t_max, verbose=False, rng=None):

    if rng is None:
        rng = np.random.default_rng(0)

    S = build_severity_index(df)
    dfm = pd.DataFrame({"id": ids, "dt": dt.astype(float), "S": S})
    # keep longitudinal subjects
    good_ids = dfm.groupby("id").size().pipe(lambda s: s[s >= 2]).index
    dfm = dfm[dfm["id"].isin(good_ids)].copy()

    # clean + tiny jitter on dt to avoid exact duplicate rows
    dfm = dfm.replace([np.inf, -np.inf], np.nan).dropna(subset=["id", "dt", "S"])
    dfm["dt"] = dfm["dt"] + 1e-6 * rng.standard_normal(len(dfm))

    # fit: severity ~ dt + (1|id)
    model = smf.mixedlm("S ~ dt", data=dfm, groups=dfm["id"], re_formula="1")
    result = None
    for meth in ("bfgs", "nm"):
        result = model.fit(method=meth, reml=True, maxiter=500, disp=False)
        break

    # get slope k (fixed effect of dt)
    if result is not None and result.converged:
        k = float(result.params.get("dt", np.nan))
    else:
        # fallback OLS for slope if MixedLM fails
        ols = smf.ols("S ~ dt", data=dfm).fit()
        k = float(ols.params.get("dt", np.nan))

    if not np.isfinite(k) or abs(k) < 1e-8:
        # if slope is (near) zero, default to small positive slope to avoid blow-up
        if verbose:
            print("[MixedLM] slope near zero; using fallback k=1.0")
        k = 1.0

    # random intercepts u_i
    re = {}
    if result is not None and hasattr(result, "random_effects"):
        re = {pid: float(eff.get("Group", 0.0)) for pid, eff in result.random_effects.items()}

    unique_ids = np.unique(ids)
    beta_raw = np.array([re.get(pid, 0.0) / k for pid in unique_ids], dtype=float)

    # shift & clip: make the smallest beta zero, cap at t_max
    beta_shift = beta_raw - np.nanmin(beta_raw)
    initial_beta = np.clip(beta_shift, 0.0, t_max)

    pid_to_beta = {pid: initial_beta[i] for i, pid in enumerate(unique_ids)}

    if verbose:
        print("beta_init summary:", pd.Series(initial_beta).describe())

    return initial_beta, pid_to_beta, result

