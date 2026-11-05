# 2026_11_04 ADNI zsym, s bounds [2, 4]

Same setup as `2026_11_04_ADNI_zsym_cvic` (zsym, z-score inputs, CVIC, Z = 1-5, 15 random startup trials,
2 restarts, 3 CV folds, step 0.05, inner_ftol 1e-7, searched lambda_f / lambda_kappa / lambda_scalar),
except the clipping level s is bounded to [2, 4] instead of [1, 3]; zsym now starts at the middle of the
bounds (s = 3). Arms: W0 (equal weights) only.

Why: in the [1, 3] runs many regions and all PPMI clinical scores sat at the lower bound s = 1, and a large
share of observations fell beyond the +-s plateaus.

Priority on 2026-11-04: PPMI W0 first, PPMI W50 second, ADNI W0 last (submitted in that order). 80 trials per study, then extended overnight to 200 (`submit_arms.py --n-trials 200`).

CVIC: each Z's trial is the lowest search-fold CVIC; every reported CVIC and the Tukey test use fresh-fold
re-scores (`rescore_cvic.py`, seed + 1000). Compare arms with CVIC and the external checks, not cv_lse
(weighted in W50).

Results: (to fill in)
