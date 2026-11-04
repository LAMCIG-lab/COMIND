# 2026_11_04 ADNI zsym + CVIC (fast)

Goal: choose Z with the ordered Tukey test on held-out CVIC, and compare
equal weights (W0) against clinical = half of the loss (W50).

Settings decided 2026-11-04: zsym only (s in [1, 3], no free c), Z = 1-5,
80 trials per study (15 random startup; launched at 60, W0 extended by 20 the same day), 2 restarts, 3 CV folds, step 0.05,
inner_ftol 1e-7. Searched: lambda_f, lambda_kappa, lambda_scalar. Fixed:
lambda_jsd = 0, scalar_K_center = 0.25 (no measurable effect in the 10_28 /
11_02 studies). Clinical inputs are still the placeholder z-score.

Order: W0 first. W50 was paused at 6-20 trials per study and is deferred to a later run
(resubmit with --arms W50; the studies resume where they stopped).

CVIC: each Z's trial is the lowest search-fold CVIC; every reported CVIC and the Tukey test use
fresh-fold re-scores (rescore_cvic.py, seed + 1000). Compare arms with CVIC and the external
checks, not cv_lse (weighted in W50).

Results: (to fill in)
