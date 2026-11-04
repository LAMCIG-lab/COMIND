# 2026_11_04 PPMI zsym + CVIC (fast)

Goal: choose Z with the ordered Tukey test on held-out CVIC, and compare
equal weights (W0) against clinical = half of the loss (W50).

Settings decided 2026-11-04: zsym only (s in [1, 3], no free c), Z = 1-5,
60 trials per study (15 random startup), 2 restarts, 3 CV folds, step 0.05,
inner_ftol 1e-7. Searched: lambda_f, lambda_kappa, lambda_scalar. Fixed:
lambda_jsd = 0, scalar_K_center = 0.25 (no measurable effect in the 10_28 /
11_02 studies). Clinical inputs are still the placeholder z-score.

Compare arms with CVIC and the external checks, not cv_lse (weighted in W50).

Results: (to fill in)
