"""
Continuation-ratio / discrete-time survival ordinal regression for
longitudinal panel data, with sklearn-compatible API.

Contains:
  - AdaBoostBinaryRegressor
  - GradientBoostBinaryClassifier
  - SVMBinaryClassifier
  - OrdinalDiscreteTimeSurvival  (with Platt / prior calibration)
"""

from __future__ import annotations

import inspect
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import AdaBoostRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.validation import check_is_fitted


# ═══════════════════════════════════════════════════════════════════════════
# Base estimator wrappers
# ═══════════════════════════════════════════════════════════════════════════

class AdaBoostBinaryRegressor(BaseEstimator):
    """
    AdaBoost.R2 regressor wrapped to expose a binary-classifier-like
    predict_proba interface for use as a base hazard model.

    Fits an AdaBoostRegressor (Drucker 1997) on a {0, 1} target, treats
    the regression output as P(y=1 | x), and clips to (eps, 1-eps) to
    avoid degenerate logits during downstream calibration.

    Parameters mirror sklearn's AdaBoostRegressor; default loss='square'
    targets the conditional mean (a probability for 0/1 targets), unlike
    the package default 'linear' which targets the conditional median.
    """

    def __init__(
        self,
        estimator=None,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        loss: str = "square",
        random_state=None,
        eps: float = 1e-6,
    ):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.random_state = random_state
        self.eps = eps

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y).astype(float)
        unique = np.unique(y)
        if not set(unique.tolist()).issubset({0.0, 1.0}):
            raise ValueError(
                f"AdaBoostBinaryRegressor expects binary {{0, 1}} targets; got {unique}"
            )
        self.regressor_ = AdaBoostRegressor(
            estimator=self.estimator,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            loss=self.loss,
            random_state=self.random_state,
        )
        self.regressor_.fit(X, y, sample_weight=sample_weight)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "regressor_")
        p1 = np.clip(self.regressor_.predict(X), self.eps, 1.0 - self.eps)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class GradientBoostBinaryClassifier(BaseEstimator):
    """
    HistGradientBoostingClassifier wrapped to match the
    AdaBoostBinaryRegressor interface used by OrdinalDiscreteTimeSurvival.

    Each threshold k in the continuation-ratio model gets its own clone
    of this estimator, fitted on the at-risk subset for that threshold.

    Parameters
    ----------
    max_iter : int
        Number of boosting rounds (equivalent to n_estimators).
    learning_rate : float
        Shrinkage applied to each tree.
    max_leaf_nodes : int
        Max leaves per tree. Controls complexity.
        31 ≈ depth-5, 15 ≈ depth-4, 7 ≈ depth-3.
    l2_regularization : float
        L2 penalty on leaf values. Helps with imbalanced/small at-risk sets.
    early_stopping : bool
        If True, uses internal validation fraction to stop early.
    random_state : int or None
    """

    def __init__(
        self,
        max_iter: int = 100,
        learning_rate: float = 0.1,
        max_leaf_nodes: int = 31,
        l2_regularization: float = 0.0,
        early_stopping: bool = False,
        random_state=None,
    ):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.l2_regularization = l2_regularization
        self.early_stopping = early_stopping
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y).astype(float)
        unique = np.unique(y)
        if not set(unique.tolist()).issubset({0.0, 1.0}):
            raise ValueError(
                f"GradientBoostBinaryClassifier expects binary {{0,1}} targets; "
                f"got {unique}"
            )
        self.clf_ = HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            l2_regularization=self.l2_regularization,
            early_stopping=self.early_stopping,
            random_state=self.random_state,
            class_weight=None,  # handled via sample_weight
        )
        self.clf_.fit(X, y.astype(int), sample_weight=sample_weight)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "clf_")
        return self.clf_.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class SVMBinaryClassifier(BaseEstimator):
    """
    SVC wrapped to match the AdaBoostBinaryRegressor interface.
    Uses hinge loss (SVM margin) natively.

    Uses decision_function scores converted to pseudo-probabilities via
    sigmoid — OrdinalDiscreteTimeSurvival's Platt calibration then
    calibrates these properly. This avoids the double-calibration problem
    of SVC(probability=True) which runs its own internal 5-fold CV.

    Parameters
    ----------
    C : float
        SVM regularization — smaller = stronger regularization.
    kernel : str
        'rbf' (nonlinear) or 'linear'. Use 'linear' for small at-risk sets.
    gamma : str or float
        Kernel coefficient for 'rbf'. Default 'scale'.
    random_state : int or None
    """

    def __init__(self, C=1.0, kernel="rbf", gamma="scale", random_state=None):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        self.clf_ = SVC(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            probability=False,     # OrdinalDiscreteTimeSurvival handles calibration
            random_state=self.random_state,
            class_weight=None,     # handled via sample_weight
        )
        self.clf_.fit(X, y.astype(int), sample_weight=sample_weight)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "clf_")
        # Convert decision function scores to pseudo-probabilities via sigmoid.
        # Platt calibration in OrdinalDiscreteTimeSurvival calibrates these.
        scores = self.clf_.decision_function(X)
        p1 = 1.0 / (1.0 + np.exp(-scores))
        p1 = np.clip(p1, 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ═══════════════════════════════════════════════════════════════════════════
# OrdinalDiscreteTimeSurvival
# ═══════════════════════════════════════════════════════════════════════════

class OrdinalDiscreteTimeSurvival(ClassifierMixin, BaseEstimator):
    """
    Continuation-ratio / discrete-time survival ordinal classifier for
    longitudinal panel data.

    For an ordered target with classes c_1 < c_2 < ... < c_K, fits K-1
    independent binary "hazard" models, one per threshold. For threshold
    k (separating c_k from c_{k+1}), the model estimates

        h_k(j; x) = P(Y_{i,j} >= c_{k+1} | not-yet-crossed, x_{i,j})

    on the at-risk subset (subjects who have not yet crossed k) at each
    visit j. Subjects are right-censored at their last observed visit if
    they never cross.

    At prediction time, hazards are accumulated within subject in time
    order to give monotone (in visit) cumulative threshold-crossing
    probabilities

        F_k(j) = 1 - prod_{j' <= j} (1 - h_k(j'; x_{j'}))

    from which per-class probabilities at each visit are derived via the
    continuation-ratio identity.

    Parameters
    ----------
    order : array-like, optional
        Expected ordered class values, lowest to highest. REQUIRED if y
        is non-numeric. If y is numeric and `order` is None, classes are
        taken as np.unique(y) sorted ascending.

    base_estimator : sklearn classifier, optional
        Binary classifier with `predict_proba`. One clone is fitted per
        threshold. Default: AdaBoostBinaryRegressor.

    subj_id_col, time_col : str
        Names of the subject-id and within-subject relative-time columns
        in X. Time values are used only to ORDER visits within a subject.

    include_visit_index : bool, default=True
        Include the within-subject visit number (1, 2, ...) as a feature.

    class_weight : {'balanced', None}, default='balanced'
        If 'balanced', applies inverse-frequency sample weights to each
        threshold's binary problem.

    enforce_threshold_ordering : bool, default=True
        At prediction time, enforce P(Y >= c_{k+1}) >= P(Y >= c_{k+2})
        by post-hoc cumulative max from the highest threshold downward.

    calibration_method : {'platt', 'prior', None}, default='platt'
        Per-threshold probability calibration applied at predict time.

        - 'platt': Held-out Platt scaling. A fraction of subjects
          (`calibration_fraction`) is held out from base-model fitting;
          raw predictions on that set are mapped to calibrated probabilities
          via sigmoid(a * logit(p_raw) + b) fit per threshold. Works for
          any base estimator and absorbs the balanced-weighting bias.

        - 'prior': Closed-form intercept correction for balanced weighting
          bias. EXACT for LogisticRegression, APPROXIMATE for nonlinear
          estimators.

        - None: no calibration; raw probabilities used as-is.

    calibration_fraction : float, default=0.25
        Fraction of training subjects held out for Platt calibration.
        Subject-level split to avoid within-subject leakage.

    calibration_random_state : int or None, default=None
        Seed for the subject-level calibration split.

    Attributes
    ----------
    classes_ : np.ndarray, shape (K,)
    threshold_models_ : dict[int, estimator or tuple]
    platt_models_ : dict[int, tuple or None]
    priors_ : dict[int, float]
    feature_names_ : list[str]
    K_ : int

    Notes
    -----
    Population-averaged model — does not estimate subject random effects.
    Cross-validate by SUBJECT (not by row) to avoid leakage.
    """

    def __init__(
        self,
        order: Optional[Sequence] = None,
        base_estimator=None,
        subj_id_col: str = "subj_id",
        time_col: str = "time",
        include_visit_index: bool = True,
        class_weight: Optional[str] = "balanced",
        enforce_threshold_ordering: bool = True,
        calibration_method: Optional[str] = "platt",
        calibration_fraction: float = 0.25,
        calibration_random_state: Optional[int] = None,
    ):
        self.order = order
        self.base_estimator = base_estimator
        self.subj_id_col = subj_id_col
        self.time_col = time_col
        self.include_visit_index = include_visit_index
        self.class_weight = class_weight
        self.enforce_threshold_ordering = enforce_threshold_ordering
        self.calibration_method = calibration_method
        self.calibration_fraction = calibration_fraction
        self.calibration_random_state = calibration_random_state

    # ── internal helpers ───────────────────────────────────────────────────

    def _validate_classes(self, y: np.ndarray) -> np.ndarray:
        observed = pd.unique(y)
        if self.order is None:
            if not np.issubdtype(np.asarray(y).dtype, np.number):
                raise ValueError(
                    f"y has non-numeric dtype {np.asarray(y).dtype}; "
                    "non-numeric targets require an explicit `order=` argument."
                )
            return np.sort(observed)
        order_arr = np.asarray(self.order)
        if len(set(order_arr.tolist())) != len(order_arr):
            raise ValueError("`order` contains duplicate values.")
        extras = pd.Index(observed).difference(pd.Index(order_arr)).tolist()
        if extras:
            raise ValueError(
                f"y contains values not in `order`: {extras}. "
                f"Provided order: {list(order_arr)}"
            )
        return order_arr

    def _split_panel(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "X must be a pandas DataFrame containing "
                f"`{self.subj_id_col}` and `{self.time_col}` columns."
            )
        for col in (self.subj_id_col, self.time_col):
            if col not in X.columns:
                raise ValueError(f"Required column `{col}` missing from X.")
        subj = X[self.subj_id_col].to_numpy()
        time = X[self.time_col].to_numpy()
        feats = X.drop(columns=[self.subj_id_col, self.time_col])
        if y is None:
            return feats, subj, time
        y_arr = np.asarray(y)
        if len(y_arr) != len(X):
            raise ValueError(f"len(y) {len(y_arr)} != len(X) {len(X)}.")
        return feats, subj, time, y_arr

    @staticmethod
    def _within_subject_visit_index(subj_sorted: np.ndarray) -> np.ndarray:
        s = pd.Series(subj_sorted)
        return s.groupby(s, sort=False).cumcount().to_numpy() + 1

    @staticmethod
    def _sort_by_subject_time(subj: np.ndarray, time: np.ndarray) -> np.ndarray:
        df = pd.DataFrame({"_s": subj, "_t": time})
        return df.sort_values(["_s", "_t"], kind="mergesort").index.to_numpy()

    @staticmethod
    def _logit(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        p = np.clip(p, eps, 1.0 - eps)
        return np.log(p / (1.0 - p))

    @staticmethod
    def _fit_platt(p_raw: np.ndarray, y: np.ndarray, min_n: int = 10):
        """Fit sigmoid(a * logit(p_raw) + b). Returns (a, b) or None."""
        if len(y) < min_n or len(np.unique(y)) < 2:
            return None
        z = OrdinalDiscreteTimeSurvival._logit(p_raw).reshape(-1, 1)
        lr = LogisticRegression(C=1e12, max_iter=1000, solver="lbfgs")
        lr.fit(z, y)
        return float(lr.coef_[0, 0]), float(lr.intercept_[0])

    @staticmethod
    def _apply_platt(p_raw: np.ndarray, ab) -> np.ndarray:
        a, b = ab
        z = OrdinalDiscreteTimeSurvival._logit(p_raw)
        return 1.0 / (1.0 + np.exp(-(a * z + b)))

    # ── fit ───────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y):
        if self.calibration_method not in (None, "platt", "prior"):
            raise ValueError(
                f"calibration_method must be None, 'platt', or 'prior'; "
                f"got {self.calibration_method!r}."
            )
        if self.calibration_method == "platt" and not (
            0.0 < self.calibration_fraction < 1.0
        ):
            raise ValueError(
                f"calibration_fraction must be in (0, 1); "
                f"got {self.calibration_fraction!r}."
            )

        classes = self._validate_classes(np.asarray(y))
        self.classes_ = classes
        self.K_ = len(classes)
        if self.K_ < 2:
            raise ValueError(f"Need at least 2 classes; got {self.K_}.")
        rank_map = {c: i for i, c in enumerate(classes)}

        feats, subj, time, y_arr = self._split_panel(X, y)
        y_rank = np.array([rank_map[v] for v in y_arr], dtype=int)

        order_idx = self._sort_by_subject_time(subj, time)
        feats_s = feats.iloc[order_idx].reset_index(drop=True)
        subj_s = subj[order_idx]
        y_rank_s = y_rank[order_idx]

        if self.include_visit_index:
            feats_s = feats_s.copy()
            feats_s["__visit_index__"] = self._within_subject_visit_index(subj_s)
        self.feature_names_ = list(feats_s.columns)

        # ── subject-level split for Platt calibration ──────────────────────
        unique_subj = pd.unique(subj_s)
        if self.calibration_method == "platt":
            rng = np.random.default_rng(self.calibration_random_state)
            shuffled = unique_subj.copy()
            rng.shuffle(shuffled)
            n_cal = max(1, int(round(self.calibration_fraction * len(shuffled))))
            if n_cal >= len(shuffled):
                raise ValueError(
                    f"calibration_fraction {self.calibration_fraction} leaves no "
                    f"subjects for base-model fitting (n_subjects={len(shuffled)})."
                )
            cal_subjects = set(shuffled[:n_cal].tolist())
            is_cal = np.array([s in cal_subjects for s in subj_s])
        else:
            is_cal = np.zeros(len(subj_s), dtype=bool)
        is_base_fit = ~is_cal

        base = (
            self.base_estimator
            if self.base_estimator is not None
            else AdaBoostBinaryRegressor()
        )
        if not hasattr(base, "predict_proba"):
            raise TypeError(
                f"base_estimator {type(base).__name__} must implement predict_proba."
            )
        supports_sw = "sample_weight" in inspect.signature(base.fit).parameters

        df_subj = pd.Series(subj_s)
        Xv = feats_s.to_numpy()
        self.threshold_models_ = {}
        self.platt_models_ = {}
        self.priors_ = {}

        for k in range(self.K_ - 1):
            crossed = (y_rank_s > k).astype(int)
            cs = pd.Series(crossed).groupby(df_subj, sort=False).cumsum().to_numpy()
            cs_before = cs - crossed
            at_risk = cs_before == 0

            if not at_risk.any():
                self.threshold_models_[k] = ("constant", 0.0)
                self.platt_models_[k] = None
                self.priors_[k] = 0.0
                continue

            base_mask = at_risk & is_base_fit
            cal_mask  = at_risk & is_cal
            X_k = Xv[base_mask]
            y_k = crossed[base_mask]

            self.priors_[k] = float(y_k.mean()) if len(y_k) > 0 else 0.0

            if len(y_k) == 0 or len(np.unique(y_k)) < 2:
                const = float(y_k.mean()) if len(y_k) > 0 else 0.0
                self.threshold_models_[k] = ("constant", const)
                self.platt_models_[k] = None
                continue

            sw = (
                compute_sample_weight("balanced", y_k)
                if self.class_weight == "balanced"
                else None
            )
            if sw is not None and not supports_sw:
                raise ValueError(
                    f"base_estimator {type(base).__name__} does not support "
                    "sample_weight; pass class_weight=None or use a different estimator."
                )

            est = clone(base)
            est.fit(X_k, y_k, sample_weight=sw) if supports_sw else est.fit(X_k, y_k)
            self.threshold_models_[k] = est

            # ── Platt fit on held-out calibration rows ─────────────────────
            if self.calibration_method == "platt" and cal_mask.any():
                p_raw = est.predict_proba(Xv[cal_mask])[:, 1]
                self.platt_models_[k] = self._fit_platt(p_raw, crossed[cal_mask])
            else:
                self.platt_models_[k] = None

        return self

    # ── predict ───────────────────────────────────────────────────────────

    def _cumulative_crossing_probs(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self)
        feats, subj, time = self._split_panel(X)
        order_idx = self._sort_by_subject_time(subj, time)
        feats_s = feats.iloc[order_idx].reset_index(drop=True)
        subj_s = subj[order_idx]

        if self.include_visit_index:
            feats_s = feats_s.copy()
            feats_s["__visit_index__"] = self._within_subject_visit_index(subj_s)

        missing = [c for c in self.feature_names_ if c not in feats_s.columns]
        if missing:
            raise ValueError(f"Features missing in X: {missing}")
        feats_s = feats_s[self.feature_names_]

        Km1 = self.K_ - 1
        Xv  = feats_s.to_numpy()
        hazards = np.zeros((len(feats_s), Km1))

        for k in range(Km1):
            m = self.threshold_models_[k]
            if isinstance(m, tuple) and m[0] == "constant":
                hazards[:, k] = m[1]
                continue

            h = m.predict_proba(Xv)[:, 1]

            if self.calibration_method == "platt":
                ab = self.platt_models_.get(k)
                if ab is not None:
                    h = self._apply_platt(h, ab)
                elif self.class_weight == "balanced":
                    # Fallback to prior correction if Platt couldn't be fit
                    pi = self.priors_.get(k, 0.5)
                    if 0.0 < pi < 1.0:
                        h = 1.0 / (1.0 + np.exp(
                            -(self._logit(h) + np.log(pi / (1.0 - pi)))))

            elif self.calibration_method == "prior":
                if self.class_weight == "balanced":
                    pi = self.priors_.get(k, 0.5)
                    if 0.0 < pi < 1.0:
                        h = 1.0 / (1.0 + np.exp(
                            -(self._logit(h) + np.log(pi / (1.0 - pi)))))

            # calibration_method is None → use raw h

            hazards[:, k] = h

        log_surv = np.log(np.clip(1.0 - hazards, 1e-12, 1.0))
        cum = (
            pd.DataFrame(log_surv)
            .groupby(subj_s, sort=False)
            .cumsum()
            .to_numpy()
        )
        F = 1.0 - np.exp(cum)

        if self.enforce_threshold_ordering and Km1 >= 2:
            for k in range(Km1 - 2, -1, -1):
                F[:, k] = np.maximum(F[:, k], F[:, k + 1])

        unsort = np.argsort(order_idx)
        return F[unsort]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns array of shape (n_rows, K) with per-class probabilities
        in `self.classes_` order.
        """
        F = self._cumulative_crossing_probs(X)
        n, Km1 = F.shape
        K = Km1 + 1
        proba = np.empty((n, K))
        proba[:, 0] = 1.0 - F[:, 0]
        for m in range(1, K - 1):
            proba[:, m] = F[:, m - 1] - F[:, m]
        proba[:, K - 1] = F[:, K - 2]
        proba = np.clip(proba, 0.0, None)
        s = proba.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        return proba / s

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Modal stage prediction per row."""
        idx = np.argmax(self.predict_proba(X), axis=1)
        return np.asarray(self.classes_)[idx]
