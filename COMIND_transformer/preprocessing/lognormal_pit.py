"""Log-normal -> Uniform(0,1) PIT, wired into sklearn two ways.

Functions operate column-wise on 2D arrays (sklearn convention).

Small EPS added before log to handle zero-valued observations that arise
naturally from the affine min-shift transform on Z-scores and from clinical
scores where 0 is a legitimate observed value (TD_score, PIGD_score).
"""
import numpy as np
from scipy.stats import norm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.validation import validate_data

EPS = 1e-6  # per prior conversation: "small epsilon is reasonable"


def fit_lognormal(X):
    """Column-wise MLE. Returns (mu, sigma) arrays of shape (n_features,)."""
    X = np.asarray(X, dtype=float)
    if np.any(X < 0):
        raise ValueError(
            "Log-normal PIT requires non-negative data (negative values would "
            "produce complex logs). Apply sign flip / affine shift / clinical "
            "flip before this transform."
        )
    logX = np.log(X + EPS)
    return logX.mean(axis=0), logX.std(axis=0, ddof=0)


def lognormal_to_uniform(X, mu, sigma):
    """Push data through the fitted log-normal CDF -> Uniform(0,1)."""
    return norm.cdf((np.log(np.asarray(X, dtype=float) + EPS) - mu) / sigma)


def make_function_transformer(mu, sigma):
    return FunctionTransformer(
        lognormal_to_uniform,
        kw_args={"mu": mu, "sigma": sigma},
        check_inverse=False,
    )


class LogNormalToUniform(BaseEstimator, TransformerMixin):
    """Estimates log-normal params in fit(), applies the PIT in transform().

    Fit on TRAINING data only; the same fitted mu_/sigma_ are then applied to
    validation data via transform(). This preserves the train-only-fit
    discipline required for any preprocessing that touches distributional
    statistics.
    """

    def fit(self, X, y=None):
        X = validate_data(self, X, dtype=float)
        self.mu_, self.sigma_ = fit_lognormal(X)
        return self

    def transform(self, X):
        X = validate_data(self, X, dtype=float, reset=False)
        return lognormal_to_uniform(X, self.mu_, self.sigma_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)
