import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
from statsmodels.miscmodels.ordinal_model import OrderedModel

class WeightedOrderedModel(OrderedModel):
    def __init__(self, endog, exog, sample_weights=None, distr="logit", **kwargs):
        super().__init__(endog, exog, distr=distr, **kwargs)
        if sample_weights is None:
            sample_weights = np.ones(self.nobs)
        w = np.asarray(sample_weights, dtype=float)
        self.sample_weights = w * (self.nobs / w.sum())

    def loglikeobs(self, params):
        return self.sample_weights * super().loglikeobs(params)

    def score_obs(self, params):
        return self.sample_weights[:, None] * super().score_obs(params)