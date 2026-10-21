import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class GeneSelector(TransformerMixin, BaseEstimator):
    """Training-only missingness filter, median imputation, and Pearson ranking."""
    def __init__(self, k=50, max_missing=0.2):
        self.k = k
        self.max_missing = max_missing

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or len(y) != len(X) or not np.isfinite(y).all():
            raise ValueError("Expected a 2D expression matrix and finite aligned response.")
        if not isinstance(self.k, int) or self.k < 1 or not 0 <= self.max_missing < 1:
            raise ValueError("k must be positive; max_missing must be in [0, 1).")
        if np.isinf(X).any():
            raise ValueError("Infinite expression values are not allowed.")
        self.n_features_in_ = X.shape[1]
        retained = np.flatnonzero(np.mean(np.isnan(X), axis=0) <= self.max_missing)
        medians = np.nanmedian(X[:, retained], axis=0)
        values = np.where(np.isnan(X[:, retained]), medians, X[:, retained])
        variable = np.var(values, axis=0) > 0
        retained, medians, values = retained[variable], medians[variable], values[:, variable]
        if len(retained) < self.k:
            raise ValueError(f"Requested {self.k} genes but only {len(retained)} eligible in this training fold.")
        centered = values - values.mean(axis=0)
        target = y - y.mean()
        denom = np.linalg.norm(centered, axis=0) * np.linalg.norm(target)
        scores = np.divide(np.abs(centered.T @ target), denom,
                           out=np.zeros(len(retained)), where=denom > 0)
        order = np.argsort(-scores, kind="stable")[:self.k]
        self.indices_ = retained[order]
        self.medians_ = medians[order]
        self.scores_ = scores[order]
        return self

    def transform(self, X):
        check_is_fitted(self, "indices_")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_ or np.isinf(X).any():
            raise ValueError("Expression dimensions or values do not match training data.")
        values = X[:, self.indices_]
        return np.where(np.isnan(values), self.medians_, values)
