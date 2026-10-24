"""Estimators that fail explicitly rather than retain unconverged fits."""
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet


class ConvergedElasticNet(ElasticNet):
    """Continue from the previous solution when a fit needs more iterations."""
    def fit(self, X, y, sample_weight=None, check_input=True):
        original_limit, original_warm = self.max_iter, self.warm_start
        self.convergence_retries_ = 0
        try:
            for attempt in range(4):
                self.max_iter = original_limit * 2**attempt
                self.warm_start = original_warm if attempt == 0 else True
                with warnings.catch_warnings(record=True) as recorded:
                    warnings.simplefilter('always', ConvergenceWarning)
                    super().fit(X, y, sample_weight=sample_weight, check_input=check_input)
                unconverged = False
                for warning in recorded:
                    if issubclass(warning.category, ConvergenceWarning):
                        unconverged = True
                    else:
                        warnings.warn(str(warning.message), warning.category)
                if not unconverged:
                    self.convergence_retries_ = attempt
                    return self
            raise RuntimeError('Elastic net did not converge after four increasing iteration budgets.')
        finally:
            self.max_iter, self.warm_start = original_limit, original_warm
