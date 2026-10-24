import unittest
import numpy as np
from gene_study.models import ConvergedElasticNet


class ConvergenceTests(unittest.TestCase):
    def test_unconverged_fit_is_rejected(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(30, 50))
        y = rng.normal(size=30)
        model = ConvergedElasticNet(alpha=1e-12, max_iter=1, tol=1e-15)
        with self.assertRaisesRegex(RuntimeError, 'did not converge'):
            model.fit(X, y)
        self.assertEqual(model.max_iter, 1)
        self.assertFalse(model.warm_start)

    def test_converged_fit_keeps_constructor_parameters(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(40, 5))
        model = ConvergedElasticNet(alpha=0.1, max_iter=1000).fit(X, X[:, 0])
        self.assertEqual(model.max_iter, 1000)
        self.assertEqual(model.convergence_retries_, 0)
        self.assertLess(np.mean((model.predict(X)-X[:, 0])**2), 0.1)
