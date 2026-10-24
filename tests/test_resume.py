import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from gene_study.experiment import Config, evaluate


class ResumeTests(unittest.TestCase):
    def test_resume_restores_all_models_and_rejects_changed_input(self):
        rng = np.random.default_rng(1)
        X = pd.DataFrame(rng.normal(size=(18, 5)), columns=list('abcde'))
        y = pd.Series(X.a+rng.normal(size=18), index=X.index)
        config = Config(gene_counts=(2,), outer_folds=3, inner_folds=2, trees=2)
        with tempfile.TemporaryDirectory() as root:
            first = evaluate(X, y, root, config)
            with patch('gene_study.experiment.GridSearchCV.fit', side_effect=AssertionError('Unexpected refit')):
                second = evaluate(X, y, root, config, resume=True)
            pd.testing.assert_frame_equal(first, second)
            with self.assertRaisesRegex(ValueError, 'Cannot resume'):
                evaluate(X, y+1, root, config, resume=True)
