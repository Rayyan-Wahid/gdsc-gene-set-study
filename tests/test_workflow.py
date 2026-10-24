import tempfile
import unittest
import numpy as np
import pandas as pd
from gene_study.data import prepare
from gene_study.features import GeneSelector
from gene_study.experiment import Config, evaluate


class WorkflowTests(unittest.TestCase):
    def test_training_only_imputation_and_selection(self):
        X = np.array([[1, 4, np.nan, 8], [2, 3, 2, 8], [3, 2, 4, 8], [4, 1, 6, 8.]])
        selector = GeneSelector(2, 0.3).fit(X, [1, 2, 3, 4])
        np.testing.assert_array_equal(selector.indices_, [0, 1])
        before = selector.medians_.copy()
        transformed = selector.transform([[np.nan, 10000, 99999, 0]])
        self.assertEqual(transformed[0, 0], 2.5)
        np.testing.assert_array_equal(selector.medians_, before)

    def test_does_not_silently_reduce_gene_count(self):
        with self.assertRaisesRegex(ValueError, "only 1 eligible"):
            GeneSelector(2).fit([[1, np.nan, 3], [2, np.nan, 3]], [1, 2])

    def test_replicates_and_missing_targets(self):
        expression = pd.DataFrame({"cell_line_id": range(12), "g": range(12)})
        responses = pd.DataFrame({"cell_line_id": list(range(12))+[0, 0],
                                  "drug_id": [7]*14, "dataset": ["GDSC2"]*14,
                                  "LN_IC50": list(range(12))+[2, np.nan]})
        X, y, audit = prepare(expression, responses, "7", "GDSC2")
        self.assertEqual(y.loc["0"], 1)
        self.assertEqual(audit["replicate_rows_collapsed"], 1)
        self.assertEqual(audit["missing_response_rows"], 1)
        self.assertTrue(X.index.is_unique)

    def test_end_to_end_shared_folds_and_baseline(self):
        rng = np.random.default_rng(3)
        X = pd.DataFrame(rng.normal(size=(24, 8)), columns=list("abcdefgh"))
        y = pd.Series(X.a * 2 + rng.normal(size=24), index=X.index)
        with tempfile.TemporaryDirectory() as temp:
            summary = evaluate(X, y, temp, Config(gene_counts=(2,), outer_folds=3,
                               inner_folds=2, trees=3, cache_dir=f"{temp}/cache"))
            pred = pd.read_csv(f"{temp}/predictions.csv")
            self.assertEqual(len(summary), 3)
            self.assertEqual(len(pred), 24*3)
            self.assertTrue((pred.groupby("cell_line_id").fold.nunique() == 1).all())
            base = pred.loc[pred.model == "mean_baseline"]
            for fold, rows in base.groupby("fold"):
                train = y.drop(rows.cell_line_id.tolist())
                np.testing.assert_allclose(rows.predicted, train.mean())
            self.assertEqual(len(pd.read_csv(f"{temp}/selected_genes.csv")), 12)


if __name__ == "__main__":
    unittest.main()
