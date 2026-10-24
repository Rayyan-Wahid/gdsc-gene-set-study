"""Build and execute a notebook that audits and explores the completed study."""
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient


root = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
    md('# Cancer cell-line drug sensitivity and gene-set size\n\n'
       'Completed MG-132 study: 50, 200, and 1,000 genes, elastic net and random forest. '
       'This notebook loads actual out-of-fold results and checks their integrity. '
       'The expensive nested-CV computation is reproducible with the commands below.'),
    code('from pathlib import Path\nimport json\nimport numpy as np\nimport pandas as pd\n'
         'from IPython.display import display, Image, Markdown\n'
         'ROOT = Path.cwd() if (Path.cwd()/"src").exists() else Path.cwd().parent\n'
         'RUN = ROOT/"results/mg132_full"\n'
         'design = json.loads((ROOT/"results/study_design/selection.json").read_text())\n'
         'manifest = json.loads((RUN/"manifest.json").read_text())'),
    md('## Data provenance and drug selection\n\n'
       'The September 2025 CMP TPM archive was joined to October 2023 GDSC2 responses using Sanger model IDs. '
       'The selected drug has maximum matched coverage, with ties resolved by numeric ID. '
       'This choice was committed before any predictive scores were computed.'),
    code('display(pd.read_csv(ROOT/"results/study_design/drug_coverage.csv").head(10))\n'
         'display(pd.Series({k: design[k] for k in ["drug_id", "drug_name", "matched_cell_lines", "selection_rule"]}))\n'
         'display(pd.Series(design["expression_preparation"]))'),
    md('## Reproducing the computation\n\nInstall the repository and dependencies, then run these commands '
       'from the repository root with the four downloaded files in `data/raw/`. Use new output paths for a new run. '
       'The published results are already available; rerunning model training may take substantial time.\n\n'
       '```bash\npython -m pip install -e ".[notebook]"\n'
       'python scripts/prepare_cmp.py --archive data/raw/rnaseq_merged_20250922.zip --models data/raw/model_list_20261013.csv --output data/cmp_sanger\n'
       'python scripts/select_drug.py --responses data/raw/GDSC2_fitted_dose_response_27Oct23.xlsx --source-xlsx data/raw/GDSC2_fitted_dose_response_27Oct23.xlsx --compounds data/raw/screened_compounds_rel_8.5_inchi.csv\n'
       'python scripts/run_study.py --jobs 2 --output results/mg132_full\n'
       'python scripts/report_study.py\n```'),
    md('## Protocol\n\nFive shared outer folds, three inner folds, seed 42. '
       'Missingness filtering, imputation, variance filtering, Pearson ranking, and scaling are fitted inside training folds. '
       'Inner-fold RMSE chooses parameters. Each outer test set remains untouched until prediction.'),
    code('display(pd.Series(manifest["config"]))\n'
         'display(pd.DataFrame(manifest["tuning"]))'),
    md('## Verify predictions and recompute fold errors'),
    code('pred = pd.read_csv(RUN/"predictions.csv")\n'
         'metrics = pd.read_csv(RUN/"fold_metrics.csv")\n'
         'summary = pd.read_csv(RUN/"summary.csv")\n'
         'assert len(pred) == 944 * 7\n'
         'assert not pred.duplicated(["cell_line_id", "model", "gene_count"]).any()\n'
         'assert (pred.groupby("cell_line_id").fold.nunique() == 1).all()\n'
         'for keys, group in pred.groupby(["fold", "model", "gene_count"]):\n'
         '    fold, model, k = keys\n'
         '    reported = metrics[(metrics.fold == fold) & (metrics.model == model) & (metrics.gene_count == k)].iloc[0]\n'
         '    np.testing.assert_allclose(np.sqrt(np.mean((group.observed-group.predicted)**2)), reported.rmse)\n'
         'print("Passed: one held-out prediction per configuration and cell line; shared folds; recomputed RMSE.")\n'
         'display(summary)'),
    code('display(Image(filename=str(RUN/"rmse_by_gene_count.png")))'),
    md('## Paired fold comparisons\n\nNegative RMSE differences favor the larger gene set. '
       'Fold standard deviations are descriptive and are not confidence intervals. Similar scores do not prove equivalence.'),
    code('display(pd.read_csv(RUN/"paired_comparison_summary.csv"))\n'
         'display(Image(filename=str(RUN/"paired_fold_errors.png")))'),
    code('display(Image(filename=str(RUN/"observed_vs_predicted.png")))'),
    md('## Tissue and expression-source diagnostics\n\nMetadata baselines use only the training-fold group mean, '
       'with an overall training-mean fallback. These are supplementary diagnostics, not causal adjustments.'),
    code('display(pd.read_csv(ROOT/"results/study_design/tissue_composition.csv"))\n'
         'display(pd.read_csv(ROOT/"results/study_design/expression_sources.csv"))\n'
         'display(pd.read_csv(RUN/"metadata_baselines.csv").groupby("baseline")[["rmse", "mae", "r2"]].mean())'),
    md('## Exploratory gene stability\n\nTraining folds overlap, so repeated selection is not independent validation. '
       'Forest importance has no direction; elastic-net weights apply to standardized expression.'),
    code('display(pd.read_csv(RUN/"top_genes_50.csv")[["model", "gene_symbol", "gene", "fold_count", "mean_abs_weight_when_selected"]])'),
    md('## Interpretation and limits\n\nRead [the study report](../results/mg132_full/report.md) and '
       '[biological interpretation](../results/mg132_full/biology.md). '
       'This is one-drug internal cell-line evaluation with mixed expression sources and random tissue-mixed folds. '
       'It provides no external or clinical validation. The best observed configuration is selected from the same '
       'evaluation comparison and should not be described as an independently established winner.'),
]
nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
output = root/'notebooks/02_mg132_results.ipynb'
nbf.write(nb, output)
NotebookClient(nb, timeout=180, resources={'metadata': {'path': str(root)}}).execute()
nbf.write(nb, output)
print(output)
