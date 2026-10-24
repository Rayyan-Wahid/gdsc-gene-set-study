"""Produce paired comparisons and cohort diagnostics from out-of-fold predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def markdown(frame):
    columns = list(frame.columns)
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---']*len(columns)) + ' |']
    for row in frame.itertuples(index=False, name=None):
        lines.append('| ' + ' | '.join(f'{x:.4f}' if isinstance(x, float) else str(x) for x in row) + ' |')
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', default='results/mg132_complete')
    p.add_argument('--prepared', default='data/cmp_sanger')
    a = p.parse_args()
    run = Path(a.run)
    summary = pd.read_csv(run/'summary.csv')
    folds = pd.read_csv(run/'fold_metrics.csv')
    pred = pd.read_csv(run/'predictions.csv')
    selected = pd.read_csv(run/'selected_genes.csv')
    frequency = pd.read_csv(run/'selection_frequency.csv')
    manifest = json.loads((run/'manifest.json').read_text())
    annotations = pd.read_csv(Path(a.prepared)/'sample_annotations.csv')
    gene_names = pd.read_csv(Path(a.prepared)/'gene_annotations.csv')
    cohort = pred.loc[pred.model == 'mean_baseline', ['cell_line_id', 'fold', 'observed']]
    assert cohort.cell_line_id.is_unique
    cohort = cohort.merge(annotations, left_on='cell_line_id', right_on='model_id', validate='one_to_one')
    pairs = []
    for model in ['elastic_net', 'random_forest']:
        table = folds.loc[folds.model == model].pivot(index='fold', columns='gene_count', values='rmse')
        for small, large in [(50, 200), (50, 1000), (200, 1000)]:
            difference = table[large]-table[small]
            for fold, value in difference.items():
                pairs.append({'model': model, 'smaller_genes': small, 'larger_genes': large,
                              'fold': fold, 'rmse_larger_minus_smaller': value})
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(run/'paired_fold_differences.csv', index=False)
    comparison = pairs.groupby(['model', 'smaller_genes', 'larger_genes']).agg(
        mean_rmse_difference=('rmse_larger_minus_smaller', 'mean'),
        folds_larger_better=('rmse_larger_minus_smaller', lambda x: int((x < 0).sum()))).reset_index()
    comparison.to_csv(run/'paired_comparison_summary.csv', index=False)
    # Diagnostic baselines use metadata, never held-out target values, for prediction.
    diagnostics = []
    for feature in ['tissue', 'expression_source']:
        for fold in sorted(cohort.fold.unique()):
            train, test = cohort.loc[cohort.fold != fold], cohort.loc[cohort.fold == fold]
            means = train.groupby(feature).observed.mean()
            estimate = test[feature].map(means).fillna(train.observed.mean())
            diagnostics.append({'baseline': feature+'_training_mean', 'fold': int(fold),
                                'rmse': np.sqrt(mean_squared_error(test.observed, estimate)),
                                'mae': mean_absolute_error(test.observed, estimate),
                                'r2': r2_score(test.observed, estimate)})
    diagnostic = pd.DataFrame(diagnostics)
    diagnostic.to_csv(run/'metadata_baselines.csv', index=False)
    diagnostic_summary = diagnostic.groupby('baseline')[['rmse', 'mae', 'r2']].mean().reset_index()
    importance = selected.assign(abs_weight=selected.weight.abs()).groupby(['model', 'gene_count', 'gene']).agg(
        mean_abs_weight_when_selected=('abs_weight', 'mean')).reset_index()
    annotated = frequency.merge(gene_names, left_on='gene', right_on='ensembl_gene_id', validate='many_to_one')
    annotated = annotated.merge(importance, on=['model', 'gene_count', 'gene'], validate='one_to_one')
    annotated.to_csv(run/'annotated_gene_stability.csv', index=False)
    top = annotated.loc[annotated.gene_count == 50].sort_values(
        ['model', 'fold_count', 'mean_abs_weight_when_selected'], ascending=[True, False, False]).groupby('model').head(15)
    top.to_csv(run/'top_genes_50.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for axis, model in zip(axes, ['elastic_net', 'random_forest']):
        sub = folds.loc[folds.model == model]
        for fold, values in sub.groupby('fold'):
            values = values.sort_values('gene_count')
            axis.plot(values.gene_count, values.rmse, marker='o', label=f'Fold {fold}')
        axis.set(title=model.replace('_', ' ').title(), xlabel='Selected genes', ylabel='RMSE (LN_IC50)')
        axis.legend(fontsize=8)
    fig.suptitle('Paired outer-fold errors for MG-132')
    fig.tight_layout()
    fig.savefig(run/'paired_fold_errors.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    limits = [pred.observed.min(), pred.observed.max()]
    for row, model in enumerate(['elastic_net', 'random_forest']):
        for col, k in enumerate([50, 200, 1000]):
            sub = pred.loc[(pred.model == model) & (pred.gene_count == k)]
            ax = axes[row, col]
            ax.scatter(sub.observed, sub.predicted, s=8, alpha=0.3)
            ax.plot(limits, limits, '--', color='gray')
            ax.set(title=f'{model.replace("_", " ")} / {k} genes', xlabel='Observed LN_IC50', ylabel='Predicted LN_IC50')
    fig.suptitle('MG-132 predictions on held-out cell lines')
    fig.tight_layout()
    fig.savefig(run/'observed_vs_predicted.png', dpi=160)
    plt.close(fig)
    best = summary.loc[summary.model != 'mean_baseline'].sort_values('rmse_mean').iloc[0]
    baseline = summary.loc[summary.model == 'mean_baseline'].iloc[0]
    relative = 100 * (baseline.rmse_mean - best.rmse_mean) / baseline.rmse_mean
    comparison_sentence = f"The lowest observed model RMSE is {abs(relative):.1f}% {'below' if relative >= 0 else 'above'} the baseline mean RMSE."
    table = summary[['model','gene_count','rmse_mean','rmse_std','mae_mean','mae_std','r2_mean','r2_std']]
    text = f'''# MG-132 gene-set comparison

## Study and primary findings

The study evaluates MG-132 (GDSC2 drug ID 1862) in {len(cohort)} matched cell lines.
Drug selection was frozen before model fitting using maximum matched coverage.
The lowest mean outer-fold RMSE is {best.rmse_mean:.4f}, from {best.model} with
{int(best.gene_count)} genes; the training-mean baseline RMSE is {baseline.rmse_mean:.4f}.
{comparison_sentence}
This identifies the best observed configuration in this comparison, not an independently
validated winner. Larger gene sets must be judged using the paired fold differences below.

## Data and preprocessing

Drug responses: GDSC2 fitted dose responses dated 27 October 2023. Expression:
CMP merged RSEM TPM archive dated 22 September 2025. Model annotations: 13 October 2026.
Expression is supplied as log2(TPM+1), preserved without another log transformation.
Sanger model IDs join the sources directly. Fourteen rows with duplicated Ensembl
IDs were excluded without inspecting responses, leaving 41,131 input genes.
The expression matrix member passed its ZIP CRC check. Source SHA-256 hashes are in
the committed study design and run manifest. All 944 matched models are cell lines;
their expression sources comprise 547 Broad and 397 Sanger measurements across 28 tissues.
Of 969 MG-132 response cell lines, 25 lacked matched expression. There were no missing
MG-132 responses or duplicate MG-132 cell-line rows in the downloaded workbook.

## Evaluation protocol

Five shuffled outer folds (seed 42) estimate performance; three shuffled inner folds
tune parameters by RMSE. Every training fold independently filters genes with more
than 20% missingness, imputes medians, removes zero-variance genes, and ranks genes by
absolute Pearson correlation with response. Elastic net also standardizes using
training statistics. Both models use the same ranking and outer splits.
Elastic net uses seeded random coordinate updates and retries unconverged fits with
increasing iteration budgets; persistent nonconvergence raises an error. An initial
cyclic-coordinate trial was stopped after convergence warnings, before any complete
outer fold was reported. The final study reruns all configurations.
The elastic-net grid is alpha 0.01/0.1/1 and l1 ratio 0.1/0.5/0.9. Random forests use
200 trees, max_features 0.3/1, and min_samples_leaf 2/5. Refit preprocessing is learned
on the complete outer training fold after tuning. No test-fold targets enter fitting.
Cached preprocessing reuses only identical training inputs and parameters.

## Performance

Values are means and sample standard deviations across five outer folds.
RMSE and MAE are in LN_IC50 units; R2 is calculated separately in each test fold.

{markdown(table)}

![RMSE by gene count](rmse_by_gene_count.png)

## Paired gene-count comparisons

Negative differences favor the larger gene set. The number of favorable folds helps
show whether an average difference is consistent. These dependent CV folds do not
provide an independent replication study, and no equivalence claim is made.

{markdown(comparison)}

![Paired errors](paired_fold_errors.png)

## Tissue and expression-source diagnostics

Supplementary metadata-only baselines predict a tissue-specific or source-specific
training mean using the same outer folds; unseen categories fall back to the overall
training mean. These diagnostics were not used to choose the drug or tune the models.
Their results help assess whether coarse tissue/source differences are predictive;
they do not remove confounding from the expression models.

{markdown(diagnostic_summary)}

## Selected genes

`annotated_gene_stability.csv` lists selection frequency and model weights.
`top_genes_50.csv` provides a compact list for biological review. Selection frequency
is descriptive: outer training sets overlap. Elastic-net coefficients refer to
standardized expression; random-forest impurity importance has no effect direction
and can be biased. Neither establishes causation or a drug-specific mechanism.

## Limitations

This is internal evaluation of one drug and dataset. Random cell-line folds permit
shared tissue and source composition across folds; they do not test generalization
to unseen tissues, patients, or laboratories. The annotation audit found no shared
patient IDs among matched models, but relatedness was not independently validated.
The merged expression sources may carry batch differences; source adjustment was not
performed. Fitted LN_IC50 values have assay/model uncertainty. Outlying responses
were retained under the prespecified rules and can affect RMSE. There is no external
test dataset or clinical validation. Similar means do not demonstrate equivalence.
MG-132 is a laboratory proteasome inhibitor; results must not be presented as patient
treatment recommendations. Biological interpretation remains exploratory.

## Sources

- [Official expression processing documentation](https://depmap.sanger.ac.uk/documentation/datasets/expression/)
- [Official GDSC assay and fitted-response documentation](https://depmap.sanger.ac.uk/documentation/datasets/drug-sensitivity/)
- [Official dataset downloads](https://cellmodelpassports.sanger.ac.uk/downloads)
- [MG132 proteasome and calpain effects in an experimental study](https://www.jci.org/articles/view/119513)
'''
    (run/'report.md').write_text(text, encoding='utf-8')
    print(markdown(table))
    print(markdown(comparison))
    print(top[['model','gene_symbol','fold_count','mean_abs_weight_when_selected']].to_string(index=False))


if __name__ == '__main__':
    main()
