"""Reproduce cohort summaries without exporting unrelated clinical annotations."""
import argparse
import json
from pathlib import Path
import pandas as pd


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--responses', default='data/gdsc2_source.csv')
    p.add_argument('--prepared', default='data/cmp_sanger')
    p.add_argument('--models', required=True)
    p.add_argument('--design', default='results/study_design')
    a = p.parse_args()
    out = Path(a.design)
    frozen = json.loads((out/'selection.json').read_text())
    r = pd.read_excel(a.responses) if a.responses.endswith('.xlsx') else pd.read_csv(a.responses)
    r = r.loc[(r.DRUG_ID == frozen['drug_id']) & (r.DATASET == frozen['dataset'])]
    response = r.groupby('SANGER_MODEL_ID').LN_IC50.mean()
    samples = pd.read_csv(Path(a.prepared)/'sample_annotations.csv')
    cohort = samples.merge(response.rename('LN_IC50'), left_on='model_id', right_index=True, validate='one_to_one')
    cohort = cohort.loc[cohort.LN_IC50.notna()]
    cohort.groupby('tissue', dropna=False).agg(cell_lines=('model_id', 'size'),
        mean_ln_ic50=('LN_IC50', 'mean'), std_ln_ic50=('LN_IC50', 'std')).sort_values(
        'cell_lines', ascending=False).to_csv(out/'tissue_composition.csv')
    cohort.groupby('expression_source').agg(cell_lines=('model_id', 'size'),
        mean_ln_ic50=('LN_IC50', 'mean')).to_csv(out/'expression_sources.csv')
    models = pd.read_csv(a.models)
    counts = models.loc[models.model_id.isin(cohort.model_id)].patient_id.value_counts()
    result = {'drug_rows': len(r), 'unique_response_lines': r.SANGER_MODEL_ID.nunique(),
              'matched_lines': len(cohort), 'replicate_rows': len(r)-r.SANGER_MODEL_ID.nunique(),
              'missing_targets': int(r.LN_IC50.isna().sum()), 'tissue_count': cohort.tissue.nunique(),
              'target_summary': cohort.LN_IC50.describe().to_dict(),
              'model_types': cohort.model_type.value_counts().to_dict(),
              'shared_patient_groups': int((counts > 1).sum()),
              'models_in_shared_patient_groups': int(counts[counts > 1].sum())}
    (out/'cohort_audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
