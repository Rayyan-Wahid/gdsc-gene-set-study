"""Freeze drug choice from coverage alone, before fitting models."""
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from prepare_cmp import sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--responses', required=True, help='Original GDSC2 XLSX or its lossless CSV export')
    p.add_argument('--source-xlsx', required=True, help='Original workbook for provenance hashing')
    p.add_argument('--prepared', default='data/cmp_sanger')
    p.add_argument('--compounds', required=True)
    p.add_argument('--output', default='results/study_design')
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Use an empty output folder; frozen choices must not be overwritten.')
    path = Path(a.responses)
    r = pd.read_excel(path) if path.suffix == '.xlsx' else pd.read_csv(path)
    r = r.rename(columns={'SANGER_MODEL_ID': 'cell_line_id', 'DRUG_ID': 'drug_id', 'DATASET': 'dataset'})
    required = {'cell_line_id', 'drug_id', 'dataset', 'LN_IC50', 'DRUG_NAME'}
    if not required.issubset(r.columns):
        raise ValueError('Unexpected response schema.')
    samples = pd.read_csv(Path(a.prepared)/'sample_annotations.csv')
    valid = r.loc[(r.dataset == 'GDSC2') & r.cell_line_id.isin(samples.model_id) & np.isfinite(r.LN_IC50)]
    counts = valid.groupby('drug_id').cell_line_id.nunique().rename('matched_cell_lines').reset_index()
    counts = counts.sort_values(['matched_cell_lines', 'drug_id'], ascending=[False, True])
    names = r[['drug_id', 'DRUG_NAME']].drop_duplicates()
    if names.drug_id.duplicated().any():
        raise ValueError('Drug ID has multiple names.')
    counts = counts.merge(names, on='drug_id', validate='one_to_one')
    counts.to_csv(out/'drug_coverage.csv', index=False)
    chosen = int(counts.iloc[0].drug_id)
    r[['cell_line_id', 'drug_id', 'dataset', 'LN_IC50']].to_csv(Path(a.prepared)/'responses.csv', index=False)
    compounds = pd.read_csv(a.compounds)
    details = compounds.loc[compounds.DRUG_ID == chosen].fillna('').to_dict('records')
    manifest = {'dataset': 'GDSC2', 'response_release': '27Oct23', 'drug_id': chosen,
                'drug_name': counts.iloc[0].DRUG_NAME, 'matched_cell_lines': int(counts.iloc[0].matched_cell_lines),
                'selection_rule': 'Maximum unique matched cell lines with finite response; ties use smallest numeric drug ID',
                'selected_before_model_fitting': True, 'source_xlsx_sha256': sha256(a.source_xlsx),
                'compounds_sha256': sha256(a.compounds), 'compound_annotations': details,
                'expression_preparation': json.loads((Path(a.prepared)/'preparation_manifest.json').read_text())}
    (out/'selection.json').write_text(json.dumps(manifest, indent=2))
    print(counts.head(10).to_string(index=False), flush=True)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
