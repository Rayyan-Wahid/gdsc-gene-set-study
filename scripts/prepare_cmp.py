"""Normalize the downloaded CMP TPM matrix without fitting any preprocessing."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--archive', required=True)
    p.add_argument('--models', required=True)
    p.add_argument('--output', default='data/processed')
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Use an empty output directory.')
    models = pd.read_csv(a.models, dtype={'COSMIC_ID': 'string'})
    if models.model_id.duplicated().any():
        raise ValueError('Duplicate model identifiers.')
    mapped = models.set_index('model_id')
    with zipfile.ZipFile(a.archive) as z:
        candidates = [n for n in z.namelist() if '_rsem_tpm_' in n and n.endswith('.csv')]
        if len(candidates) != 1:
            raise ValueError(f'Expected one TPM matrix, found {candidates}')
        member = candidates[0]
        with z.open(member) as stream:
            reader = csv.reader(io.TextIOWrapper(stream))
            ids, names, sources, annotation = [next(reader) for _ in range(4)]
        if ids[0] != 'model_id' or annotation[:3] != ['gene_symbol', 'ensembl_gene_id', 'gene_id']:
            raise ValueError('Unexpected CMP matrix header.')
        sample_ids = ids[3:]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError('Duplicate matrix model IDs.')
        print('Reading TPM member; ZIP CRC is checked while reading.', flush=True)
        with z.open(member) as stream:
            matrix = pd.read_csv(stream, skiprows=4, header=None,
                                 names=annotation[:3]+sample_ids)
    genes = matrix[annotation[:3]].copy()
    invalid_genes = genes.ensembl_gene_id.isna() | genes.ensembl_gene_id.duplicated(keep=False)
    genes.loc[invalid_genes].to_csv(out/'excluded_gene_annotations.csv', index=False)
    matrix = matrix.loc[~invalid_genes]
    genes = genes.loc[~invalid_genes]
    retained = [s for s in sample_ids if s in mapped.index]
    values = matrix[retained].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError('Infinite expression values.')
    expression = pd.DataFrame(values.T, columns=genes.ensembl_gene_id,
                              index=retained)
    expression.index.name = 'cell_line_id'
    expression.to_csv(out/'expression.csv.gz', compression={'method': 'gzip', 'compresslevel': 1})
    genes.to_csv(out/'gene_annotations.csv', index=False)
    meta = mapped.loc[retained, ['COSMIC_ID', 'model_name', 'tissue', 'cancer_type', 'model_type']].copy()
    meta['expression_source'] = pd.Series(dict(zip(sample_ids, sources[3:])))
    meta.reset_index().to_csv(out/'sample_annotations.csv', index=False)
    manifest = {'archive_name': Path(a.archive).name, 'archive_sha256': sha256(a.archive),
                'models_name': Path(a.models).name, 'models_sha256': sha256(a.models),
                'matrix_member': member, 'matrix_crc_verified': True,
                'expression_scale': 'Supplied CMP log2(TPM+1); no additional transformation',
                'matrix_samples': len(sample_ids), 'mapped_samples': len(retained),
                'unmapped_or_ambiguous_samples': len(sample_ids)-len(retained),
                'genes': len(genes),
                'missing_values': int(np.isnan(values).sum()),
                'excluded_gene_rows': int(invalid_genes.sum()),
                'duplicate_gene_policy': 'exclude all rows with ambiguous Ensembl identifiers',
                'gene_identifier': 'ensembl_gene_id', 'sample_identifier': 'Sanger model_id'}
    (out/'preparation_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
