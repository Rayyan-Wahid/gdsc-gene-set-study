"""Run the frozen real-data study; limits nested thread oversubscription."""
import os
for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[variable] = '1'
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from gene_study.data import prepare, read_table
from gene_study.experiment import Config, evaluate
from gene_study.cli import digest


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', default='data/cmp_sanger')
    p.add_argument('--design', default='results/study_design/selection.json')
    p.add_argument('--output', default='results/full_study')
    p.add_argument('--jobs', type=int, default=2)
    a = p.parse_args()
    frozen = json.loads(Path(a.design).read_text())
    root = Path(a.prepared)
    print('Loading expression and matching the frozen drug choice.', flush=True)
    X, y, audit = prepare(read_table(root/'expression.csv.gz'), read_table(root/'responses.csv'),
                          frozen['drug_id'], frozen['dataset'])
    audit.update({'synthetic': False, 'frozen_selection': frozen,
                  'expression_sha256': digest(root/'expression.csv.gz'),
                  'responses_sha256': digest(root/'responses.csv')})
    print(f'{len(X)} cell lines; {X.shape[1]} input genes', flush=True)
    result = evaluate(X, y, a.output, Config(n_jobs=a.jobs, cache_dir='outputs/training_cache'), audit)
    print(result.to_string(index=False), flush=True)
