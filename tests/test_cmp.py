import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

import pandas as pd


class CMPPreparationTests(unittest.TestCase):
    def test_mapping_transposition_and_ambiguous_gene_exclusion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            buffer = io.StringIO()
            writer = csv.writer(buffer)
            writer.writerows([
                ['model_id', '', '', 'SIDM1', 'SIDM2', 'SIDM3'],
                ['model_name', '', '', 'A', 'B', 'C'],
                ['data_source', '', '', 'Sanger', 'Broad', 'Sanger'],
                ['gene_symbol', 'ensembl_gene_id', 'gene_id', '', '', ''],
                ['A', 'ENS1', 'G1', 1, 2, 3],
                ['B', 'ENS2', 'G2', 4, 5, 6],
                ['C', 'ENS2', 'G3', 7, 8, 9],
            ])
            archive = root/'fixture.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('rnaseq_merged_rsem_tpm_fixture.csv', buffer.getvalue())
            models = root/'models.csv'
            pd.DataFrame({'model_id': ['SIDM1', 'SIDM2'], 'COSMIC_ID': [101, 102],
                          'model_name': ['A', 'B'], 'tissue': ['Lung', 'Skin'],
                          'cancer_type': ['A', 'B'], 'model_type': ['Cell Line']*2}).to_csv(models, index=False)
            subprocess.run([sys.executable, 'scripts/prepare_cmp.py', '--archive', str(archive),
                            '--models', str(models), '--output', str(root/'out')], check=True,
                           capture_output=True, text=True)
            expression = pd.read_csv(root/'out/expression.csv.gz')
            self.assertEqual(expression.columns.tolist(), ['cell_line_id', 'ENS1'])
            self.assertEqual(expression.ENS1.tolist(), [1, 2])
            self.assertEqual(expression.cell_line_id.tolist(), ['SIDM1', 'SIDM2'])
            manifest = json.loads((root/'out/preparation_manifest.json').read_text())
            self.assertEqual(manifest['excluded_gene_rows'], 2)
            self.assertEqual(manifest['unmapped_or_ambiguous_samples'], 1)
            self.assertTrue(manifest['matrix_crc_verified'])
