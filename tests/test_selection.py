import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import pandas as pd


class SelectionTests(unittest.TestCase):
    def test_coverage_counts_unique_lines_and_breaks_numeric_ties(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pd.DataFrame({'model_id': ['S1', 'S2']}).to_csv(root/'sample_annotations.csv', index=False)
            (root/'preparation_manifest.json').write_text('{}')
            response = root/'response.csv'
            pd.DataFrame({'SANGER_MODEL_ID': ['S1','S2','S1','S2','S1','S3'],
                          'DRUG_ID': [2,2,10,10,10,10], 'DATASET': ['GDSC2']*6,
                          'LN_IC50': [100,-100,0,0,0,0],
                          'DRUG_NAME': ['A','A','B','B','B','B']}).to_csv(response, index=False)
            compounds = root/'compounds.csv'
            pd.DataFrame({'DRUG_ID':[2,10], 'DRUG_NAME':['A','B']}).to_csv(compounds,index=False)
            subprocess.run([sys.executable,'scripts/select_drug.py','--responses',str(response),
                            '--source-xlsx',str(response),'--prepared',str(root),
                            '--compounds',str(compounds),'--output',str(root/'design')],
                           check=True,capture_output=True,text=True)
            result=json.loads((root/'design/selection.json').read_text())
            self.assertEqual(result['drug_id'],2)
            self.assertEqual(result['matched_cell_lines'],2)
            self.assertTrue(result['selected_before_model_fitting'])
