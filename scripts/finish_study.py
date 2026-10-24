"""Run the canonical study and generate outputs, with persistent progress logs."""
import json
from pathlib import Path
import subprocess
import sys
import traceback


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    status = root/'outputs/study_status.json'
    def record(stage, state):
        status.write_text(json.dumps({'stage': stage, 'state': state}, indent=2))
    steps = [
        ('training', ['scripts/run_study.py', '--jobs', '2', '--output', 'results/mg132_complete', '--resume']),
        ('report', ['scripts/report_study.py', '--run', 'results/mg132_complete']),
        ('notebook', ['scripts/build_notebook.py']),
    ]
    try:
        for stage, command in steps:
            record(stage, 'running')
            subprocess.run([sys.executable, '-u', *command], cwd=root, check=True)
        record('outputs', 'complete')
    except Exception:
        record(stage, 'failed')
        traceback.print_exc()
        raise
