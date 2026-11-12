"""Serialize all GPU generation captures; frozen outputs are resumable."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

if __name__ == '__main__':
    for directory, script in [('expert-h', 'run.py'), ('challenge-h', 'run.py'),
                              ('timing-g', 'profile.py'), ('timing-h', 'profile.py')]:
        run = ROOT / directory
        print('Starting ' + directory, flush=True)
        with (run / 'generation.log').open('a') as log:
            subprocess.run([sys.executable, '-u', str(run / script)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        print('Completed ' + directory, flush=True)
