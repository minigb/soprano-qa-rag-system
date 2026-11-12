"""Run the second frozen prompt once the first generation controller has ended."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

if __name__ == '__main__':
    assert (ROOT / 'challenge-k/completion.json').exists(), 'Finish the first sequential capture before starting this GPU job'
    for directory, script in [('expert-l', 'run.py'), ('challenge-l', 'run.py'), ('timing-l', 'profile.py')]:
        run = ROOT / directory
        print('Starting ' + directory, flush=True)
        with (run / 'generation.log').open('a') as log:
            subprocess.run([sys.executable, '-u', str(run / script)], stdout=log, stderr=subprocess.STDOUT, check=True)
        print('Completed ' + directory, flush=True)
