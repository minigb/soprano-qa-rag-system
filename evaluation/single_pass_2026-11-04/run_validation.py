"""Run the final GPU captures sequentially; never overlap inference workloads."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    for directory, script in (('single-pass-g-clean', 'profile.py'),
                              ('expert-g', 'run.py'), ('challenge-g', 'run.py')):
        root = ROOT / directory
        print('Starting ' + directory, flush=True)
        with (root / 'generation.log').open('a') as log:
            subprocess.run([sys.executable, '-u', str(root / script)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        print('Completed ' + directory, flush=True)


if __name__ == '__main__':
    main()
