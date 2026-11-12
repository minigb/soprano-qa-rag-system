"""Finish offline judging, fixed metrics and separate serial timing in order."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run(command, log_path, *, env=None):
    with log_path.open('a') as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    for name in ['expert', 'challenge']:
        run([sys.executable, '-u', str(ROOT/name/'watch_judge.py')], ROOT/name/'judge.log')
        print(name + ' judging complete', flush=True)
        if name == 'expert':
            run(['/home/minhee/.cache/soprano-rag-eval-venv/bin/python', '-u', str(ROOT/name/'score.py')],
                ROOT/name/'score.log', env={**os.environ, 'HF_HUB_OFFLINE': '1'})
            print('Fixed BERTScore complete', flush=True)
    run([sys.executable, '-u', str(ROOT/'timing_sequential.py')], ROOT/'timing.log')
    print('Separate serial timing complete', flush=True)
    run([sys.executable, str(ROOT/'analyze.py')], ROOT/'analysis.log')
    print('Analysis complete', flush=True)


if __name__ == '__main__':
    main()
