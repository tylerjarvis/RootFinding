"""Run the existing test suite with ChebyshevSubdivisionSolver.TRANSFORM_METHOD forced to 'fast'.

Usage (from repo root):
    uv run --no-sync python experiments/fast_transform/e4_solve_1d/run_tests_fast.py [--no-x] [--method dense|fast|auto] [-- extra pytest args]
"""
import sys
import pytest
import yroots.ChebyshevSubdivisionSolver as C
from yroots import FastTransform as F

argv = sys.argv[1:]
extra = argv[argv.index('--') + 1:] if '--' in argv else []
argv = argv[:argv.index('--')] if '--' in argv else argv
C.TRANSFORM_METHOD = argv[argv.index('--method') + 1] if '--method' in argv else 'fast'
F.NUFFT_NTHREADS = 1
args = ['tests', '-q', '-p', 'no:cacheprovider', '-rfE']
if '--no-x' not in argv:
    args.insert(2, '-x')
args += extra
print('TRANSFORM_METHOD =', C.TRANSFORM_METHOD, 'args =', args, flush=True)
code = pytest.main(args)
print('TRANSFORM_METHOD at end =', C.TRANSFORM_METHOD)
sys.exit(code)
