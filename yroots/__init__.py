# Do not delete this file. It tells python that yroots is a module you can import from.
# Public-facing functions should be imported here so they can be used directly.
# YRoots

name = "yroots"

# Cap numba's LLVM optimization level at 2 before numba is imported below.
# At -O3 (numba's default) LLVM's SimpleLoopUnswitch pass blows up on
# TransformChebInPlace1D for high-dimensional coefficient arrays: compiling the
# 5-D signature takes ~2 minutes at -O3 versus ~2 seconds at -O2. Generated code
# is unaffected -- roots come out bit-identical and solve times are unchanged --
# so the only thing -O3 buys here is a five-minute wait on a cold numba cache.
# Set NUMBA_OPT in the environment to override.
import os as _os
_os.environ.setdefault("NUMBA_OPT", "2")
import sys as _sys
if "numba" in _sys.modules:  # numba imported first; make it re-read the env
    from numba.core import config as _numba_config
    _numba_config.reload_config()

from .Combined_Solver import solve
from .ChebyshevApproximator import chebApproximate
from .polynomial import MultiPower
from .polynomial import MultiCheb