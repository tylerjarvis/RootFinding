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
# Set NUMBA_OPT in the environment to override. Setting numba's config.OPT from code does not
# survive this, since the cap is applied when yroots is imported, after that assignment would run.
import os as _os
import sys as _sys
_yroots_set_numba_opt = "NUMBA_OPT" not in _os.environ
_os.environ.setdefault("NUMBA_OPT", "2")
if _yroots_set_numba_opt and "numba" in _sys.modules:
    # numba read NUMBA_OPT when it was first imported, which was before the line above, so ask it
    # to read the environment again. reload_config is not part of numba's public API, so a future
    # version may rename it or stop honouring the variable. Either way the cap just does not
    # apply, which costs compile time on a cold cache and nothing else -- far too little to let it
    # take `import yroots` down with it, which is what calling it unguarded used to do.
    try:
        from numba.core import config as _numba_config
        _numba_config.reload_config()
        _yroots_opt_capped = _numba_config.OPT == 2
    except Exception:
        _yroots_opt_capped = False
    if not _yroots_opt_capped:
        import warnings as _warnings
        _warnings.warn(
            "yroots could not cap numba's optimization level at -O2: numba was imported before "
            "yroots and did not re-read NUMBA_OPT. Compiling the solver's higher-dimensional "
            "signatures may take minutes rather than seconds the first time, on a cold numba "
            "cache. To avoid this, import yroots before numba, or set NUMBA_OPT=2 in the "
            "environment before starting python.",
            RuntimeWarning,
        )

from .Combined_Solver import solve
from .ChebyshevApproximator import chebApproximate
from .polynomial import MultiPower
from .polynomial import MultiCheb