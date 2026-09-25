"""Cases that still break the solver, captured as xfail tests.

These came out of probing ``main`` for inputs the rest of the suite does not cover. Each
is written the way it *should* pass and marked ``xfail(strict=True)``, so the suite stays
green while the defect stands and turns red the moment one is fixed without the test
being updated. The reason string on each mark records what actually happens today.

The polynomial defects found in the same sweep have since been fixed; their regression
tests live in ``test_polynomial.py``. So have the degenerate systems that used to recurse
until python's stack limit (a curve of roots, the zero polynomial, two equations that agree
to rounding error). ``solve`` now raises a ValueError for them, and their regression tests
live in ``test_solve_api.py``.

Two further inputs are left unguarded by choice rather than by oversight, and have no
test here because a test for them would have to hang to prove the point: non-finite
bounds (nan defeats the ``b < a`` check and the approximator then never converges) and
``minBoundingIntervalSize <= 0`` (which removes the re-solve loop's only stopping rule).

Note on what is *not* here: randomized sweeps over well-posed systems (random Chebyshev
systems in 1-3 dimensions, on unit and non-unit boxes, against multi-start Newton ground
truth) found no missed, spurious, duplicated, or out-of-box roots, and no disagreement
between the ``exact``, ``returnBoundingBoxes``, and ``minBoundingIntervalSize`` code
paths.
"""
import numpy as np
import pytest

from yroots.polynomial import MultiCheb


############################### polynomials ##################################

@pytest.mark.xfail(raises=TypeError, strict=True,
                   reason="MultiCheb has no __mul__; only MultiPower defines one")
def test_multiplying_two_cheb_polynomials():
    """``MultiPower * MultiPower`` works, ``MultiCheb * MultiCheb`` raises TypeError.

    Chebyshev multiplication is not the convolution MultiPower uses, so the operator
    cannot simply be inherited -- but the two classes are presented as interchangeable
    representations, and the asymmetry only shows up at the call site.
    """
    c = MultiCheb(np.array([0., 1.]))                     # T_1 = x
    product = c * c                                       # x^2 = (T_0 + T_2)/2
    assert np.allclose(product(np.array([[0.3], [0.7]])), np.array([0.09, 0.49]))
