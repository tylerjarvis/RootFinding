"""Cases that still break the solver, captured as xfail tests.

These came out of probing ``main`` for inputs the rest of the suite does not cover. Each
is written the way it *should* pass and marked ``xfail(strict=True)``, so the suite stays
green while the defect stands and turns red the moment one is fixed without the test
being updated. The reason string on each mark records what actually happens today.

The polynomial defects found in the same sweep have since been fixed; their regression
tests live in ``test_polynomial.py``. Most of what is left has one root cause:
``solvePolyRecursive`` has no depth cap, so a system it cannot separate recurses until
python's stack limit instead of returning a result or reporting the problem.

Two further inputs are left unguarded by choice rather than by oversight, and have no
test here because a test for them would have to hang to prove the point: non-finite
bounds (nan defeats the ``b < a`` check and the approximator then never converges) and
``minBoundingIntervalSize <= 0`` (which removes the re-solve loop's only stopping rule).

Note on what is *not* here: randomized sweeps over well-posed systems (random Chebyshev
systems in 1-3 dimensions, on unit and non-unit boxes, against multi-start Newton ground
truth) found no missed, spurious, duplicated, or out-of-box roots, and no disagreement
between the ``exact``, ``returnBoundingBoxes``, and ``minBoundingIntervalSize`` code
paths. The failures below are all at the degenerate edge.
"""
import numpy as np
import pytest

import yroots as yr
from yroots.polynomial import MultiCheb


############################### solver: degenerate systems ###################

@pytest.mark.xfail(raises=RecursionError, strict=True,
                   reason="a system this ill conditioned recurses until python's stack "
                          "limit instead of returning its root or reporting the problem")
def test_ill_conditioned_system_keeps_its_root_at_1e_10():
    """The same two nearly parallel lines as ``test_ill_conditioned_system_keeps_its_root``.

    That test parametrizes eps down to 1e-7. The root at (0.3, 0) survives to 1e-9; from
    3e-10 down (0 included) the solver never stops subdividing and dies with a
    RecursionError several seconds in.

    The root is still recoverable in principle: it sits at (0.3, 0) exactly, and
    machine-precision perturbations of the coefficients only move it by about u/eps, or
    2e-6 here. Whether to return it or to fail cleanly is a design decision; the stack
    overflow is neither.
    """
    eps = 1e-10
    f = lambda x, y: x + y - 0.3
    g = lambda x, y: x + (1 + eps) * y - 0.3
    roots = yr.solve([f, g], [-1, -1], [1, 1])

    assert len(roots) == 1
    assert np.allclose(roots[0], [0.3, 0.0], atol=1e-6)


@pytest.mark.xfail(raises=RecursionError, strict=True,
                   reason="a system with a curve of solutions recurses until python's "
                          "stack limit instead of raising")
def test_a_system_with_infinitely_many_roots_reports_the_problem():
    """Two copies of the same equation: every point on a line solves the system.

    ``solve`` documents that an infinite root set may get the solver "stuck in recursion",
    but a duplicated equation is an easy mistake to make and the result is a bare
    RecursionError from deep inside the solver, with nothing pointing at the input.
    """
    f = lambda x, y: x + y - 0.3
    with pytest.raises(ValueError):
        yr.solve([f, f], [-1, -1], [1, 1])


@pytest.mark.xfail(raises=RecursionError, strict=True,
                   reason="the identically zero polynomial recurses until python's stack limit")
def test_the_zero_polynomial_reports_the_problem():
    """``MultiCheb(np.zeros(3))`` is zero everywhere, so every point is a root.

    Same stack overflow as above, from an input that is trivially recognizable: the
    coefficient tensor is all zeros before any approximation work begins.
    """
    with pytest.raises(ValueError):
        yr.solve(MultiCheb(np.zeros(3)), -1, 1)


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
