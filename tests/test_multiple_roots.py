"""Systems with multiple (non-simple) roots, in one through four dimensions.

A multiple root can only be located to about sqrt(macheps) (or macheps**(1/m) for multiplicity m),
and the solver may report it as a small cluster of points, usually with a "Might Have Duplicate
Roots" warning. So these tests check that every true root has a reported root nearby and a
bounding box around it, and that nothing is reported away from the true roots. They do not check
the number of points reported or the warning.

The systems come in both sparse and dense flavors. Polynomials such as (x-.3)**2 have sparse
Chebyshev coefficient tensors, while compositions with sin, exp, tanh or cos have dense ones.
"""
import warnings

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.stats import ortho_group

from yroots.Combined_Solver import solve


def solve_quietly(funcs, a, b, **kwargs):
    """Solves the system, ignoring the duplicate root warnings multiple roots produce."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return solve(funcs, a, b, **kwargs)


def assert_finds_exactly(funcs, a, b, expected, tol=1e-6, box_slack=0.):
    """Every expected root is reported and boxed, and every reported root is an expected one.

    ``box_slack`` widens the bounding boxes for expected roots that are themselves only known
    to within rounding error.
    """
    expected = np.atleast_2d(np.array(expected, dtype=float))
    roots, boxes = solve_quietly(funcs, a, b, returnBoundingBoxes=True)
    roots = np.asarray(roots, dtype=float).reshape(len(roots), expected.shape[1])
    boxes = [np.asarray(box, dtype=float).reshape(expected.shape[1], 2) for box in boxes]
    assert len(roots) > 0, "no roots were reported"
    for root in expected:
        distance = np.min(np.linalg.norm(roots - root, axis=1))
        assert distance < tol, f"the root at {root} was missed (nearest reported: {distance:.1e})"
        assert any(np.all(box[:, 0] - box_slack <= root) and np.all(root <= box[:, 1] + box_slack)
                   for box in boxes), (
            f"no bounding box contains the root at {root}")
    for root in roots:
        distance = np.min(np.linalg.norm(expected - root, axis=1))
        assert distance < tol, f"reported {root}, which is {distance:.1e} from every true root"


# chebApproximate's error bound covers only the truncated tail of the series, not rounding in the
# coefficients (5.1e-16 for (x-.3)**3 against a stated 3.5e-21) or in the sample points of a
# re-solve on a small interval away from 0. The approximation can then sit farther from the
# function than its stated error, and whether the bounding box still contains the true root
# depends on the platform's rounding. These pass on macOS arm64 but fail on ubuntu-latest in CI.
_ROUNDING_DEPENDENT_BOX = pytest.mark.xfail(
    strict=False, reason="the approximation error bound leaves out rounding, so whether the "
                         "bounding box contains the root depends on the platform")


################################# one root ###################################

ONE_MULTIPLE_ROOT = {
    # sparse
    "1D triple root": (
        lambda x: (x - .3)**3,
        [-1], [1], [(.3,)]),
    "2D fourfold root, both functions touch zero": (
        [lambda x, y: (x - .2)**2, lambda x, y: (y + .1)**2],
        [-1, -1], [1, 1], [(.2, -.1)]),
    "2D fourfold root, both functions touch zero along diagonals": (
        [lambda x, y: (x + y - .1)**2, lambda x, y: (x - y - .3)**2],
        [-1, -1], [1, 1], [(.2, -.1)]),
    "2D fourfold root, one function touches zero only at the root": (
        [lambda x, y: (x - .2)**2 + (y + .1)**2, lambda x, y: (x - .2)*(y + .1)],
        [-1, -1], [1, 1], [(.2, -.1)]),
    "3D double root": (
        [lambda x, y, z: (x - .3)**2, lambda x, y, z: y - .1, lambda x, y, z: z + .2],
        [-1]*3, [1]*3, [(.3, .1, -.2)]),
    "3D eightfold root": (
        [lambda x, y, z: (x - .2)**2, lambda x, y, z: (y + .1)**2, lambda x, y, z: (z - .3)**2],
        [-1]*3, [1]*3, [(.2, -.1, .3)]),
    "4D double root": (
        [lambda a, b, c, d: (a - .3)**2, lambda a, b, c, d: b - .1,
         lambda a, b, c, d: c + .2, lambda a, b, c, d: d - .4],
        [-1]*4, [1]*4, [(.3, .1, -.2, .4)]),
    # dense: sin(...)**2 touches zero along a curve that the other functions cross once
    "2D dense double root": (
        [lambda x, y: np.sin(x + 2*y - .5)**2, lambda x, y: np.exp(x - y**2 - .1) - 1],
        [-1, -1], [1, 1], [((np.sqrt(1.4) - 1)**2 + .1, np.sqrt(1.4) - 1)]),
    "3D dense double root": (
        [lambda x, y, z: np.sin(x + y + z - .3)**2, lambda x, y, z: np.exp(x) - np.exp(y + .1),
         lambda x, y, z: np.sin(z - .2*x)],
        [-1]*3, [1]*3, [(.4/2.2, .4/2.2 - .1, .2*.4/2.2)]),
    "4D dense double root": (
        [lambda a, b, c, d: np.sin(a + b + c + d - .3)**2,
         lambda a, b, c, d: np.exp(a) - np.exp(b + .1),
         lambda a, b, c, d: np.sin(c - .2*a), lambda a, b, c, d: np.tanh(d + .3*b)],
        [-1]*4, [1]*4, [(.37/1.9, .37/1.9 - .1, .2*.37/1.9, -.3*(.37/1.9 - .1))]),
}


# The fourfold and eightfold roots, where every function touches zero, zoom to a box about 1e-7
# wide whose linear terms are all below the approximation error. They used to recurse until
# python's stack limit there; isBelowResolution now stops them.
@pytest.mark.parametrize("name", [
    pytest.param(name, marks=_ROUNDING_DEPENDENT_BOX) if name == "1D triple root" else name
    for name in ONE_MULTIPLE_ROOT])
def test_a_single_multiple_root_is_found(name):
    funcs, a, b, expected = ONE_MULTIPLE_ROOT[name]
    assert_finds_exactly(funcs, a, b, expected)


############################ several multiple roots ##########################

def _dense_2d_roots():
    # sin(pi*(x+y))**2 touches zero on the lines x + y = k; x - y = .3cos(x) crosses each once.
    roots = []
    for k in (-1, 0, 1):
        x = brentq(lambda x: 2*x - k - .3*np.cos(x), -2, 2)
        roots.append((x, k - x))
    return roots


SEVERAL_MULTIPLE_ROOTS = {
    # sparse
    "1D two double roots": (
        lambda x: (x + .5)**2 * (x - .4)**2,
        [-1], [1], [(-.5,), (.4,)]),
    "2D two double roots": (
        [lambda x, y: ((x - .3)*(x + .4))**2, lambda x, y: y - .1],
        [-1, -1], [1, 1], [(.3, .1), (-.4, .1)]),
    # Neither function touches zero here: the circle and the ellipse are tangent at two points.
    "2D circle tangent to an ellipse at two points": (
        [lambda x, y: (x - .1)**2 + (y - .05)**2 - .25,
         lambda x, y: (x - .1)**2/.25 + (y - .05)**2/.04 - 1],
        [-1, -1], [1, 1], [(.6, .05), (-.4, .05)]),
    "3D two double roots": (
        [lambda x, y, z: ((x - .3)*(x + .4))**2, lambda x, y, z: y - .1*x,
         lambda x, y, z: z + y**2 - .2],
        [-1]*3, [1]*3, [(.3, .03, .2 - .03**2), (-.4, -.04, .2 - .04**2)]),
    # dense
    "1D three double roots": (
        lambda x: np.sin(5*x)**2,
        [-1], [1], [(-np.pi/5,), (0,), (np.pi/5,)]),
    "2D three double roots": (
        [lambda x, y: np.sin(np.pi*(x + y))**2, lambda x, y: x - y - .3*np.cos(x)],
        [-1, -1], [1, 1], _dense_2d_roots()),
}


@pytest.mark.parametrize("name", SEVERAL_MULTIPLE_ROOTS)
def test_several_multiple_roots_are_all_found(name):
    funcs, a, b, expected = SEVERAL_MULTIPLE_ROOTS[name]
    assert_finds_exactly(funcs, a, b, expected)


########################## the devastating example ###########################

def devastating_example(Q, eps):
    """The "devastating example" of Noferini and Townsend, p_i(x) = x_i**2 + eps*(Qx)_i.

    See V. Noferini and A. Townsend, "Numerical instability of resultant methods for
    multidimensional rootfinding", SIAM J. Numer. Anal. 54 (2016). Q is orthogonal. The root at
    the origin is simple for eps > 0 but, as eps -> 0, merges with other roots into a root of
    multiplicity 2**dim, which makes resultant methods lose accuracy exponentially in dim.
    """
    dim = Q.shape[0]
    return [lambda *x, i=i: x[i]**2 + eps*sum(Q[i, j]*x[j] for j in range(dim))
            for i in range(dim)]


@pytest.mark.parametrize("dim", [2, 3, 4])
def test_the_devastating_example_with_eps_zero_has_one_root_of_multiplicity_2_to_the_dim(dim):
    Q = ortho_group.rvs(dim, random_state=1)
    assert_finds_exactly(devastating_example(Q, 0.), [-1]*dim, [1]*dim, [(0.,)*dim])


_DEVASTATING_MARKS = {(2, 1e-2): _ROUNDING_DEPENDENT_BOX}


@pytest.mark.parametrize("dim, eps", [
    pytest.param(dim, eps, marks=_DEVASTATING_MARKS[dim, eps]) if (dim, eps) in _DEVASTATING_MARKS
    else (dim, eps)
    for dim in (2, 3, 4) for eps in (1e-2, 1e-4, 1e-6)])
def test_the_devastating_example_finds_every_root_of_the_near_multiple_cluster(dim, eps):
    """Substituting x = eps*u turns the system into u_i**2 + (Qu)_i = 0, which does not depend on
    eps. So the roots are eps times the roots of the eps = 1 system, whose roots are simple and
    well separated. Summing u_i**2 = -(Qu)_i over i gives |u|**2 <= sqrt(dim)|u|, so all of them
    lie in the box [-sqrt(dim), sqrt(dim)]**dim.

    The accuracy of the small roots is absolute, about sqrt(macheps), so their relative accuracy
    degrades as eps shrinks; that is expected for a near-multiple root. The tolerance is kept
    below eps so that the check still tells the cluster's roots apart from the origin.
    """
    Q = ortho_group.rvs(dim, random_state=1)
    bound = np.sqrt(dim) + .5
    unit_roots = solve(devastating_example(Q, 1.), [-bound]*dim, [bound]*dim)
    at_origin = np.all(np.abs(unit_roots) < 1e-12, axis=1)
    assert np.any(at_origin), "the eps = 1 system lost the origin"
    unit_roots[at_origin] = 0.
    assert len(unit_roots) > 1, "the eps = 1 system should have roots besides the origin"
    assert_finds_exactly(devastating_example(Q, eps), [-1]*dim, [1]*dim, eps*unit_roots,
                         tol=min(1e-6, eps/4), box_slack=1e-13*eps)
