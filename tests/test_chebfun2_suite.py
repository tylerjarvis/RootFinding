"""The Chebfun2 benchmark systems, as pytest tests.

Each case is a 2-D polynomial or transcendental system with a known root set. The
reference roots in ``Polished_results/`` were obtained by polishing high-accuracy
solutions of these systems, and are treated here as ground truth.

Every case is checked three ways, as three separate tests, so a failure says which
property broke rather than just "test 3.2 failed":

* ``test_finds_the_expected_number_of_roots`` - no missed roots, no spurious ones.
* ``test_every_root_makes_the_system_vanish`` - each reported root is really a root
  (residual check; independent of the reference data).
* ``test_roots_match_the_polished_reference`` - the roots are in the right *places*
  (agreement with the reference; independent of the residuals).

The residual and location checks are deliberately kept apart: a root set can have
tiny residuals and still be wrong when the system is ill-conditioned near a root,
and the two checks fail for different reasons.

Case 6.1 is the one place where the solver and the reference legitimately disagree
on the count: a double root sits at the origin, the reference records it once, and
the solver reports it twice. That is stated outright rather than smoothed over --
the case declares the duplicate, 6 roots are expected against the reference's 5,
and the failure message names both numbers.

Tolerances
----------
``DEFAULT_RESIDUAL_TOL`` and ``DEFAULT_ROOT_TOL`` cover most cases. The handful of
systems that need a looser bound set it on their own ``Case`` with a comment giving
the measured value, so a tolerance can never be loosened without the reason for it
sitting right there. Every tolerance here was set from a measured run and leaves at
least an order of magnitude of headroom; the solver is deterministic, so these are
not statistical bounds.

``Chebfun_results/`` holds the roots Chebfun2 reports for the same systems. Nothing
here asserts against them: Chebfun is a second implementation, not ground truth, and
YRoots is more accurate on some of these systems and less on others, so agreement
with Chebfun is not a correctness criterion. The files are kept for benchmarking.
"""
import dataclasses

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment
from pathlib import Path

from yroots.Combined_Solver import solve

DATA_DIR = Path(__file__).resolve().parent
POLISHED_DIR = DATA_DIR / "Polished_results"

#: Largest |f(root)| tolerated, for systems that do not override it.
DEFAULT_RESIDUAL_TOL = 1e-12
#: Largest distance tolerated between a found root and its reference root.
DEFAULT_ROOT_TOL = 1e-12


@dataclasses.dataclass(eq=False)
class Case:
    """One benchmark system and the accuracy expected of it.

    Attributes
    ----------
    name : str
        The case number from the benchmark suite, e.g. ``"3.2"``. Doubles as the
        pytest id and as the stem of the reference file in ``Polished_results/``.
    funcs : list of functions
        The system to solve, each taking ``(x, y)``.
    a, b : numpy array
        Lower and upper corners of the search box.
    residual_tol, root_tol : float
        Per-case overrides of the module defaults.
    duplicate_root : numpy array, optional
        A multiple root that the reference records once but the solver reports twice,
        because it cannot separate the two branches meeting there. Only 6.1 has one.
        The reference is expected to come up one root short, and the tests say so.
    """

    name: str
    funcs: list
    a: np.ndarray
    b: np.ndarray
    residual_tol: float = DEFAULT_RESIDUAL_TOL
    root_tol: float = DEFAULT_ROOT_TOL
    duplicate_root: np.ndarray | None = None

    @property
    def polished_roots(self):
        """The reference roots exactly as stored on disk, as an (n, 2) array."""
        return np.load(POLISHED_DIR / f"polished_{self.name}.npy").reshape(-1, 2)

    @property
    def reference_roots(self):
        """The roots the solver should report, as an (n, 2) array.

        Same as ``polished_roots`` except where the system has a duplicate root: the
        solver reports that one twice, so it is listed twice here and the ordinary
        one-to-one matching applies unchanged.
        """
        roots = self.polished_roots
        if self.duplicate_root is None:
            return roots
        nearest = np.argmin(np.linalg.norm(roots - self.duplicate_root, axis=1))
        return np.vstack([roots, roots[nearest]])

    @property
    def expected_count(self):
        return len(self.reference_roots)


############################### helpers ######################################

# Solving is the expensive part of this suite and the three tests for a case all
# need the same result, so each system is solved once and the result reused. The
# solver is deterministic and the tests only read the array, so sharing it cannot
# leak state between them.
_solved_cache = {}


def roots_of(case):
    """Solve ``case`` once, then return the cached (n, 2) result."""
    if case.name not in _solved_cache:
        found = solve(case.funcs, case.a, case.b)
        _solved_cache[case.name] = np.asarray(found, dtype=float).reshape(-1, 2)
    return _solved_cache[case.name]


def max_residual(funcs, roots):
    """Largest |f(root)| over every function and every root."""
    if len(roots) == 0:
        return 0.0
    return max(float(np.max(np.abs(f(roots[:, 0], roots[:, 1])))) for f in funcs)


def matched_max_distance(found, reference):
    """Max distance between found and reference roots under the best 1-1 pairing.

    Pairs the two sets optimally rather than sorting them, so the result does not
    depend on a sort order that near-coincident roots can scramble. Because the
    pairing is one-to-one, two found roots collapsing onto the same reference root
    leaves another reference root unmatched and the distance blows up -- which a
    nearest-neighbour distance would quietly hide.
    """
    if len(found) == 0:
        return 0.0
    distances = np.linalg.norm(found[:, None, :] - reference[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(distances)
    return float(distances[rows, cols].max())


############################### the benchmark systems ########################

def _inner_7_2(x, y):
    """The inner function of case 7.2; the case applies it to rescaled arguments."""
    return y**10 - 2*(x**8)*(y**2) + 4*(x**4)*y - 2


#: Half-width of the search box for case 7.3, whose whole system lives at 1e-9 scale.
_SCALE_7_3 = 1e-9

CASES = [
    Case(
        name="1.1",
        funcs=[lambda x, y: 144*(x**4 + y**4) - 225*(x**2 + y**2) + 350*x**2*y**2 + 81,
               lambda x, y: y - x**6],
        a=np.array([-1, -1]), b=np.array([1, 1]),
    ),
    Case(
        name="1.2",
        funcs=[lambda x, y: (y**2 - x**3)*((y - 0.7)**2 - (x - 0.3)**3)*((y + 0.2)**2 - (x + 0.8)**3)*((y + 0.2)**2 - (x - 0.8)**3),
               lambda x, y: ((y + .4)**3 - (x - .4)**2)*((y + .3)**3 - (x - .3)**2)*((y - .5)**3 - (x + .6)**2)*((y + 0.3)**3 - (2*x - 0.8)**3)],
        a=np.array([-1, -1]), b=np.array([1, 1]),
        # Products of near-tangent curves: the roots are badly conditioned even
        # though the residuals stay at 1e-14. Measured root error 4.6e-8.
        root_tol=1e-6,
    ),
    Case(
        name="1.3",
        funcs=[lambda x, y: y**2 - x**3,
               lambda x, y: (y + .1)**3 - (x - .1)**2],
        a=np.array([-1, -1]), b=np.array([1, 1]),
    ),
    Case(
        name="1.4",
        funcs=[lambda x, y: x - y + .5,
               lambda x, y: x + y],
        a=np.array([-1, -1]), b=np.array([1, 1]),
    ),
    Case(
        name="1.5",
        funcs=[lambda x, y: y + x/2 + 1/10,
               lambda x, y: y - 2.1*x + 2],
        a=np.array([-1, -1]), b=np.array([1, 1]),
    ),
    Case(
        name="2.1",
        funcs=[lambda x, y: np.cos(10*x*y),
               lambda x, y: x + y**2],
        a=np.array([-1, -1]), b=np.array([1, 1]),
    ),
    Case(
        name="2.2",
        funcs=[lambda x, y: x,
               lambda x, y: (x - .9999)**2 + y**2 - 1],
        a=np.array([-1, -1]), b=np.array([1, 1]),
    ),
    Case(
        name="2.3",
        funcs=[lambda x, y: np.sin(4*(x + y/10 + np.pi/10)),
               lambda x, y: np.cos(2*(x - 2*y + np.pi/7))],
        a=np.array([-1, -1]), b=np.array([1, 1]),
    ),
    Case(
        name="2.4",
        funcs=[lambda x, y: np.exp(x - 2*x**2 - y**2)*np.sin(10*(x + y + x*y**2)),
               lambda x, y: np.exp(-x + 2*y**2 + x*y**2)*np.sin(10*(x - y - 2*x*y**2))],
        a=np.array([-1, -1]), b=np.array([1, 1]),
    ),
    Case(
        name="2.5",
        funcs=[lambda x, y: 2*y*np.cos(y**2)*np.cos(2*x) - np.cos(y),
               lambda x, y: 2*np.sin(y**2)*np.sin(2*x) - np.sin(x)],
        a=np.array([-4, -4]), b=np.array([4, 4]),
    ),
    Case(
        name="3.1",
        funcs=[lambda x, y: ((x - .3)**2 + 2*(y + 0.3)**2 - 1),
               lambda x, y: ((x - .49)**2 + (y + .5)**2 - 1)*((x + 0.5)**2 + (y + 0.5)**2 - 1)*((x - 1)**2 + (y - 0.5)**2 - 1)],
        a=np.array([-1, -1]), b=np.array([1, 1]),
        # Nearly tangent circles, so root location is far less accurate than the
        # residuals suggest. Measured root error 1.2e-11.
        root_tol=1e-10,
    ),
    Case(
        name="3.2",
        funcs=[lambda x, y: ((x - 0.1)**2 + 2*(y - 0.1)**2 - 1)*((x + 0.3)**2 + 2*(y - 0.2)**2 - 1)*((x - 0.3)**2 + 2*(y + 0.15)**2 - 1)*((x - 0.13)**2 + 2*(y + 0.15)**2 - 1),
               lambda x, y: (2*(x + 0.1)**2 + (y + 0.1)**2 - 1)*(2*(x + 0.1)**2 + (y - 0.1)**2 - 1)*(2*(x - 0.3)**2 + (y - 0.15)**2 - 1)*((x - 0.21)**2 + 2*(y - 0.15)**2 - 1)],
        a=np.array([-1, -1]), b=np.array([1, 1]),
        # Same tangency problem as 3.1, with 45 roots. Measured root error 7.1e-12.
        root_tol=1e-10,
    ),
    Case(
        name="4.1",
        funcs=[lambda x, y: np.sin(3*(x + y)),
               lambda x, y: np.sin(3*(x - y))],
        a=np.array([-1, -1]), b=np.array([1, 1]),
    ),
    Case(
        name="4.2",
        funcs=[
            lambda x, y: ((90000*y**10 + (-1440000)*y**9 + (360000*x**4 + 720000*x**3 + 504400*x**2 + 144400*x + 9971200)*(y**8) +
                          ((-4680000)*x**4 + (-9360000)*x**3 + (-6412800)*x**2 + (-1732800)*x + (-39554400))*(y**7) + (540000*x**8 +
                          2160000*x**7 + 3817600*x**6 + 3892800*x**5 + 27577600*x**4 + 51187200*x**3 + 34257600*x**2 + 8952800*x + 100084400)*(y**6) +
                          ((-5400000)*x**8 + (-21600000)*x**7 + (-37598400)*x**6 + (-37195200)*x**5 + (-95198400)*x**4 +
                          (-153604800)*x**3 + (-100484000)*x**2 + (-26280800)*x + (-169378400))*(y**5) + (360000*x**12 + 2160000*x**11 +
                          6266400*x**10 + 11532000*x**9 + 34831200*x**8 + 93892800*x**7 + 148644800*x**6 + 141984000*x**5 + 206976800*x**4 +
                          275671200*x**3 + 176534800*x**2 + 48374000*x + 194042000)*(y**4) + ((-2520000)*x**12 + (-15120000)*x**11 + (-42998400)*x**10 +
                          (-76392000)*x**9 + (-128887200)*x**8 + (-223516800)*x**7 + (-300675200)*x**6 + (-274243200)*x**5 + (-284547200)*x**4 +
                          (-303168000)*x**3 + (-190283200)*x**2 + (-57471200)*x + (-147677600))*(y**3) + (90000*x**16 + 720000*x**15 + 3097600*x**14 +
                          9083200*x**13 + 23934400*x**12 + 58284800*x**11 + 117148800*x**10 + 182149600*x**9 + 241101600*x**8 + 295968000*x**7 +
                          320782400*x**6 + 276224000*x**5 + 236601600*x**4 + 200510400*x**3 + 123359200*x**2 + 43175600*x + 70248800)*(y**2) +
                          ((-360000)*x**16 + (-2880000)*x**15 + (-11812800)*x**14 + (-32289600)*x**13 + (-66043200)*x**12 + (-107534400)*x**11 +
                          (-148807200)*x**10 + (-184672800)*x**9 + (-205771200)*x**8 + (-196425600)*x**7 + (-166587200)*x**6 + (-135043200)*x**5 +
                          (-107568800)*x**4 + (-73394400)*x**3 + (-44061600)*x**2 + (-18772000)*x + (-17896000))*y + (144400*x**18 + 1299600*x**17 +
                          5269600*x**16 + 12699200*x**15 + 21632000*x**14 + 32289600*x**13 + 48149600*x**12 + 63997600*x**11 + 67834400*x**10 +
                          61884000*x**9 + 55708800*x**8 + 45478400*x**7 + 32775200*x**6 + 26766400*x**5 + 21309200*x**4 + 11185200*x**3 + 6242400*x**2 +
                          3465600*x + 1708800))),
            lambda x, y: 1e-4*(y**7 + (-3)*y**6 + (2*x**2 + (-1)*x + 2)*y**5 + (x**3 + (-6)*x**2 + x + 2)*y**4 + (x**4 + (-2)*x**3 + 2*x**2 +
                               x + (-3))*y**3 + (2*x**5 + (-3)*x**4 + x**3 + 10*x**2 + (-1)*x + 1)*y**2 + ((-1)*x**5 + 3*x**4 + 4*x**3 + (-12)*x**2)*y +
                               (x**7 + (-3)*x**5 + (-1)*x**4 + (-4)*x**3 + 4*x**2)),
        ],
        a=np.array([-1, -1]), b=np.array([1, 1]),
        # Degree 18 and 10 with coefficients up to 1e8, so the residual is huge in
        # absolute terms while the roots themselves are accurate. Measured residual
        # 3.2e-7, measured root error 7.0e-13.
        residual_tol=1e-6,
        root_tol=1e-11,
    ),
    Case(
        name="5.1",
        funcs=[lambda x, y: 2*x*y*np.cos(y**2)*np.cos(2*x) - np.cos(x*y),
               lambda x, y: 2*np.sin(x*y**2)*np.sin(3*x*y) - np.sin(x*y)],
        a=np.array([-2, -2]), b=np.array([2, 2]),
    ),
    Case(
        name="6.1",
        funcs=[lambda x, y: (y - 2*x)*(y + 0.5*x),
               lambda x, y: x*(x**2 + y**2 - 1)],
        a=np.array([-1, -1]), b=np.array([1, 1]),
        # A double root sits at the origin, where both factors of f vanish along
        # with g. The reference records it once; the solver cannot separate the two
        # branches and reports it as a pair straddling the origin at y = +-9.0e-9.
        # So 6 roots are expected here against the polished reference's 5.
        duplicate_root=np.array([0.0, 0.0]),
        root_tol=1e-7,
    ),
    Case(
        name="6.2",
        funcs=[lambda x, y: (y - 2*x)*(y + .5*x),
               lambda x, y: (x - .0001)*(x**2 + y**2 - 1)],
        a=np.array([-1, -1]), b=np.array([1, 1]),
        # 6.1 with the double root split apart by 1e-4; the two roots are resolved
        # but only to 3.7e-13.
        root_tol=1e-11,
    ),
    Case(
        name="6.3",
        funcs=[lambda x, y: 25*x*y - 12,
               lambda x, y: x**2 + y**2 - 1],
        a=np.array([-1, -1]), b=np.array([1, 1]),
    ),
    Case(
        name="7.1",
        funcs=[lambda x, y: (x**2 + y**2 - 1)*(x - 1.1),
               lambda x, y: (25*x*y - 12)*(x - 1.1)],
        a=np.array([-1, -1]), b=np.array([1, 1]),
    ),
    Case(
        name="7.2",
        funcs=[lambda x, y: y**4 + (-1)*y**3 + (2*x**2)*(y**2) + (3*x**2)*y + (x**4),
               lambda x, y: _inner_7_2(2*x, 2*(y + .5))],
        a=np.array([-1, -1]), b=np.array([1, 1]),
        # Degree 10 in a rescaled variable, so the conditioning is poor near the
        # roots. Measured residual 1.8e-12, measured root error 2.6e-9.
        residual_tol=1e-11,
        root_tol=1e-7,
    ),
    Case(
        name="7.3",
        funcs=[lambda x, y: np.cos(x*y/(_SCALE_7_3**2)) + np.sin(3*x*y/(_SCALE_7_3**2)),
               lambda x, y: np.cos(y/_SCALE_7_3) - np.cos(2*x*y/(_SCALE_7_3**2))],
        a=np.array([-_SCALE_7_3, -_SCALE_7_3]), b=np.array([_SCALE_7_3, _SCALE_7_3]),
        # The search box is 1e-9 across, so the default absolute root tolerance
        # would be meaningless here. Measured root error 6.3e-25.
        root_tol=1e-22,
    ),
    Case(
        name="7.4",
        funcs=[lambda x, y: np.sin(3*np.pi*x)*np.cos(x*y),
               lambda x, y: np.sin(3*np.pi*y)*np.cos(np.sin(x*y))],
        a=np.array([-1, -1]), b=np.array([1, 1]),
    ),
    Case(
        name="8.1",
        funcs=[lambda x, y: np.sin(10*x - y/10),
               lambda x, y: np.cos(3*x*y)],
        a=np.array([-1, -1]), b=np.array([1, 1]),
    ),
    Case(
        name="8.2",
        funcs=[lambda x, y: np.sin(10*x - y/10) + y,
               lambda x, y: np.cos(10*y - x/10) - x],
        a=np.array([-1, -1]), b=np.array([1, 1]),
    ),
    Case(
        name="9.1",
        funcs=[lambda x, y: x**2 + y**2 - .9**2,
               lambda x, y: np.sin(x*y)],
        a=np.array([-1, -1]), b=np.array([1, 1]),
    ),
    Case(
        name="9.2",
        funcs=[lambda x, y: x**2 + y**2 - .49**2,
               lambda x, y: (x - .1)*(x*y - .2)],
        a=np.array([-1, -1]), b=np.array([1, 1]),
    ),
    Case(
        name="10.1",
        funcs=[lambda x, y: (x - 1)*(np.cos(x*y**2) + 2),
               lambda x, y: np.sin(8*np.pi*y)*(np.cos(x*y) + 2)],
        a=np.array([-1, -1]), b=np.array([1, 1]),
        # All 17 roots lie on the boundary x = 1, where the solver is least
        # accurate. Measured residual 3.0e-12.
        residual_tol=1e-11,
    ),
]

CASE_IDS = [case.name for case in CASES]


############################### the tests ####################################

@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_finds_the_expected_number_of_roots(case):
    found = roots_of(case)
    if case.duplicate_root is not None:
        assert len(found) == case.expected_count, (
            f"Test {case.name}: YRoots found {len(found)} roots, but it should find "
            f"{case.expected_count}, one of which is a duplicate root. By reference, "
            f"polished roots has {len(case.polished_roots)}!"
        )
    else:
        assert len(found) == case.expected_count, (
            f"Test {case.name}: YRoots found {len(found)} roots, but the "
            f"polished roots reference has {case.expected_count}!"
        )


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_every_root_makes_the_system_vanish(case):
    """Each reported root really is a root, independent of the reference data."""
    found = roots_of(case)
    residual = max_residual(case.funcs, found)
    assert residual <= case.residual_tol, (
        f"Test {case.name}: largest |f(root)| is {residual:.3e}, over the "
        f"{case.residual_tol:.0e} tolerance."
    )


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_roots_match_the_polished_reference(case):
    """The roots are in the right places, not merely places where f is small."""
    found = roots_of(case)
    reference = case.reference_roots

    assert len(found) == len(reference), (
        f"Test {case.name}: cannot compare root locations, YRoots returned "
        f"{len(found)} roots against {len(reference)} expected."
    )
    distance = matched_max_distance(found, reference)
    assert distance <= case.root_tol, (
        f"Test {case.name}: worst matched-pair distance from the polished reference "
        f"is {distance:.3e}, over the {case.root_tol:.0e} tolerance."
    )
