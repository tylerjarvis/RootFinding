"""Unit tests for the public yroots.solve API (yroots.Combined_Solver)."""
import contextlib
import signal

import numpy as np
import pytest

import yroots as yr
from yroots.Combined_Solver import relativeRoundingError, solve


def sorted_rows(array):
    array = np.atleast_2d(np.asarray(array, dtype=float))
    if array.size == 0:
        return array
    return array[np.lexsort(array.T[::-1])]


############################### input validation #############################

def test_rejects_non_callable_input():
    with pytest.raises(ValueError, match="input function 1 is not callable"):
        solve([lambda x, y: x, "not a function"], [-1, -1], [1, 1])


def test_rejects_inverted_bounds():
    f = lambda x, y: x
    g = lambda x, y: y
    with pytest.raises(ValueError, match="lower bound is greater"):
        solve([f, g], np.array([1.2, -1]), np.array([1, 1]))


def test_rejects_mismatched_bound_lengths():
    f = lambda x, y: x
    g = lambda x, y: y
    with pytest.raises(ValueError, match="1 lower bounds were given but 2 upper bounds"):
        solve([f, g], [-1], [1, 1])


def test_rejects_bounds_that_do_not_match_the_function_signature():
    # two bounds, but only one function -> the solver looks for a one dimensional system
    with pytest.raises(ValueError, match="must match the dimension"):
        solve(lambda x, y: x + y, [-1], [1])


def test_rejects_a_polynomial_whose_dimension_is_not_the_system_size():
    poly = yr.MultiPower(np.array([[1.0, 2.0], [3.0, 4.0]]))     # 2 variables, 1 equation
    with pytest.raises(ValueError, match="N polynomials of dimension N"):
        solve(poly, -1, 1)


############################### input handling ###############################

def test_a_single_function_does_not_need_to_be_in_a_list():
    from_bare = solve(lambda x: x ** 2 - 0.25, -1, 1)
    from_list = solve([lambda x: x ** 2 - 0.25], -1, 1)
    assert np.allclose(sorted_rows(from_bare), sorted_rows(from_list))
    assert np.allclose(np.sort(np.ravel(from_bare)), [-0.5, 0.5], atol=1e-10)


def test_scalar_bounds_are_broadcast_to_every_dimension():
    funcs = [lambda x, y: x - 0.5, lambda x, y: y + 0.25]
    scalar = solve(funcs, -1, 1)
    listed = solve(funcs, [-1, -1], [1, 1])
    arrays = solve(funcs, np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    assert np.allclose(sorted_rows(scalar), sorted_rows(listed))
    assert np.allclose(sorted_rows(scalar), sorted_rows(arrays))


def test_functions_may_be_given_as_a_numpy_array():
    funcs = np.array([lambda x, y: x - 0.5, lambda x, y: y + 0.25], dtype=object)
    roots = solve(funcs, [-1, -1], [1, 1])
    assert np.allclose(roots, [[0.5, -0.25]], atol=1e-10)


def test_solve_does_not_modify_its_inputs():
    coeff = np.zeros((3, 3))
    coeff[2, 0], coeff[1, 2], coeff[0, 0] = 5.0, 3.0, 2.0
    poly = yr.MultiCheb(coeff.copy())
    original = poly.coeff.copy()
    power = yr.MultiPower(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    original_power = power.coeff.copy()

    a, b = [-2, -2], [2, 2]
    solve([poly, power], a, b)

    assert np.array_equal(poly.coeff, original)
    assert np.array_equal(power.coeff, original_power)
    assert a == [-2, -2] and b == [2, 2]


def test_integer_coefficient_polynomials_are_accepted():
    """Regression test: integer coefficients used to reach the solver un-cast and crash it."""
    coeff = np.array([0, 1, 2, 3])                    # T_1 + 2 T_2 + 3 T_3
    expected = np.sort(np.polynomial.chebyshev.chebroots(coeff.astype(float)))

    for bounds in [(-1, 1), (-2, 2)]:
        roots = solve(yr.MultiCheb(coeff), *bounds)
        assert np.allclose(np.sort(np.ravel(roots)), expected, atol=1e-10)

    power = yr.MultiPower(np.array([-1, 0, 4]))       # 4x^2 - 1
    roots = solve(power, -1, 1)
    assert np.allclose(np.sort(np.ravel(roots)), [-0.5, 0.5], atol=1e-10)

    # narrow and unsigned integer types crash the same way if they are not cast
    for dtype in (np.int8, np.uint8, np.uint32):
        roots = solve(yr.MultiCheb(coeff.astype(dtype)), -1, 1)
        assert np.allclose(np.sort(np.ravel(roots)), expected, atol=1e-10)


############################### results ######################################

def test_finds_the_root_of_a_linear_system():
    roots = solve([lambda x, y: x - 0.5, lambda x, y: y + 0.25], [-1, -1], [1, 1])
    assert roots.shape == (1, 2)
    assert np.allclose(roots[0], [0.5, -0.25], atol=1e-10)


def test_finds_all_roots_of_a_system_with_known_solutions():
    # sin(pi x) = 0 and y = x^2  ->  (-1, 1), (0, 0), (1, 1)
    f = lambda x, y: np.sin(np.pi * x)
    g = lambda x, y: y - x ** 2
    roots = sorted_rows(solve([f, g], [-1.5, -0.5], [1.5, 2.5]))
    assert roots.shape == (3, 2)
    assert np.allclose(roots, [[-1.0, 1.0], [0.0, 0.0], [1.0, 1.0]], atol=1e-10)


def test_roots_of_a_three_dimensional_system():
    f = lambda x, y, z: x - 0.5
    g = lambda x, y, z: y + x
    h = lambda x, y, z: z - y ** 2
    roots = solve([f, g, h], [-1, -1, -1], [1, 1, 1])
    assert roots.shape == (1, 3)
    assert np.allclose(roots[0], [0.5, -0.5, 0.25], atol=1e-10)


def test_polynomial_objects_agree_with_the_equivalent_callables():
    poly_roots = solve([yr.MultiPower(np.array([[-0.25, 0.0], [1.0, 0.0]])),      # x - 0.25
                        yr.MultiPower(np.array([[0.5, 1.0]]))],                   # y + 0.5
                       [-1, -1], [1, 1])
    callable_roots = solve([lambda x, y: x - 0.25, lambda x, y: y + 0.5], [-1, -1], [1, 1])
    assert np.allclose(sorted_rows(poly_roots), sorted_rows(callable_roots), atol=1e-10)


def test_multicheb_and_multipower_can_be_mixed():
    cheb = np.zeros((3, 3))
    cheb[2, 0], cheb[1, 2], cheb[0, 0] = 5.0, 3.0, 2.0
    power = np.zeros((4, 4))
    power[3, 0], power[1, 2], power[2, 0], power[0, 2], power[0, 0] = 5.0, 4.0, 3.0, 2.0, 1.0

    f, g = yr.MultiPower(power), yr.MultiCheb(cheb)
    roots = solve([f, g], [-1, -1], [1, 1])
    assert len(roots) > 0
    assert np.max(np.abs(f(roots))) < 1e-13
    assert np.max(np.abs(g(roots))) < 1e-13


def test_roots_are_found_outside_the_unit_box():
    f = lambda x, y: x - 3.5
    g = lambda x, y: y + 2.25
    roots = solve([f, g], [-5, -5], [5, 5])
    assert np.allclose(roots, [[3.5, -2.25]], atol=1e-9)


def test_an_asymmetric_search_box():
    f = lambda x, y: np.cos(x) - 0.5          # x = pi/3 in [0, 2]
    g = lambda x, y: y - x
    roots = solve([f, g], [0, 0], [2, 2])
    assert roots.shape == (1, 2)
    assert np.allclose(roots[0], [np.pi / 3, np.pi / 3], atol=1e-10)


@pytest.mark.parametrize("eps", [1e-3, 1e-5, 1e-7])
def test_ill_conditioned_system_keeps_its_root(eps):
    """Regression test: ill conditioned systems used to lose their root entirely.

    The interval padding exists so rounding error cannot discard a root, but it was
    computed from the reciprocal condition number and so never grew past machine
    precision. These two nearly parallel lines meet at (0.3, 0) and were thrown out.
    """
    f = lambda x, y: x + y - 0.3
    g = lambda x, y: x + (1 + eps) * y - 0.3
    roots = solve([f, g], [-1, -1], [1, 1])

    assert len(roots) == 1, f"root lost for a system with condition number ~{1/eps:.0e}"
    assert np.allclose(roots[0], [0.3, 0.0], atol=1e-6)


############################### polynomial input at any scale ################

@contextlib.contextmanager
def fails_instead_of_hanging(seconds):
    """Turn a hang into a test failure.

    Every regression below used to hang (or recurse until python's stack limit) rather than
    fail, which would stall the whole suite instead of reporting one broken test. Needs
    SIGALRM, so on a platform without it the body just runs unguarded.
    """
    if not hasattr(signal, "setitimer"):
        yield
        return

    def give_up(signum, frame):
        raise TimeoutError(f"solve did not finish within {seconds}s")

    previous = signal.signal(signal.SIGALRM, give_up)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


# y = x^2 and y = x^3 - x/2 meet at x = 0 and at x = (1 - sqrt 3)/2; the third crossing,
# x = (1 + sqrt 3)/2, is outside both search boxes used below.
_X0 = (1 - np.sqrt(3)) / 2
TWO_CURVES_ROOTS = np.array([[_X0, _X0 ** 2], [0.0, 0.0]])


def two_curves(scale, kind):
    """The two-curve system above as polynomial objects, every coefficient times ``scale``."""
    parabola = np.zeros((4, 2))
    parabola[0, 1], parabola[2, 0] = 1, -1                                # y - x^2
    cubic = np.zeros((4, 2))
    cubic[0, 1], cubic[3, 0], cubic[1, 0] = 1, -1, 0.5                    # y - x^3 + x/2
    polys = [yr.MultiPower(scale * parabola), yr.MultiPower(scale * cubic)]
    if kind is yr.MultiCheb:
        polys = [yr.MultiCheb(p.to_cheb()) for p in polys]
    return polys


@pytest.mark.parametrize("box", [([-1, -1], [1, 1]), ([-0.9, -1.2], [1.1, 0.8])],
                         ids=["unit box", "shifted box"])
@pytest.mark.parametrize("kind", [yr.MultiPower, yr.MultiCheb], ids=["MultiPower", "MultiCheb"])
@pytest.mark.parametrize("scale", [1e8, 1, 1e-8, 1e-16, 1e-30])
def test_polynomial_input_is_solved_at_any_scale(scale, kind, box):
    """Regression test: a polynomial given as coefficients below order 1 was never solved.

    Such a polynomial is exact, so its only error is rounding in its coefficients. That was
    set to a fixed macheps, which is only the right size for coefficients of order 1: scaled
    to 1e-14 it was about 2% of them and solve no longer finished, and from 1e-16 down it
    exceeded every coefficient. Passed as callables the same system was solved at every
    scale, since the approximator's error scales with the function.
    """
    with fails_instead_of_hanging(30):
        roots = solve(two_curves(scale, kind), *box)

    assert roots.shape == (2, 2)
    assert np.allclose(sorted_rows(roots), TWO_CURVES_ROOTS, atol=1e-10)


def test_the_polynomial_input_error_is_relative_to_the_coefficients():
    """The error is macheps times the size of the coefficients, so it scales with them."""
    coeff = np.array([[0.5, -3.0], [2.0, 0.25]])
    macheps = 2 ** -52

    assert relativeRoundingError(coeff) == macheps * 5.75
    for scale in [1e8, 1e-30]:
        assert np.isclose(relativeRoundingError(scale * coeff), scale * relativeRoundingError(coeff))


def test_the_zero_polynomial_keeps_a_nonzero_error():
    """A relative error of exactly 0 made solve report no roots for a polynomial that is
    zero everywhere, which is worse than the loud failure it has otherwise (see
    test_known_failures.py), so the zero polynomial keeps the fixed macheps."""
    assert relativeRoundingError(np.zeros((3, 3))) == 2 ** -52


def singular_4d_system(scale=1):
    """Two independent copies of a double root at the origin, as MultiPower objects.

    Same system as test_bounding_boxes.py's test_boxes_match_roots_4d_two_singularities:
    (y - 2x)(y + x/2), x(x^2 + y^2 - 1), and the same two in (z, w). It has 25 roots.
    """
    f = np.zeros((3,) * 4); f[0, 2, 0, 0], f[1, 1, 0, 0], f[2, 0, 0, 0] = 1, -1.5, -1
    g = np.zeros((4,) * 4); g[3, 0, 0, 0], g[1, 2, 0, 0], g[1, 0, 0, 0] = 1, 1, -1
    h = np.zeros((3,) * 4); h[0, 0, 0, 2], h[0, 0, 1, 1], h[0, 0, 2, 0] = 1, -1.5, -1
    k = np.zeros((4,) * 4); k[0, 0, 3, 0], k[0, 0, 1, 2], k[0, 0, 1, 0] = 1, 1, -1
    return [yr.MultiPower(scale * c) for c in (f, g, h, k)]


@pytest.mark.parametrize("scale", [1e8, 1, 1e-8, 1e-20])
def test_a_4d_polynomial_system_with_singular_roots_is_solved_at_any_scale(scale):
    """Regression test: this system crashed at scale 1 and did not finish at 1e-20.

    The crash was getSubdivisionDims leaving nothing to subdivide in a nearly converged
    box (see test_ChebyshevSubdivisionSolver.py); the fixed macheps error of polynomial
    input then kept it from finishing at 1e-20. The count it reports is still wrong -- see
    test_known_failures.py -- but it finishes, and every root it reports is a root.
    """
    polys = singular_4d_system(scale)
    with fails_instead_of_hanging(30):
        roots = solve(polys, -np.ones(4), np.ones(4))

    assert len(roots) > 0
    for p in polys:
        assert np.max(np.abs(p(roots))) < 1e-12 * scale


@pytest.mark.parametrize("d", [1e-7, 2e-8])
def test_a_root_on_an_axis_next_to_a_close_neighbour(d):
    """Regression test: x = 0, y(y - d) = 0 raised ZeroDivisionError from the compiled core.

    Solving it zooms x down to a subnormal width, and numba computed the power of two used to
    rescale that row as 0.0 (see test_boundingIntervalCore_handles_a_subnormal_linear_term).
    Only the crash is asserted here.
    """
    roots = solve([lambda x, y: x + 0 * y, lambda x, y: y * (y - d)], [-1, -1], [1, 1])

    assert len(roots) >= 1
    assert np.all(np.abs(roots[:, 0]) < 1e-12)
    assert np.all((roots[:, 1] > -1e-8) & (roots[:, 1] < d + 1e-8))


############################### close and multiple roots #####################

def test_two_close_roots_away_from_the_origin_and_off_the_axes_are_both_found():
    d = 1e-9
    roots = solve([lambda x, y: (x - y) * (x - y - d), lambda x, y: x + y - 0.5], [-1, -1], [1, 1])

    expected = [[0.25, 0.25], [0.25 + d / 2, 0.25 - d / 2]]
    assert np.allclose(sorted_rows(roots), expected, rtol=0, atol=1e-3 * d)


def test_a_double_root_is_reported_once():
    """y^2 = 0 has one root. On a small enough neighborhood its dip vanishes into the error and
    solving there finds nothing, so the root found on the whole box has to be kept."""
    roots = solve([lambda x, y: x + 0 * y, lambda x, y: y * y], [-1, -1], [1, 1])

    assert roots.shape == (1, 2)
    assert np.all(np.abs(roots) < 1e-8)


@pytest.mark.parametrize("d", [1e-7, 1e-8])
def test_close_roots_of_a_function_with_absolute_rounding_error_do_not_hang(d):
    """1 - cos(t) cancels inside the function, so it is off by about macheps near t = 0 however
    small the neighborhood. Approximating it there needs an ever higher degree; solve has to give
    up on the neighborhood and keep the point it found on the whole box, which lies between the
    two roots."""
    g = lambda x, y: 1 - np.cos(y - d / 2) - (d / 2) ** 2 / 2
    with fails_instead_of_hanging(30):
        roots = solve([lambda x, y: x + 0 * y, g], [-1, -1], [1, 1])

    assert len(roots) >= 1
    assert np.all((roots[:, 1] > -d) & (roots[:, 1] < 2 * d))


############################### empty results ################################

def test_no_roots_returns_an_empty_array_of_the_right_shape():
    """Regression test: an empty python list used to be returned instead of an array."""
    roots = solve([lambda x, y: x ** 2 + y ** 2 + 3, lambda x, y: x + y + 9], [-1, -1], [1, 1])
    roots = np.asarray(roots)
    assert roots.shape == (0, 2)
    assert roots.size == 0
    assert roots[:, 0].shape == (0,)          # indexing by dimension works with no roots


def test_no_roots_with_bounding_boxes():
    roots, boxes = solve([lambda x, y: x ** 2 + y ** 2 + 3, lambda x, y: x + y + 9],
                         [-1, -1], [1, 1], returnBoundingBoxes=True)
    assert np.asarray(roots).shape == (0, 2)
    assert np.asarray(boxes).shape == (0, 2, 2)


############################### bounding boxes ###############################

def test_bounding_boxes_contain_their_roots():
    f = lambda x, y: np.sin(4 * (x + y / 10 + np.pi / 10))
    g = lambda x, y: np.cos(2 * (x - 2 * y + np.pi / 7))
    a, b = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
    roots, boxes = solve([f, g], a, b, returnBoundingBoxes=True)

    assert len(roots) == len(boxes) > 0
    boxes = np.asarray(boxes)
    assert boxes.shape == (len(roots), 2, 2)
    for root, box in zip(roots, boxes):
        assert np.all(box[:, 0] <= root) and np.all(root <= box[:, 1])
        assert np.all(box[:, 0] >= a) and np.all(box[:, 1] <= b)
        assert np.all(box[:, 1] - box[:, 0] < 1e-4)


def test_returning_bounding_boxes_does_not_change_the_roots():
    funcs = [lambda x, y: x - 0.5, lambda x, y: y + 0.25]
    plain = solve(funcs, [-1, -1], [1, 1])
    with_boxes, _ = solve(funcs, [-1, -1], [1, 1], returnBoundingBoxes=True)
    assert np.allclose(sorted_rows(plain), sorted_rows(with_boxes))


############################### options ######################################

def test_exact_mode_finds_the_same_roots():
    f = lambda x, y: np.sin(4 * (x + y / 10 + np.pi / 10))
    g = lambda x, y: np.cos(2 * (x - 2 * y + np.pi / 7))
    fast = sorted_rows(solve([f, g], [-1, -1], [1, 1], exact=False))
    exact = sorted_rows(solve([f, g], [-1, -1], [1, 1], exact=True))
    assert fast.shape == exact.shape
    assert np.allclose(fast, exact, atol=1e-10)


def test_verbose_mode_reports_progress(capsys):
    solve([lambda x, y: x - 0.5, lambda x, y: y + 0.25], [-1, -1], [1, 1], verbose=True)
    output = capsys.readouterr().out
    assert "Approximation shapes" in output
    assert "Found 1 root" in output


def test_a_larger_minBoundingIntervalSize_gives_looser_boxes():
    funcs = [lambda x, y: np.sin(4 * (x + y / 10 + np.pi / 10)),
             lambda x, y: np.cos(2 * (x - 2 * y + np.pi / 7))]
    tight = np.asarray(solve(funcs, [-1, -1], [1, 1], returnBoundingBoxes=True)[1])
    loose = np.asarray(solve(funcs, [-1, -1], [1, 1], returnBoundingBoxes=True,
                             minBoundingIntervalSize=1e-2)[1])
    assert tight.shape == loose.shape
    assert np.max(tight[:, :, 1] - tight[:, :, 0]) <= np.max(loose[:, :, 1] - loose[:, :, 0])


############################### accuracy #####################################

def test_reported_roots_actually_solve_the_system():
    f = lambda x, y: np.exp(x + y) - 2
    g = lambda x, y: x - y ** 2
    roots = solve([f, g], [-1, -1], [1, 1])
    assert len(roots) > 0
    assert np.max(np.abs(f(roots[:, 0], roots[:, 1]))) < 1e-12
    assert np.max(np.abs(g(roots[:, 0], roots[:, 1]))) < 1e-12


def test_every_root_is_inside_the_search_interval():
    f = lambda x, y: np.sin(3 * x * y)
    g = lambda x, y: y - x ** 3
    a, b = np.array([-0.9, -0.9]), np.array([0.9, 0.9])
    roots = solve([f, g], a, b)
    assert len(roots) > 0
    assert np.all(roots >= a) and np.all(roots <= b)


def test_roots_are_not_duplicated():
    f = lambda x, y: np.sin(4 * (x + y / 10 + np.pi / 10))
    g = lambda x, y: np.cos(2 * (x - 2 * y + np.pi / 7))
    roots = solve([f, g], [-1, -1], [1, 1])
    distances = np.linalg.norm(roots[:, None, :] - roots[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    assert np.min(distances) > 1e-6


@pytest.mark.parametrize("scale", [1e8, 1e4, 1e-4, 1e-10, 1e-20, 1e-30])
def test_scaling_the_system_does_not_change_the_roots(scale):
    """Multiplying every equation by a constant is a change of units; the roots do not move.

    The approximator used to floor its convergence value at an absolute macheps, so a system
    scaled below about 1e-16 had its error bound come out negative and every interval holding a
    root discarded -- the solver returned nothing at all, without a warning.
    """
    from scipy.optimize import linear_sum_assignment
    f = lambda x, y: np.sin(3 * (x + y))
    g = lambda x, y: np.sin(3 * (x - y))
    a, b = np.array([-1.0, -1.0]), np.array([1.0, 1.0])

    reference = solve([f, g], a, b)
    scaled = solve([lambda x, y: scale * f(x, y), lambda x, y: scale * g(x, y)], a, b)

    assert len(scaled) == len(reference), (
        f"scaling by {scale:.0e} changed the root count from {len(reference)} to {len(scaled)}")
    # Pair the two sets optimally rather than sorting them: roots whose coordinates differ in the
    # last bit can otherwise swap places and look as though they moved.
    distances = np.linalg.norm(scaled[:, None, :] - reference[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(distances)
    assert distances[rows, cols].max() < 1e-12


@pytest.mark.parametrize("funcs, root", [
    ([lambda x, y: (x - .3)**2, lambda x, y: y - .1], (.3, .1)),
    ([lambda x, y: y - .1, lambda x, y: (x - .3)**2], (.3, .1)),
    ([lambda x, y: (y - .1)**2, lambda x, y: x - .3], (.3, .1)),
])
def test_a_double_root_where_one_function_only_touches_zero_is_reported(funcs, root):
    """(x-.3)**2 only touches zero, so the root at (.3, .1) is a double root.

    Its Chebyshev approximation is purely quadratic, and the quadratic check computed the sum of
    the non-quadratic coefficients as |M|.sum() minus the quadratic ones. That cancelled to
    -2e-16, below the rounded minimum of the quadratic part, so the whole search box was thrown
    out and solve returned nothing. Once past that, the final step's box had x a few 1e-8 wide
    and y of width zero; getSubdivisionDims treated both as collapsed, kept only y, and then had
    no dimension left to subdivide.
    """
    with pytest.warns(UserWarning, match="Might Have Duplicate Roots"):
        roots, boxes = solve(funcs, [-1, -1], [1, 1], returnBoundingBoxes=True)
    assert len(roots) > 0, "the double root was discarded"
    assert np.max(np.abs(roots - np.array(root))) < 1e-6
    assert any(np.all(box[:, 0] <= root) and np.all(root <= box[:, 1]) for box in boxes), (
        "no bounding box contains the double root")


############################### package import ###############################

def _import_yroots_in_subprocess(preamble):
    """Import yroots in a fresh interpreter, after running `preamble` in it.

    The optimization cap in yroots/__init__.py runs once, at import, so by the time a test
    function executes it has already happened and cannot be re-triggered in process without
    tampering with sys.modules. A subprocess is the honest way to exercise it.
    """
    import os
    import subprocess
    import sys
    import textwrap
    import yroots
    code = textwrap.dedent(preamble) + textwrap.dedent("""
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import yroots
        print("IMPORT-OK")
        for w in caught:
            print("WARNING", w.category.__name__, str(w.message)[:200])
    """)
    env = dict(os.environ)
    # Import the same yroots this test session is using, not whatever is installed.
    package_parent = os.path.dirname(os.path.dirname(os.path.abspath(yroots.__file__)))
    env["PYTHONPATH"] = package_parent + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("NUMBA_OPT", None)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)


def test_importing_yroots_survives_a_numba_that_cannot_reread_its_config():
    """The optimization cap is a convenience, so failing to apply it must not break the import.

    yroots asks numba to re-read NUMBA_OPT through reload_config, which is not public numba API.
    Called unguarded, a numba that renamed it took `import yroots` down with it.
    """
    result = _import_yroots_in_subprocess("""
        import numba
        from numba.core import config
        def _raises(*args, **kwargs):
            raise AttributeError("pretend this numba has no reload_config")
        config.reload_config = _raises
    """)
    assert "IMPORT-OK" in result.stdout, (
        f"importing yroots failed when numba could not re-read its config:\n{result.stderr}")
    assert "WARNING RuntimeWarning" in result.stdout, (
        f"no warning was raised to explain the slow compile that follows:\n{result.stdout}")
    assert "NUMBA_OPT" in result.stdout, "the warning does not name the variable to set"


def test_importing_yroots_after_numba_is_quiet_when_the_cap_applies():
    """The other side: when the re-read works there is nothing to warn about."""
    result = _import_yroots_in_subprocess("import numba")
    assert "IMPORT-OK" in result.stdout, result.stderr
    assert "RuntimeWarning" not in result.stdout, (
        f"warned even though the cap applied cleanly:\n{result.stdout}")



############################### duplicate roots ##############################

def _solve_recording_warnings(funcs, a, b, **kwargs):
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        roots = solve(funcs, a, b, **kwargs)
    return roots, [str(w.message) for w in caught]


@pytest.mark.parametrize("kwargs", [dict(max_cpu=1, parallel_depth=0), dict(max_cpu=2, parallel_depth=1)],
                         ids=["serial", "parallel"])
def test_duplicate_roots_are_listed_by_set(kwargs):
    """y = x**2 and y = -x**2 touch at the origin, so the solver hands back two points it cannot tell
    apart. The warning names those points, in the coordinates of the search interval, and suggests
    checking the Jacobian there."""
    funcs = [lambda x, y: y - x**2, lambda x, y: y + x**2]
    roots, messages = _solve_recording_warnings(funcs, -1, 1, **kwargs)
    dupMessages = [m for m in messages if "duplicates" in m]
    assert len(dupMessages) == 1, messages
    message = dupMessages[0]
    assert "Set 1:" in message and "Set 2:" not in message
    assert "We suggest you check the rank of the Jacobian at these points." in message
    for root in roots:
        assert "(" + ", ".join(f"{x:.16g}" for x in root) + ")" in message
    assert not any("Might Have Duplicate Roots" in m for m in messages)


def test_simple_roots_raise_no_duplicate_warning():
    funcs = [lambda x, y: x - 0.3, lambda x, y: y + 0.2]
    roots, messages = _solve_recording_warnings(funcs, -1, 1)
    assert len(roots) == 1
    assert not any("duplicates" in m for m in messages), messages


def test_duplicate_warning_lists_each_set_on_its_own_line():
    import warnings
    from yroots.Combined_Solver import _warnDuplicateRoots
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _warnDuplicateRoots([np.array([[0.5, 0.5], [0.5000000000062979, 0.5]]),
                             np.array([[1e-8, 2.0, 0.0], [1.1e-8, 2.0, 0.0], [9e-9, 2.0, 0.0]])])
    assert len(caught) == 1
    assert str(caught[0].message).splitlines() == [
        "The roots in each of the following sets might be duplicates of each other:",
        "  Set 1: (0.5, 0.5), (0.5000000000062979, 0.5)",
        "  Set 2: (1e-08, 2, 0), (1.1e-08, 2, 0), (9e-09, 2, 0)",
        "We suggest you check the rank of the Jacobian at these points.",
    ]
