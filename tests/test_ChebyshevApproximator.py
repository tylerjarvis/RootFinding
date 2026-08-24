"""Unit tests for yroots.ChebyshevApproximator."""
import math

import numpy as np
import pytest
from numpy.polynomial import chebyshev as C

import yroots.ChebyshevApproximator as CA
from yroots.ChebyshevApproximator import (transform, interval_approximate_nd, startedConverging,
                                          hasConverged, getFinalDegree, checkConstantInDimension,
                                          getChebyshevDegrees, getApproxError, chebApproximate,
                                          evaluateGrid)
from yroots.polynomial import MultiCheb, MultiPower


def cheb_poly_func(coeff):
    """A plain python callable evaluating the Chebyshev tensor coeff."""
    def f(*point):
        return MultiCheb(coeff, clean_zeros=False)(np.array([point]))[0]
    return f


############################### transform ####################################

def test_transform_maps_the_endpoints():
    a, b = np.array([-3.0, 0.0]), np.array([1.0, 10.0])
    assert np.allclose(transform(np.array([-1.0, -1.0]), a, b), a)
    assert np.allclose(transform(np.array([1.0, 1.0]), a, b), b)
    assert np.allclose(transform(np.array([0.0, 0.0]), a, b), (a + b) / 2)


def test_transform_is_the_identity_on_the_unit_box():
    x = np.array([-1.0, -0.25, 0.0, 0.5, 1.0])
    assert np.allclose(transform(x, -np.ones(5), np.ones(5)), x)


def test_transform_with_scalar_bounds():
    assert np.isclose(transform(0.5, 0.0, 4.0), 3.0)


############################ interval_approximate_nd #########################

@pytest.mark.parametrize("shape,seed", [((4,), 1), ((3, 3), 2), ((2, 3, 2), 3)])
def test_approximation_is_exact_for_chebyshev_polynomials(shape, seed):
    coeff = np.random.default_rng(seed).standard_normal(shape)
    degs = np.array(shape) - 1
    dim = len(shape)
    approx = interval_approximate_nd(cheb_poly_func(coeff), degs,
                                     -np.ones(dim), np.ones(dim))
    assert approx.shape == coeff.shape
    assert np.allclose(approx, coeff)


def test_approximation_shape_follows_degs():
    f = lambda x, y: np.exp(x) * np.sin(y)
    approx = interval_approximate_nd(f, np.array([5, 7]), -np.ones(2), np.ones(2))
    assert approx.shape == (6, 8)


def test_degree_zero_dimension_is_kept():
    """A degree of 0 in a dimension must produce a length-1 axis, not a length-2 one."""
    f = lambda x, y: np.cos(x)          # constant in y
    approx = interval_approximate_nd(f, np.array([6, 0]), -np.ones(2), np.ones(2))
    assert approx.shape == (7, 1)
    # the approximation interpolates cos exactly at the Chebyshev points it was built from
    grid = np.cos(np.arange(7) * np.pi / 6)
    assert np.allclose(C.chebval(grid, approx[:, 0]), np.cos(grid))


def test_interval_approximate_nd_does_not_modify_degs():
    """Regression test: degs used to be modified in place.

    chebApproximate passes the same array on to getApproxError, so mutating it made the
    reported approximation error too large by a factor of two for every constant dimension.
    """
    degs = np.array([4, 0])
    interval_approximate_nd(lambda x, y: np.exp(x), degs, -np.ones(2), np.ones(2))
    assert np.array_equal(degs, [4, 0])


def test_sup_norm_is_returned():
    f = lambda x: 3 * np.cos(x)
    approx, supNorm = interval_approximate_nd(f, np.array([20]), np.array([-np.pi]),
                                              np.array([np.pi]), retSupNorm=True)
    assert np.isclose(supNorm, 3.0)
    assert np.isclose(C.chebval(0.0, approx), 3.0)


def test_approximation_on_a_shifted_interval():
    f = np.exp
    a, b = np.array([1.0]), np.array([3.0])
    approx = interval_approximate_nd(f, np.array([20]), a, b)
    x = np.linspace(-1, 1, 25)
    assert np.allclose(C.chebval(x, approx), np.exp(transform(x, a, b)), atol=1e-12)


def test_polynomial_objects_are_evaluated_the_same_as_callables():
    coeff = np.array([[0.5, -1.0], [2.0, 0.25]])
    degs = np.array([3, 3])
    from_object = interval_approximate_nd(MultiCheb(coeff), degs, -np.ones(2), np.ones(2))
    from_callable = interval_approximate_nd(cheb_poly_func(coeff), degs, -np.ones(2), np.ones(2))
    assert np.allclose(from_object, from_callable)


############################### evaluateGrid #################################

# evaluateGrid hands the whole grid to f in one call and only falls back to evaluating point by
# point when that does not work. These check that the fallback is reached whenever it needs to be
# and that the two paths agree, since which one runs is invisible from the outside.

def grid_of(shape, a, b):
    """The Chebyshev grid of the given shape on [a,b], as meshgrid arrays."""
    axes = [transform(np.cos(np.arange(n) * np.pi / (n - 1)), a[i], b[i]) for i, n in enumerate(shape)]
    return np.meshgrid(*axes, indexing="ij")


def point_by_point(f, cheb_grid, shape):
    """What evaluateGrid's fallback path produces."""
    pts = np.column_stack(tuple(g.flatten() for g in cheb_grid))
    return np.array([f(*pt) for pt in pts]).reshape(shape)


@pytest.mark.parametrize("f", [
    lambda x, y: np.sin(3 * x * y) + x - 0.5,          # vectorized
    lambda x, y: math.sin(3 * x * y) + x - 0.5,        # scalar only: raises on arrays
    lambda x, y: x * y if x > 0 else -x * y,           # scalar only: ambiguous truth value
    lambda x, y: 2.5,                                  # constant: one value for the whole grid
    lambda x, y: x ** 2 - 0.25,                        # ignores an input
])
def test_evaluateGrid_agrees_with_evaluating_point_by_point(f):
    shape = (7, 5)
    cheb_grid = grid_of(shape, [-1.0, -1.0], [1.0, 1.0])
    values = evaluateGrid(f, cheb_grid, shape)
    assert values.shape == shape
    assert np.array_equal(values, point_by_point(f, cheb_grid, shape))


def test_evaluateGrid_calls_a_vectorized_function_once():
    calls = []
    def f(x, y):
        calls.append(np.shape(x))
        return x + y
    shape = (4, 3)
    evaluateGrid(f, grid_of(shape, [-1.0, -1.0], [1.0, 1.0]), shape)
    assert calls == [shape]


def test_approximation_of_a_scalar_only_function():
    """A function that cannot take arrays is still approximated, through the fallback path."""
    f = lambda x, y: math.exp(x) + math.sin(2 * y)
    vectorized = lambda x, y: np.exp(x) + np.sin(2 * y)
    a, b = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
    approx, error = chebApproximate(f, a, b)
    x = np.linspace(-1, 1, 9)
    grid_x, grid_y = np.meshgrid(x, x, indexing="ij")
    assert np.max(np.abs(C.chebgrid2d(x, x, approx) - vectorized(grid_x, grid_y))) <= error + 1e-14


############################### convergence checks ###########################

def test_startedConverging():
    assert startedConverging(np.array([10.0, 5.0, 1e-14, 1e-15, 1e-16, 1e-16, 1e-17]), 1e-12)
    assert not startedConverging(np.array([1e-14, 1e-15, 1e-16, 1e-16, 1e-2]), 1e-12)
    # only the last five coefficients matter
    assert startedConverging(np.array([1e9, 0.0, 0.0, 0.0, 0.0, 0.0]), 1e-12)


def test_startedConverging_with_zero_tolerance():
    """An identically zero function has tol == 0; its zero coefficients must count as converged."""
    assert startedConverging(np.zeros(8), 0.0)
    assert not startedConverging(np.array([0.0, 0.0, 0.0, 0.0, 1e-30]), 0.0)


def test_hasConverged():
    coeff = np.array([1.0, 0.5, 0.25])
    coeff2 = np.array([1.0, 0.5, 0.25, 1e-16, 1e-17])
    assert hasConverged(coeff, coeff2, 1e-12)

    coeff2_bad = np.array([1.0, 0.5, 0.25, 1e-3, 0.0])
    assert not hasConverged(coeff, coeff2_bad, 1e-12)

    shifted = np.array([1.0, 0.5, 0.3, 0.0, 0.0])
    assert not hasConverged(coeff, shifted, 1e-12)


def test_hasConverged_is_multidimensional():
    coeff = np.ones((2, 2))
    coeff2 = np.zeros((4, 4))
    coeff2[:2, :2] = 1.0
    assert hasConverged(coeff, coeff2, 1e-12)
    coeff2[3, 3] = 1.0
    assert not hasConverged(coeff, coeff2, 1e-12)


def test_hasConverged_does_not_modify_its_inputs():
    coeff = np.ones((2, 2))
    coeff2 = np.zeros((4, 4))
    coeff2[:2, :2] = 1.0
    hasConverged(coeff, coeff2, 1e-12)
    assert np.all(coeff == 1.0)
    assert coeff2[0, 0] == 1.0 and coeff2[3, 3] == 0.0


############################### getFinalDegree ###############################

def test_getFinalDegree_finds_the_last_significant_coefficient():
    coeff = np.array([1.0, .5, .2, .05, 1e-9, 1e-12, 1e-15, 1e-16, 0.0, 0.0])
    degree, epsVal, rho = getFinalDegree(coeff, 1e-10)
    assert degree == 5
    assert np.isclose(epsVal, 2e-15)
    assert rho > 1


def test_getFinalDegree_of_a_constant():
    coeff = np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    degree, epsVal, rho = getFinalDegree(coeff, 1e-10)
    assert degree == 0
    assert rho > 1


def test_getFinalDegree_of_all_zeros():
    """An identically zero function converges perfectly; the rate must not come out as 0."""
    degree, epsVal, rho = getFinalDegree(np.zeros(10), 0.0)
    assert degree == 0
    assert epsVal == 0
    assert rho == np.inf


def test_getFinalDegree_is_at_least_one_for_non_constants():
    coeff = np.array([1.0, 1e-13, 1e-14, 1e-15, 1e-16, 1e-16, 1e-16, 1e-16])
    degree, _, _ = getFinalDegree(coeff, 1e-16)
    assert degree >= 1


######################### checkConstantInDimension ###########################

def test_checkConstantInDimension_detects_a_constant_dimension():
    f = lambda x, y: np.exp(x) + 3
    a, b = -np.ones(2), np.ones(2)
    assert checkConstantInDimension(f, a, b, 1, 1e-10)      # constant in y
    assert not checkConstantInDimension(f, a, b, 0, 1e-10)  # varies in x


def test_checkConstantInDimension_on_a_fully_constant_function():
    f = lambda x, y, z: 7.0
    a, b = -np.ones(3), np.ones(3)
    assert all(checkConstantInDimension(f, a, b, d, 1e-10) for d in range(3))


def test_checkConstantInDimension_works_for_polynomial_objects():
    coeff = np.zeros((3, 3))
    coeff[2, 0], coeff[0, 0] = 1.0, 5.0     # depends on x only
    poly = MultiCheb(coeff)
    a, b = -np.ones(2), np.ones(2)
    assert checkConstantInDimension(poly, a, b, 1, 1e-10)
    assert not checkConstantInDimension(poly, a, b, 0, 1e-10)


############################ getChebyshevDegrees #############################

def test_getChebyshevDegrees_of_a_polynomial():
    coeff = np.zeros((4, 3))
    coeff[3, 0], coeff[0, 2], coeff[1, 1] = 1.0, 2.0, -1.0
    degs, epsilons, rhos = getChebyshevDegrees(cheb_poly_func(coeff), -np.ones(2), np.ones(2), 1e-10)
    assert list(degs) == [3, 2]
    assert len(epsilons) == len(rhos) == 2
    assert np.all(rhos > 1)


def test_getChebyshevDegrees_marks_constant_dimensions_as_zero():
    f = lambda x, y: np.sin(3 * x)
    degs, epsilons, rhos = getChebyshevDegrees(f, -np.ones(2), np.ones(2), 1e-10)
    assert degs[1] == 0
    assert degs[0] > 3
    assert epsilons[1] == 0 and rhos[1] == np.inf


############################### getApproxError ###############################

def test_getApproxError_is_zero_when_convergence_is_perfect():
    assert getApproxError(np.array([5, 5]), np.array([0.0, 0.0]), np.array([np.inf, np.inf])) == 0


def test_getApproxError_matches_the_hand_computed_sum():
    # one dimension: the tail is eps/(rho-1)
    degs, eps, rhos = np.array([4]), np.array([1e-15]), np.array([3.0])
    assert np.isclose(getApproxError(degs, eps, rhos), 1e-15 / 2)

    # two dimensions: (0,1) -> (degs[0]+1)*eps/(rho1-1), (1,0) -> (degs[1]+1)*eps/(rho0-1),
    #                 (1,1) -> eps/((rho0-1)(rho1-1))
    degs, eps, rhos = np.array([1, 2]), np.array([1e-12, 1e-12]), np.array([3.0, 5.0])
    expected = 2 * 1e-12 / 4 + 3 * 1e-12 / 2 + 1e-12 / (2 * 4)
    assert np.isclose(getApproxError(degs, eps, rhos), expected)


def test_getApproxError_grows_with_epsilon():
    degs, rhos = np.array([5, 5]), np.array([2.0, 2.0])
    small = getApproxError(degs, np.array([1e-16, 1e-16]), rhos)
    large = getApproxError(degs, np.array([1e-10, 1e-10]), rhos)
    assert 0 < small < large


############################### chebApproximate ##############################

@pytest.mark.parametrize("f,a,b", [
    (lambda x: np.sin(x), np.array([-1.0]), np.array([1.0])),
    (lambda x: np.exp(x), np.array([-2.0]), np.array([3.0])),
    (lambda x, y: np.sin(x) * np.cos(y), np.array([-1.0, -1.0]), np.array([1.0, 1.0])),
    (lambda x, y: 1 / (x + 4) + y ** 3, np.array([-1.0, -2.0]), np.array([1.0, 0.5])),
])
def test_chebApproximate_is_accurate_within_the_reported_error(f, a, b):
    coeff, error = chebApproximate(f, a, b)
    dim = len(a)
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1, 1, (40, dim))
    approx = np.array([MultiCheb(coeff, clean_zeros=False)(np.array([p]))[0] for p in pts])
    truth = np.array([f(*transform(p, a, b)) for p in pts])

    assert error >= 0
    assert np.max(np.abs(approx - truth)) < 1e-10
    # the reported bound should be a bound, not an underestimate of the true error
    assert np.max(np.abs(approx - truth)) <= max(error, 1e-13)


def test_chebApproximate_reproduces_a_chebyshev_polynomial_exactly():
    coeff = np.zeros((3, 4))
    coeff[2, 3], coeff[0, 0], coeff[1, 1] = 2.0, -0.5, 1.5
    approx, error = chebApproximate(cheb_poly_func(coeff), np.array([-1.0, -1.0]),
                                    np.array([1.0, 1.0]))
    assert approx.shape == coeff.shape
    assert np.allclose(approx, coeff, atol=1e-13)
    assert error < 1e-14


def test_chebApproximate_of_a_constant():
    approx, error = chebApproximate(lambda x, y: 4.0, np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    assert approx.shape == (1, 1)
    assert np.isclose(approx[0, 0], 4.0)
    assert error == 0


def test_chebApproximate_of_the_zero_function_terminates():
    """Regression test: the zero function used to make the degree guess double forever."""
    approx, error = chebApproximate(lambda x, y: 0.0 * x + 0.0 * y,
                                    np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    assert np.allclose(approx, 0.0)
    assert error == 0


def test_chebApproximate_accepts_lists_and_scalars():
    from_lists, _ = chebApproximate(lambda x, y: np.sin(x + y), [-1, -1], [1, 1])
    from_arrays, _ = chebApproximate(lambda x, y: np.sin(x + y), np.array([-1.0, -1.0]),
                                     np.array([1.0, 1.0]))
    assert np.allclose(from_lists, from_arrays)

    scalar, _ = chebApproximate(np.sin, -1, 1)
    assert scalar.ndim == 1


def test_chebApproximate_accepts_polynomial_objects():
    coeff = np.zeros((3, 3))
    coeff[2, 0], coeff[1, 1] = 1.0, -2.0
    poly = MultiCheb(coeff)
    approx, _ = chebApproximate(poly, np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    assert np.allclose(approx, poly.coeff, atol=1e-13)

    power = MultiPower(np.array([[1.0, 0.0], [0.0, 1.0]]))     # 1 + xy
    approx, _ = chebApproximate(power, np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    assert np.allclose(approx, [[1.0, 0.0], [0.0, 1.0]], atol=1e-13)


def test_chebApproximate_rejects_non_callables():
    with pytest.raises(ValueError, match="not callable"):
        chebApproximate("x**2", np.array([-1.0]), np.array([1.0]))


def test_chebApproximate_rejects_mismatched_bounds():
    with pytest.raises(ValueError, match="lower bounds"):
        chebApproximate(lambda x, y: x + y, np.array([-1.0, -1.0]), np.array([1.0]))


def test_chebApproximate_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="lower bound is greater"):
        chebApproximate(lambda x: x, np.array([1.0]), np.array([-1.0]))


def test_chebApproximate_rejects_bounds_of_the_wrong_dimension():
    with pytest.raises(ValueError, match="must match the dimension"):
        chebApproximate(lambda x, y: x + y, np.array([-1.0]), np.array([1.0]))
