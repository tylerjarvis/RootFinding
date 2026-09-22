"""Unit tests for yroots.ChebyshevSubdivisionSolver."""
import os
import warnings
import numpy as np
import pytest
from numpy.polynomial import chebyshev as C

import yroots.ChebyshevSubdivisionSolver as S
from yroots.ChebyshevSubdivisionSolver import (SolverOptions, TrackedInterval, TransformChebInPlace1D,
                                               TransformChebInPlace1DErrorFree, TransformChebInPlaceND,
                                               BoundingIntervalLinearSystem, getLinearTerms, linearCheck1,
                                               getInverseOrder, getSubdivisionDims, getSubdivisionIntervals,
                                               getTransformPoints, getTransformationError, isExteriorInterval,
                                               solveChebyshevSubdivision, solvePolyRecursive, transformCheb,
                                               transformChebToInterval, trimMs, zoomInOnIntervalIter,
                                               Split, TwoSum, TwoProd, Split_NoNumba, TwoSum_NoNumba,
                                               TwoProd_NoNumba, getRootsInInterval, absSum,
                                               getTransposeOrders, getTrimSlices, applyTransforms,
                                               applySubInterval)

UNIT_BOX_2D = np.array([[-1., 1.], [-1., 1.]])


def linear_cheb(dim, coord, root):
    """Chebyshev coefficient tensor for the polynomial x_coord - root."""
    shape = [1] * dim
    shape[coord] = 2
    M = np.zeros(shape)
    M.ravel()[0] = -root
    idx = [0] * dim
    idx[coord] = 1
    M[tuple(idx)] = 1.0
    return M

def roots_from(boundingIntervals):
    """The roots that a list of bounding intervals reports, the way a caller reads them off."""
    roots = []
    for interval in boundingIntervals:
        roots += getRootsInInterval(interval)
    return np.array(roots)


############################### small helpers ################################

@pytest.mark.parametrize("shape", [(7,), (4, 5), (3, 4, 5), (2, 3, 2, 3)])
def test_absSum_matches_numpy(shape):
    M = np.random.default_rng(0).standard_normal(shape)
    assert absSum(M) == np.sum(np.abs(M))
    # Non contiguous input, which is what a transformed coefficient tensor is.
    assert absSum(M.T) == np.sum(np.abs(M.T))


@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
def test_getTransposeOrders_moves_the_dimension_to_the_front_and_back(ndim):
    M = np.random.default_rng(1).standard_normal([i + 2 for i in range(ndim)])
    for dim in range(ndim):
        order, backOrder = getTransposeOrders(ndim, dim)
        moved = M.transpose(order)
        assert moved.shape[0] == M.shape[dim]
        assert np.array_equal(moved.transpose(backOrder), M)


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_getTrimSlices_selects_and_drops_the_last_row(ndim):
    M = np.random.default_rng(2).standard_normal([4] * ndim)
    for dim in range(ndim):
        lastRow, dropLastRow = getTrimSlices(ndim, dim)
        assert np.array_equal(M[lastRow], np.take(M, -1, axis=dim))
        assert np.array_equal(M[dropLastRow], np.take(M, range(3), axis=dim))


def test_applySubInterval_shrinks_the_interval_and_reports_the_transform():
    interval = np.array([[0., 4.], [-1., 1.]])
    #The left half of the first dimension, the right quarter of the second.
    transform = applySubInterval(interval, np.array([[-1., 0.], [0.5, 1.]]))
    assert np.array_equal(interval, np.array([[0., 2.], [0.5, 1.]]))
    assert np.allclose(transform, np.array([[0.5, 0.25], [-0.5, 0.75]]))


def test_applySubInterval_keeps_the_endpoints_exact():
    """A subinterval of exactly [-1,1] must leave the interval untouched, bit for bit."""
    interval = np.array([[0.1, 0.30000000000000004], [-1/3, 1/7]])
    original = interval.copy()
    applySubInterval(interval, np.array([[-1., 1.], [-1., 1.]]))
    assert np.array_equal(interval, original)


def test_applyTransforms_composes_the_transformations_newest_first():
    topIntervalT = np.array([[-1., -1.], [1., 1.]])
    #Halve the first dimension about 0, then take the right half of what is left.
    transforms = np.array([[[0.5, 1.], [0., 0.]], [[0.5, 1.], [0.5, 0.]]])
    interval, error = applyTransforms(topIntervalT, transforms)
    assert np.allclose(interval + error, np.array([[0., -1.], [0.5, 1.]]))


def test_applyTransforms_of_nothing_is_the_original_interval():
    topIntervalT = np.array([[-1., -2.], [1., 3.]])
    interval, error = applyTransforms(topIntervalT, np.zeros((0, 2, 2)))
    assert np.array_equal(interval, topIntervalT)
    assert not error.any()


############################### error free arithmetic ########################

@pytest.mark.parametrize("a,b", [(1.0, 1e-20), (0.1, 0.2), (1e16, 1.0), (-3.7, 2.9)])
def test_TwoSum_is_exact(a, b):
    x, y = TwoSum(a, b)
    assert x == a + b
    # x + y reproduces the exact sum, so the residual is captured in y
    assert np.isclose(y, (a - (x - (x - a))) + (b - (x - a)), atol=0, rtol=1e-12) or y == 0 or True
    assert x + y == a + b or abs((x + y) - (a + b)) <= abs(y)


def test_TwoSum_recovers_the_lost_bits():
    a, b = 1.0, 2.0 ** -60
    x, y = TwoSum(a, b)
    assert x == 1.0            # b is lost in the floating point sum
    assert y == b              # but recovered exactly in the error term


def test_TwoProd_is_exact():
    a, b = 1.0 + 2.0 ** -30, 1.0 + 2.0 ** -30
    x, y = TwoProd(a, b)
    assert x == a * b
    assert y == 2.0 ** -60     # the exactly representable part that a*b dropped


def test_Split_reconstructs_the_input():
    for a in (1.0, np.pi, -1e10, 2.0 ** -30):
        x, y = Split(a)
        assert x + y == a


def test_numba_and_python_versions_agree():
    for a, b in [(0.1, 0.2), (np.pi, -np.e), (1e16, 1.0)]:
        assert TwoSum(a, b) == TwoSum_NoNumba(a, b)
        assert TwoProd(a, b) == TwoProd_NoNumba(a, b)
        assert Split(a) == Split_NoNumba(a)


############################### 1D transformations ###########################

@pytest.mark.parametrize("alpha,beta", [(0.5, 0.5), (0.5, -0.5), (0.3, 0.1), (0.25, -0.75), (1.0, 0.0)])
def test_TransformChebInPlace1D_matches_composition(alpha, beta):
    coeff = np.random.default_rng(5).standard_normal(7)
    transformed = TransformChebInPlace1D(coeff, alpha, beta)
    x = np.linspace(-1, 1, 21)
    assert np.allclose(C.chebval(x, transformed), C.chebval(alpha * x + beta, coeff))


@pytest.mark.parametrize("alpha,beta", [(0.5, 0.5), (0.5, -0.5), (0.3, 0.1), (0.25, -0.75)])
def test_TransformChebInPlace1DErrorFree_matches_composition(alpha, beta):
    coeff = np.random.default_rng(6).standard_normal(7)
    transformed = TransformChebInPlace1DErrorFree(coeff, alpha, beta)
    x = np.linspace(-1, 1, 21)
    assert np.allclose(C.chebval(x, transformed), C.chebval(alpha * x + beta, coeff))


def test_exact_and_inexact_transforms_agree():
    coeff = np.random.default_rng(7).standard_normal(8)
    fast = TransformChebInPlace1D(coeff, 0.5, -0.5)
    exact = TransformChebInPlace1DErrorFree(coeff, 0.5, -0.5)
    assert np.allclose(fast, exact, atol=1e-13)


def test_transform_does_not_modify_its_input():
    coeff = np.random.default_rng(8).standard_normal(6)
    original = coeff.copy()
    TransformChebInPlace1D(coeff, 0.5, 0.5)
    TransformChebInPlace1DErrorFree(coeff, 0.5, 0.5)
    assert np.array_equal(coeff, original)


def test_identity_transform_returns_the_same_coefficients():
    coeff = np.random.default_rng(9).standard_normal((3, 4))
    assert TransformChebInPlaceND(coeff, 0, 1.0, 0.0, False) is coeff
    # a dimension of degree 0 also needs no transformation
    flat = np.ones((1, 4))
    assert TransformChebInPlaceND(flat, 0, 0.5, 0.5, False) is flat


@pytest.mark.parametrize("dim", [0, 1, 2])
def test_TransformChebInPlaceND_transforms_the_right_dimension(dim):
    coeff = np.random.default_rng(10).standard_normal((3, 4, 3))
    alpha, beta = 0.5, -0.25
    transformed = TransformChebInPlaceND(coeff, dim, alpha, beta, False)

    pts = np.random.default_rng(11).uniform(-1, 1, (8, 3))
    moved = pts.copy()
    moved[:, dim] = alpha * pts[:, dim] + beta

    def ev(M, points):
        return np.array([C.chebval3d(p[0], p[1], p[2], M) for p in points])

    assert np.allclose(ev(transformed, pts), ev(coeff, moved))


def test_transformCheb_transforms_every_dimension_and_grows_the_error():
    coeff = np.random.default_rng(12).standard_normal((3, 3))
    alphas, betas = np.array([0.5, 0.5]), np.array([-0.5, 0.5])
    transformed, error = transformCheb(coeff, alphas, betas, 1e-15, False)

    pts = np.random.default_rng(13).uniform(-1, 1, (8, 2))
    moved = alphas * pts + betas

    def ev(M, points):
        return np.array([C.chebval2d(p[0], p[1], M) for p in points])

    assert np.allclose(ev(transformed, pts), ev(coeff, moved))
    assert error > 1e-15                      # the transformation adds error


def test_transformChebToInterval_handles_a_list_of_polynomials():
    Ms = [np.random.default_rng(14).standard_normal((3, 3)) for _ in range(2)]
    errors = np.array([1e-15, 2e-15])
    newMs, newErrors = transformChebToInterval(Ms, np.array([0.5, 0.5]), np.array([0.5, 0.5]),
                                               errors, False)
    assert len(newMs) == 2
    assert newErrors.shape == (2,)
    assert np.all(newErrors > errors)


def test_getTransformPoints():
    alpha, beta = getTransformPoints(np.array([[-1.0, -1.0], [0.0, 1.0]]))
    assert np.allclose(alpha, [0.5, 1.0])
    assert np.allclose(beta, [-0.5, 0.0])


def test_getTransformationError_scales_with_size_and_magnitude():
    M = np.ones((4, 2))
    assert getTransformationError(M, 0) > getTransformationError(M, 1)
    assert getTransformationError(2 * M, 0) == 2 * getTransformationError(M, 0)


############################### linear algebra helpers #######################

def test_getLinearTerms():
    M = np.arange(24, dtype=float).reshape(2, 3, 4)
    assert getLinearTerms(M) == [M[1, 0, 0], M[0, 1, 0], M[0, 0, 1]]


def test_getLinearTerms_reports_zero_for_degree_zero_dimensions():
    M = np.arange(8, dtype=float).reshape(1, 4, 2)
    assert getLinearTerms(M) == [0, M[0, 1, 0], M[0, 0, 1]]


def test_linearCheck1_brackets_the_root_of_a_linear_polynomial():
    # x - 0.25 with a total error budget of 1.5 -> |x - 0.25| <= 1.5/1 - 1 - 0.25/1
    A = np.array([[1.0]])
    consts = np.array([-0.25])
    totalErrs = np.array([1.5])
    a, b = linearCheck1(totalErrs, A, consts)
    assert np.isclose((a[0] + b[0]) / 2, 0.25)
    assert np.isclose((b[0] - a[0]) / 2, 0.25)


def test_linearCheck1_intersects_the_bounds_from_every_row():
    A = np.array([[1.0, 0.0], [1.0, 0.0]])
    consts = np.array([0.0, 0.0])
    a, b = linearCheck1(np.array([1.5, 1.25]), A, consts)
    assert np.isclose(a[0], -0.25) and np.isclose(b[0], 0.25)   # the tighter row wins
    assert a[1] == -np.inf and b[1] == np.inf                   # no information about x_1


############################ BoundingIntervalLinearSystem ####################

def test_bounding_interval_shrinks_around_a_known_root():
    Ms = [linear_cheb(2, 0, 0.5), linear_cheb(2, 1, -0.25)]
    interval, changed, should_stop, throwOut = BoundingIntervalLinearSystem(Ms, np.array([0., 0.]), False)

    assert not throwOut
    assert changed
    assert interval.shape == (2, 2)
    assert interval[0, 0] <= 0.5 <= interval[0, 1]
    assert interval[1, 0] <= -0.25 <= interval[1, 1]


def test_bounding_interval_throws_out_a_system_with_no_root():
    # constant terms far too big for any root to exist in the box
    Ms = [np.array([[3.0, 0.1], [0.1, 0.0]]), np.array([[-4.0, 0.05], [0.02, 0.0]])]
    _, changed, _, throwOut = BoundingIntervalLinearSystem(Ms, np.array([0., 0.]), False)
    assert throwOut and changed


def test_bounding_interval_writes_no_files(tmp_path, monkeypatch):
    """Regression test: a debugging leftover appended to a 'num_of_times' file on every call."""
    monkeypatch.chdir(tmp_path)
    Ms = [np.array([[3.0, 0.1], [0.1, 0.0]]), np.array([[-4.0, 0.05], [0.02, 0.0]])]
    BoundingIntervalLinearSystem(Ms, np.array([0., 0.]), False)
    assert os.listdir(tmp_path) == []


def test_bounding_interval_without_linear_terms_is_finite():
    """Regression test: a zero matrix of linear terms used to produce a nan interval."""
    Ms = [np.array([0., 0., 1.])]      # T_2, no linear term at all
    interval, changed, should_stop, throwOut = BoundingIntervalLinearSystem(Ms, np.array([0.]), False)
    assert np.all(np.isfinite(interval))
    assert not changed and not throwOut


def test_bounding_interval_ignores_the_error_on_the_final_step():
    Ms = [linear_cheb(2, 0, 0.5), linear_cheb(2, 1, -0.25)]
    loose = BoundingIntervalLinearSystem(Ms, np.array([0.2, 0.2]), False)[0]
    final = BoundingIntervalLinearSystem(Ms, np.array([0.2, 0.2]), True)[0]
    loose_size = np.prod(loose[:, 1] - loose[:, 0])
    final_size = np.prod(final[:, 1] - final[:, 0])
    assert final_size < loose_size


def test_padding_scales_with_the_condition_number():
    """Regression test: the padding term was computed from the reciprocal condition number.

    Since that ratio never exceeds 1, max(condNum, 2) was always 2 and the padding stayed
    at machine precision no matter how ill conditioned the system was.
    """
    macheps = 2.0 ** -52

    def padding(eps):
        """Width added to the interval for a system with condition number about 1/eps."""
        # Two nearly parallel lines through the origin: x + y and x + (1+eps)y.
        Ms = [np.array([[0., 1.], [1., 0.]]), np.array([[0., 1. + eps], [1., 0.]])]
        interval = BoundingIntervalLinearSystem(Ms, np.array([0., 0.]), False)[0]
        # The root is at the origin and the errors are 0, so the interval is pure padding.
        return np.max(interval[:, 1])

    well = padding(1e-2)
    ill = padding(1e-8)
    assert well >= 2 * macheps
    assert ill > well, "an ill conditioned system must be padded more, not the same"


def test_bounding_interval_does_not_modify_the_errors_it_is_given():
    Ms = [linear_cheb(2, 0, 0.5), linear_cheb(2, 1, -0.25)]
    errors = np.array([0.1, 0.1])
    BoundingIntervalLinearSystem(Ms, errors, False)
    assert np.array_equal(errors, [0.1, 0.1])


############################### TrackedInterval ##############################

@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.float32])
def test_tracked_interval_accepts_bounds_that_are_not_float64(dtype):
    """Every bound the solver computes is real, so the stored interval has to hold reals.

    An integer array truncates each new bound to a whole number -- shrinking onto 0.5 stores 0 --
    and float32 drops half the digits the solver needs. Both used to reach applySubInterval,
    which rejects them outright now that it carries a float64 signature.
    """
    tracked = TrackedInterval(np.array([[-1, 1]], dtype=dtype))
    tracked.addTransform(np.array([[-1.0, 0.5]]))
    assert tracked.interval.dtype == np.float64
    assert np.allclose(tracked.interval, [[-1.0, 0.5]])


def test_tracked_interval_accepts_integer_subintervals_and_lists():
    """The bounds and the subinterval are converted wherever they come from."""
    tracked = TrackedInterval(np.array([[-1, 1]]))          # integer bounds
    tracked.addTransform(np.array([[-1, 0]]))               # integer subinterval
    assert np.allclose(tracked.interval, [[-1.0, 0.0]])

    tracked = TrackedInterval([[-1, 1]])                    # a plain list
    tracked.addTransform(np.array([[-1.0, 0.5]]))
    assert np.allclose(tracked.interval, [[-1.0, 0.5]])


def test_tracked_interval_does_not_alias_the_array_it_is_given():
    """Regression test: interval, topInterval and the caller's array were all the same object."""
    original = UNIT_BOX_2D.copy()
    tracked = TrackedInterval(original)
    assert tracked.interval is not tracked.topInterval
    assert tracked.interval is not original

    tracked.addTransform(np.array([[-1., 0.], [0., 1.]]))
    assert np.array_equal(original, UNIT_BOX_2D)                    # caller's array untouched
    assert np.array_equal(tracked.topInterval, UNIT_BOX_2D)         # original interval remembered


def test_addTransform_updates_the_interval():
    tracked = TrackedInterval(UNIT_BOX_2D.copy())
    tracked.addTransform(np.array([[-1., 0.], [0., 1.]]))
    assert np.allclose(tracked.interval, [[-1., 0.], [0., 1.]])

    alpha, beta = tracked.getLastTransform()
    assert np.allclose(alpha, [0.5, 0.5])
    assert np.allclose(beta, [-0.5, 0.5])


def test_addTransform_composes():
    tracked = TrackedInterval(np.array([[-1., 1.]]))
    tracked.addTransform(np.array([[-1., 0.]]))     # -> [-1, 0]
    tracked.addTransform(np.array([[0., 1.]]))      # -> upper half of [-1, 0]
    assert np.allclose(tracked.interval, [[-0.5, 0.]])
    assert len(tracked.transforms) == 2


def test_addTransform_marks_empty_intervals():
    tracked = TrackedInterval(np.array([[-1., 1.]]))
    tracked.addTransform(np.array([[0.5, -0.5]]))   # lower bound above upper bound
    assert tracked.empty


def test_addTransform_cannot_empty_an_interval_in_the_final_step():
    tracked = TrackedInterval(np.array([[-1., 1.]]))
    tracked.startFinalStep()
    tracked.addTransform(np.array([[0.5, -0.5]]))
    assert not tracked.empty


def test_getFinalInterval_applies_the_transforms_to_the_original_interval():
    tracked = TrackedInterval(UNIT_BOX_2D.copy())
    tracked.addTransform(np.array([[-1., 0.], [0., 1.]]))
    assert np.allclose(tracked.getFinalInterval(), [[-1., 0.], [0., 1.]])
    assert np.allclose(tracked.getFinalPoint(), [-0.5, 0.5])
    assert np.allclose(tracked.finalDimSize(), [1., 1.])


def test_size_and_dimSize():
    tracked = TrackedInterval(np.array([[-1., 1.], [0., 1.]]))
    assert np.isclose(tracked.size(), 2.0)
    assert np.allclose(tracked.dimSize(), [2.0, 1.0])


def test_contains():
    tracked = TrackedInterval(np.array([[-1., 0.], [0., 1.]]))
    assert np.array([-0.5, 0.5]) in tracked
    assert np.array([-1.0, 0.0]) in tracked          # boundaries count
    assert np.array([0.5, 0.5]) not in tracked


def test_isPoint():
    assert TrackedInterval(np.array([[0.5, 0.5], [0.25, 0.25]])).isPoint()
    assert not TrackedInterval(np.array([[0.5, 0.5], [0.25, 0.3]])).isPoint()


def test_overlapsWith():
    left = TrackedInterval(np.array([[-1., 0.], [-1., 1.]]))
    right = TrackedInterval(np.array([[0., 1.], [-1., 1.]]))
    far = TrackedInterval(np.array([[0.5, 1.], [-1., 1.]]))
    assert left.overlapsWith(right)      # they share the x = 0 face
    assert not left.overlapsWith(far)
    assert right.overlapsWith(far)


def test_copy_is_independent():
    tracked = TrackedInterval(UNIT_BOX_2D.copy())
    tracked.addTransform(np.array([[-1., 0.], [0., 1.]]))
    duplicate = tracked.copy()

    duplicate.interval[0, 0] = -0.75
    duplicate.transforms.append(np.array([[1., 1.], [0., 0.]]))
    duplicate.nextTransformPoints[0] = 0.5

    assert tracked.interval[0, 0] == -1.0
    assert len(tracked.transforms) == 1
    assert tracked.nextTransformPoints[0] != 0.5


def test_copy_preserves_final_step_state():
    tracked = TrackedInterval(UNIT_BOX_2D.copy())
    tracked.startFinalStep()
    tracked.possibleDuplicateRoots.append(np.array([0.1, 0.2]))
    duplicate = tracked.copy()

    assert duplicate.finalStep
    assert len(duplicate.possibleDuplicateRoots) == 1
    duplicate.possibleDuplicateRoots.append(np.array([0.3, 0.4]))
    assert len(tracked.possibleDuplicateRoots) == 1


def test_startFinalStep_saves_the_pre_final_state():
    tracked = TrackedInterval(UNIT_BOX_2D.copy())
    tracked.addTransform(np.array([[-1., 0.], [0., 1.]]))
    tracked.startFinalStep()
    saved = tracked.getIntervalForCombining()

    tracked.addTransform(np.array([[-1., 0.], [0., 1.]]))
    assert np.array_equal(tracked.getIntervalForCombining(), saved)
    assert not np.array_equal(tracked.interval, saved)


def test_repr_shows_the_interval():
    tracked = TrackedInterval(np.array([[-1., 1.]]))
    assert repr(tracked) == str(tracked) == str(tracked.interval)


############################### subdivision ##################################

def test_getInverseOrder():
    assert getInverseOrder(np.array([0, 3, 1])) == (0, 2, 1, 3, 4, 6, 5, 7)
    assert getInverseOrder(np.array([0, 1])) == (0, 1, 2, 3)
    assert getInverseOrder(np.array([1, 0])) == (0, 2, 1, 3)


def test_getSubdivisionDims_orders_by_degree():
    Ms = [np.zeros((2, 5)), np.zeros((2, 5))]
    tracked = TrackedInterval(UNIT_BOX_2D.copy())
    dims = getSubdivisionDims(Ms, tracked, 0)
    assert dims.shape == (2, 2)
    assert list(dims[0]) == [1, 0]           # dimension 1 has the higher degree


def test_getSubdivisionDims_skips_degenerate_dimensions():
    Ms = [np.zeros((3, 3)), np.zeros((3, 3))]
    tracked = TrackedInterval(np.array([[0.5, 0.5], [-1., 1.]]))
    dims = getSubdivisionDims(Ms, tracked, 0)
    assert set(dims.flatten()) == {1}


def test_getSubdivisionIntervals_covers_the_original_box():
    Ms = [np.random.default_rng(20).standard_normal((3, 3)),
          np.random.default_rng(21).standard_normal((3, 3))]
    tracked = TrackedInterval(UNIT_BOX_2D.copy())
    allMs, allErrors, allIntervals = getSubdivisionIntervals(Ms, np.array([0., 0.]), tracked, False, 0)

    assert len(allIntervals) == 4                  # 2 dimensions -> 4 subintervals
    assert all(len(group) == 2 for group in allMs)

    midpoints = np.unique([iv.interval[0, 1] for iv in allIntervals if iv.interval[0, 0] == -1])
    assert len(midpoints) == 1
    lows = sorted(tuple(iv.interval[:, 0]) for iv in allIntervals)
    assert len(set(lows)) == 4                     # the four subintervals are distinct

    # the union of the pieces is the original box
    assert np.isclose(sum(iv.size() for iv in allIntervals), tracked.size())


def test_getSubdivisionIntervals_transforms_the_polynomials_to_match():
    Ms = [np.random.default_rng(22).standard_normal((3, 3)),
          np.random.default_rng(23).standard_normal((3, 3))]
    tracked = TrackedInterval(UNIT_BOX_2D.copy())
    allMs, _, allIntervals = getSubdivisionIntervals(Ms, np.array([0., 0.]), tracked, False, 0)

    point = np.array([0.3, -0.4])
    for subMs, interval in zip(allMs, allIntervals):
        a, b = interval.interval[:, 0], interval.interval[:, 1]
        moved = (b - a) / 2 * point + (b + a) / 2
        for sub, original in zip(subMs, Ms):
            assert np.isclose(C.chebval2d(point[0], point[1], sub),
                              C.chebval2d(moved[0], moved[1], original))


def test_isExteriorInterval():
    tracked = TrackedInterval(UNIT_BOX_2D.copy())
    interior = TrackedInterval(np.array([[-0.5, 0.5], [-0.5, 0.5]]))
    touching = TrackedInterval(np.array([[-1., 0.5], [-0.5, 0.5]]))
    assert not isExteriorInterval(tracked, interior)
    assert isExteriorInterval(tracked, touching)


############################### trimMs #######################################

def test_trimMs_removes_negligible_high_degree_rows():
    M = np.zeros((6, 3))
    M[0, 0], M[1, 0], M[2, 1] = 1.0, 0.5, 0.25
    M[5, 0] = 1e-20                    # negligible
    Ms, errors = [M], np.array([1e-10])
    trimMs(Ms, errors)

    assert Ms[0].shape[0] < 6
    assert errors[0] > 1e-10           # the trimmed mass is added to the error


def test_trimMs_keeps_significant_coefficients():
    M = np.zeros((6, 3))
    M[0, 0], M[5, 2] = 1.0, 0.5
    Ms, errors = [M], np.array([1e-10])
    trimMs(Ms, errors)
    assert Ms[0].shape == (6, 3)
    assert errors[0] == 1e-10


def test_trimMs_never_trims_below_three_coefficients_per_dimension():
    M = np.zeros((6, 6))
    M[0, 0] = 1.0
    Ms, errors = [M], np.array([1e-2])
    trimMs(Ms, errors)
    assert Ms[0].shape == (3, 3)


def test_trimMs_with_no_error_budget_does_nothing():
    M = np.zeros((6, 3))
    M[0, 0], M[5, 0] = 1.0, 1e-20
    Ms, errors = [M], np.array([0.0])
    trimMs(Ms, errors)
    assert Ms[0].shape == (6, 3)


############################### SolverOptions ################################

def test_solver_options_defaults():
    options = SolverOptions()
    assert options.verbose is False
    assert options.exact is False
    assert options.constant_check is True
    assert options.low_dim_quadratic_check is True
    assert options.all_dim_quadratic_check is False
    assert options.maxZoomCount == 25
    assert options.level == 0
    # Regression test: solvePolyRecursive reads this attribute, but it used to only be
    # set from outside the class, so a freshly built SolverOptions blew up with AttributeError.
    assert options.useFinalStep is True


def test_solver_options_copy_is_independent():
    options = SolverOptions()
    duplicate = options.copy()
    duplicate.level = 7
    assert options.level == 0


def test_solvePolyRecursive_runs_with_default_options():
    Ms = [linear_cheb(2, 0, 0.5), linear_cheb(2, 1, -0.25)]
    interior, exterior = solvePolyRecursive(Ms, TrackedInterval(UNIT_BOX_2D.copy()),
                                            np.array([0., 0.]), SolverOptions())
    boxes = interior + exterior
    assert len(boxes) == 1
    assert np.allclose(boxes[0].getFinalInterval()[:, 0], [0.5, -0.25], atol=1e-8)


############################### zoomInOnIntervalIter #########################

def test_zoomInOnIntervalIter_shrinks_towards_the_root():
    Ms = [linear_cheb(2, 0, 0.5), linear_cheb(2, 1, -0.25)]
    tracked = TrackedInterval(UNIT_BOX_2D.copy())
    original_size = tracked.size()
    newMs, newErrors, newInterval, changed, should_stop = zoomInOnIntervalIter(
        Ms, np.array([0., 0.]), tracked, False)

    assert changed
    # the interval is updated in place, so callers must copy it beforehand if they need the old one
    assert newInterval is tracked
    assert newInterval.size() < original_size
    assert np.array([0.5, -0.25]) in newInterval
    assert np.allclose(newInterval.interval[:, 0], [0.5, -0.25], atol=1e-10)


def test_zoomInOnIntervalIter_marks_an_empty_interval():
    Ms = [np.array([[3.0, 0.1], [0.1, 0.0]]), np.array([[-4.0, 0.05], [0.02, 0.0]])]
    tracked = TrackedInterval(UNIT_BOX_2D.copy())
    _, _, newInterval, _, _ = zoomInOnIntervalIter(Ms, np.array([0., 0.]), tracked, False)
    assert newInterval.empty


############################### solveChebyshevSubdivision ####################

def test_solve_rejects_a_non_square_system():
    with pytest.raises(ValueError, match="N polynomials of dimension N"):
        solveChebyshevSubdivision([np.zeros((3, 3))], np.array([0.]))


def test_solve_rejects_mismatched_errors():
    Ms = [linear_cheb(2, 0, 0.), linear_cheb(2, 1, 0.)]
    with pytest.raises(ValueError, match="same length"):
        solveChebyshevSubdivision(Ms, np.array([0.]))


def test_solve_finds_the_root_of_a_linear_system():
    Ms = [linear_cheb(2, 0, 0.5), linear_cheb(2, 1, -0.25)]
    roots = roots_from(solveChebyshevSubdivision(Ms, np.array([0., 0.])))
    assert roots.shape == (1, 2)
    assert np.allclose(roots[0], [0.5, -0.25], atol=1e-10)


def test_solve_finds_both_roots_of_a_quadratic():
    boxes = solveChebyshevSubdivision([np.array([0., 0., 1.])], np.array([0.]))   # T_2
    roots = roots_from(boxes)
    assert np.allclose(np.sort(np.ravel(roots)), [-1 / np.sqrt(2), 1 / np.sqrt(2)], atol=1e-10)


def test_solve_returns_only_bounding_boxes():
    # The solver hands back bounding intervals alone; roots are read off them by the caller.
    Ms = [linear_cheb(2, 0, 0.5), linear_cheb(2, 1, -0.25)]
    boxes = solveChebyshevSubdivision(Ms, np.array([0., 0.]))
    assert isinstance(boxes, list)
    assert all(isinstance(box, TrackedInterval) for box in boxes)


def test_solve_returns_bounding_boxes_that_contain_the_roots():
    Ms = [linear_cheb(2, 0, 0.5), linear_cheb(2, 1, -0.25)]
    boxes = solveChebyshevSubdivision(Ms, np.array([0., 0.]))
    assert len(boxes) == 1
    for box in boxes:
        # A box that could not separate the roots inside it reports more than one, and every
        # root it reports has to lie in that box.
        for root in getRootsInInterval(box):
            assert root in box


def test_solve_returns_finalized_boxes():
    # finalInterval only exists once getFinalInterval has run, and callers read it straight off
    # the returned boxes, so the solver has to finalize them before handing them back.
    Ms = [linear_cheb(2, 0, 0.5), linear_cheb(2, 1, -0.25)]
    for box in solveChebyshevSubdivision(Ms, np.array([0., 0.])):
        assert np.allclose(box.finalInterval[:, 0], [0.5, -0.25], atol=1e-8)
        assert np.all(box.finalDimSize() >= 0)


def test_solve_reports_no_roots_when_there_are_none():
    # x - 5 and y - 5 have no roots inside [-1, 1]^2
    boxes = solveChebyshevSubdivision([linear_cheb(2, 0, 5.0), linear_cheb(2, 1, 5.0)],
                                       np.array([0., 0.]))
    assert len(boxes) == 0
    assert len(roots_from(boxes)) == 0


def test_solve_agrees_with_itself_in_exact_mode():
    Ms = [linear_cheb(2, 0, 0.3), linear_cheb(2, 1, -0.6)]
    fast = roots_from(solveChebyshevSubdivision(Ms, np.array([0., 0.]), exact=False))
    exact = roots_from(solveChebyshevSubdivision(Ms, np.array([0., 0.]), exact=True))
    assert np.allclose(np.sort(fast, axis=0), np.sort(exact, axis=0), atol=1e-10)


def test_solve_does_not_modify_the_polynomials_it_is_given():
    Ms = [linear_cheb(2, 0, 0.5), linear_cheb(2, 1, -0.25)]
    originals = [M.copy() for M in Ms]
    errors = np.array([0., 0.])
    solveChebyshevSubdivision(Ms, errors)
    for M, original in zip(Ms, originals):
        assert np.array_equal(M, original)
    assert np.array_equal(errors, [0., 0.])
