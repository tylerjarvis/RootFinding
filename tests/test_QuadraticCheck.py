"""Unit tests for yroots.QuadraticCheck.

The quadratic check is a one sided test: returning True is a guarantee that the
polynomial (together with its error bound) has no root in [-1,1]^n, while returning
False means nothing at all. The tests below therefore focus on soundness -- whenever
the check certifies an interval, the polynomial really is bounded away from zero.
"""
import numpy as np
import pytest
from numpy.polynomial import chebyshev as C

from yroots.QuadraticCheck import (get_fixed_vars, quadratic_check, quadratic_check_2D,
                                   quadratic_check_3D, quadratic_check_nd)

TOL = 1e-8


def grid_min_abs_2d(coeff, n=60):
    g = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(g, g, indexing='ij')
    return np.min(np.abs(C.chebval2d(X, Y, coeff)))


def grid_min_abs_3d(coeff, n=21):
    g = np.linspace(-1, 1, n)
    X, Y, Z = np.meshgrid(g, g, g, indexing='ij')
    return np.min(np.abs(C.chebval3d(X, Y, Z, coeff)))


############################### get_fixed_vars ###############################

def test_get_fixed_vars():
    assert get_fixed_vars(2) == [(0,), (1,)]
    assert get_fixed_vars(3) == [(0, 1), (0, 2), (1, 2), (0,), (1,), (2,)]


def test_get_fixed_vars_excludes_corners_and_the_interior():
    for dim in (2, 3, 4):
        subsets = get_fixed_vars(dim)
        assert all(0 < len(s) < dim for s in subsets)      # never all fixed, never none fixed
        assert len(subsets) == len(set(subsets))           # no duplicates


############################### dispatch #####################################

def test_quadratic_check_dispatches_on_dimension():
    coeff2 = np.zeros((3, 3))
    coeff2[0, 0] = 10.0
    assert quadratic_check(coeff2, TOL) == quadratic_check_2D(coeff2, TOL)

    coeff3 = np.zeros((3, 3, 3))
    coeff3[0, 0, 0] = 10.0
    assert quadratic_check(coeff3, TOL) == quadratic_check_3D(coeff3, TOL)

    assert quadratic_check(coeff2, TOL, nd_check=True) == quadratic_check_nd(coeff2.copy(), TOL)


def test_dimension_specific_checks_refuse_the_wrong_dimension():
    assert quadratic_check_2D(np.ones((3, 3, 3)), TOL) is False
    assert quadratic_check_3D(np.ones((3, 3)), TOL) is False


def test_quadratic_check_handles_one_dimensional_input():
    constant = np.array([5.0, 0.1, 0.1])
    assert quadratic_check(constant, TOL)                     # 5 + small terms is never 0
    assert not quadratic_check(np.array([0.0, 1.0, 0.0]), TOL)  # x has a root at 0


############################### certification ################################

def test_certifies_a_dominant_constant_term_2d():
    coeff = np.zeros((3, 3))
    coeff[0, 0], coeff[1, 1], coeff[2, 0] = 10.0, 0.5, 0.3
    assert quadratic_check_2D(coeff, TOL)
    assert grid_min_abs_2d(coeff) > TOL


def test_certifies_a_dominant_constant_term_3d():
    coeff = np.zeros((3, 3, 3))
    coeff[0, 0, 0], coeff[1, 0, 0], coeff[0, 1, 1] = 10.0, 1.0, 0.5
    coeff[2, 0, 0], coeff[1, 1, 1] = 0.4, 0.05
    assert quadratic_check_3D(coeff, TOL)
    assert grid_min_abs_3d(coeff) > TOL


def test_certifies_a_dominant_constant_term_in_higher_dimensions():
    coeff = np.zeros((3, 3, 3, 3))
    coeff[0, 0, 0, 0], coeff[1, 0, 0, 0], coeff[0, 0, 1, 1] = 8.0, 0.5, 0.25
    assert quadratic_check_nd(coeff, TOL)


############################### soundness ####################################

def test_does_not_certify_polynomials_with_obvious_roots_2d():
    x = np.zeros((3, 3))
    x[1, 0] = 1.0                                   # f = x, root along x = 0
    assert not quadratic_check_2D(x, TOL)

    circle = np.zeros((3, 3))
    circle[0, 0], circle[2, 0], circle[0, 2] = -0.25, 0.5, 0.5     # x^2 + y^2 - 1.25
    assert not quadratic_check_2D(circle, TOL)

    saddle = np.zeros((3, 3))
    saddle[1, 1] = 1.0                              # f = xy
    assert not quadratic_check_2D(saddle, TOL)


def test_does_not_certify_polynomials_with_obvious_roots_3d():
    z = np.zeros((3, 3, 3))
    z[0, 0, 1] = 1.0
    assert not quadratic_check_3D(z, TOL)

    sphere = np.zeros((3, 3, 3))
    sphere[0, 0, 0] = -0.5
    sphere[2, 0, 0] = sphere[0, 2, 0] = sphere[0, 0, 2] = 0.5
    assert not quadratic_check_3D(sphere, TOL)


def test_does_not_certify_a_constant_that_is_smaller_than_the_tolerance():
    coeff = np.zeros((3, 3))
    coeff[0, 0] = 1e-12
    assert not quadratic_check_2D(coeff, TOL)


@pytest.mark.parametrize("seed", range(6))
def test_random_2d_certifications_are_sound(seed):
    rng = np.random.default_rng(seed)
    certified = 0
    for _ in range(50):
        coeff = rng.standard_normal((3, 3)) * rng.choice([1.0, 0.3, 3.0])
        coeff[0, 0] += rng.choice([0.0, 3.0, -3.0])
        if quadratic_check_2D(coeff, TOL):
            certified += 1
            # certifying means |f| > tol everywhere in the box
            assert grid_min_abs_2d(coeff) > TOL
    assert certified > 0        # the sweep actually exercised the certifying branch


@pytest.mark.parametrize("seed", range(3))
def test_random_3d_certifications_are_sound(seed):
    rng = np.random.default_rng(100 + seed)
    certified = 0
    for _ in range(40):
        coeff = np.zeros((3, 3, 3))
        for spot in rng.integers(0, 3, (6, 3)):
            coeff[tuple(spot)] = rng.standard_normal() * 0.4
        coeff[0, 0, 0] = rng.choice([0.0, 3.0, -3.0]) + 0.2 * rng.standard_normal()
        if quadratic_check_3D(coeff, TOL):
            certified += 1
            assert grid_min_abs_3d(coeff) > TOL
    assert certified > 0


############################### consistency ##################################

@pytest.mark.parametrize("seed", range(4))
def test_2d_and_nd_checks_agree(seed):
    rng = np.random.default_rng(200 + seed)
    for _ in range(40):
        coeff = rng.standard_normal((3, 3))
        coeff[0, 0] += rng.choice([0.0, 3.0])
        assert quadratic_check_2D(coeff, TOL) == quadratic_check_nd(coeff.copy(), TOL)


@pytest.mark.parametrize("seed", range(3))
def test_3d_and_nd_checks_agree(seed):
    rng = np.random.default_rng(300 + seed)
    for _ in range(25):
        coeff = np.zeros((3, 3, 3))
        for spot in rng.integers(0, 3, (6, 3)):
            coeff[tuple(spot)] = rng.standard_normal() * 0.4
        coeff[0, 0, 0] = rng.choice([0.0, 3.0, -3.0])
        assert quadratic_check_3D(coeff, TOL) == quadratic_check_nd(coeff.copy(), TOL)


def test_a_larger_tolerance_never_certifies_more():
    rng = np.random.default_rng(400)
    for _ in range(60):
        coeff = rng.standard_normal((3, 3))
        coeff[0, 0] += 3.0
        if quadratic_check_2D(coeff, 1.0):
            assert quadratic_check_2D(coeff, 1e-12)


############################### input handling ###############################

def test_checks_do_not_modify_their_input():
    for coeff in (np.ones((3, 3)), np.ones((3, 3, 3)), np.ones((3, 3, 3, 3))):
        original = coeff.copy()
        quadratic_check(coeff, TOL)
        quadratic_check_nd(coeff, TOL)
        assert np.array_equal(coeff, original)


def test_checks_handle_tensors_smaller_than_the_quadratic_part():
    small = np.array([[5.0, 0.1], [0.1, 0.0]])      # only degree 1 terms are present
    assert quadratic_check_2D(small, TOL)
    assert quadratic_check_nd(small, TOL)

    tiny = np.array([[7.0]])
    assert quadratic_check_2D(tiny, TOL)
    assert quadratic_check_nd(tiny, TOL)


def test_checks_handle_tensors_larger_than_the_quadratic_part():
    coeff = np.zeros((5, 5))
    coeff[0, 0], coeff[4, 4] = 10.0, 0.5
    assert quadratic_check_2D(coeff, TOL)
    assert quadratic_check_nd(coeff, TOL)

    coeff[4, 4] = 20.0                              # now the higher order part can dominate
    assert not quadratic_check_2D(coeff, TOL)
