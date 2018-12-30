import numpy as np
from yroots.IntervalChecks import constant_term_check, full_quad_check, full_cubic_check, curvature_check, \
linear_check, quadratic_check
from yroots.polynomial import MultiCheb,MultiPower

def test_zero_check2D():
    interval_checks = [constant_term_check,full_quad_check, curvature_check, full_cubic_check]
    subinterval_checks = [linear_check,quadratic_check]
    a = -np.ones(2)
    b = np.ones(2)
    tol = 1.e-4
    interval_checks.extend([lambda x, tol: f(x, [(a,b)],[False], tol)[0]  for f in subinterval_checks])

    for method in interval_checks:
        # this function barely does not have a zero
        # may return true or false
        c = np.array([
        [ 7.5,  2, -2, -1],
        [ 2,  2,  1,  0],
        [-1,  1,  0,  0],
        [ 1,  0,  0,  0]
        ])
        # assert can_eliminate(c, a, b) is not True

        # this function obviously does not have a zero
        # must return false
        c = np.array([
        [ 100,  2, -2, -1],
        [ 2,  2,  1,  0],
        [-1,  1,  0,  0],
        [ 1,  0,  0,  0]
        ])
        assert method(c,tol) == False

        # has zeros, must return true
        c = np.array([
        [ 6.5,  2, -2, -1],
        [ 2,  2,  1,  0],
        [-1,  1,  0,  0],
        [ 1,  0,  0,  0]
        ])
        assert method(c,tol) == True

        # has zeros, must return true
        c = np.array([
        [1.019, .2,   .5],
        [0.2,   .001, 0],
        [0.5,  0,     0],
        ])
        assert method(c,tol) == True

        # has zeros, must return true
        c = np.array([
        [-1.3, .2,   0.5],
        [.2,   .001, 0],
        [.5,  0,     0],
        ])
        assert method(c,tol) == True

        # has zeros, must return true
        c = np.array([
        [-1.6, .2,   .5, .1],
        [.2,   .001, .1, 0],
        [0.5,  .1,    0, 0],
        [0.1,   0,    0, 0]
        ])
        assert method(c,tol) == True

    if __name__ == "__main__":
        test_zero_check2D()
