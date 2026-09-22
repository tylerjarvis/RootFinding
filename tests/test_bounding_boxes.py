"""Tests that solve(..., returnBoundingBoxes=True) pairs each root with its own box.

Regression tests for the misalignment where a bounding interval holding several roots
the solver could not separate contributed one box but several roots, so the two
returned lists went out of step: every root after the first duplicate was paired with
the previous root's box, and the trailing roots had no box at all.

These tests deliberately assert `len(roots) == len(boxes)` *before* checking
containment. A check written as `for root, box in zip(roots, boxes)` cannot catch this
bug: zip stops at the shorter list, so the unpaired roots are skipped and the shifted
pairs are the only thing left to look at. Length first, then containment.
"""
import numpy as np
import pytest

import yroots as yr
import yroots.ChebyshevSubdivisionSolver as ChebyshevSubdivisionSolver


def assert_boxes_match_roots(roots, boxes, funcs=None, tol=1e-12):
    """Assert every root has its own bounding box and lies inside it.

    Parameters
    ----------
    roots : numpy array
        Roots as returned by ``solve``.
    boxes : numpy array
        Bounding boxes as returned by ``solve(..., returnBoundingBoxes=True)``.
    funcs : list of functions, optional
        If given, also assert each function is within ``tol`` of zero at every root,
        so a test cannot pass by returning aligned-but-wrong output.
    tol : float
        Residual tolerance for ``funcs``.
    """
    roots = np.atleast_2d(np.asarray(roots, dtype=float))
    boxes = np.asarray(boxes)

    # Length first. Containment alone is vacuous when boxes is the shorter list.
    assert len(boxes) == len(roots), (
        f"{len(roots)} roots but {len(boxes)} bounding boxes: the returned lists are "
        "not index-aligned, so roots[i] does not correspond to boxes[i]"
    )

    for i in range(len(roots)):
        box = ChebyshevSubdivisionSolver.TrackedInterval(np.asarray(boxes[i]))
        assert roots[i] in box, (
            f"root {i} at {roots[i].tolist()} does not lie in its own box "
            f"{np.asarray(boxes[i]).tolist()}"
        )

    if funcs is not None:
        coords = [roots[:, k] for k in range(roots.shape[1])]
        for j, f in enumerate(funcs):
            resid = np.max(np.abs(f(*coords)))
            assert resid < tol, f"function {j} has residual {resid:.3e} at the roots"


def test_boxes_match_roots_simple():
    """Control: a system with only simple, well-separated roots."""
    f = lambda x, y: np.sin(4*(x + y/10 + np.pi/10))
    g = lambda x, y: np.cos(2*(x - 2*y + np.pi/7))
    a, b = np.array([-1., -1.]), np.array([1., 1.])

    roots, boxes = yr.solve([f, g], a, b, returnBoundingBoxes=True)

    assert len(roots) > 0
    assert_boxes_match_roots(roots, boxes, funcs=[f, g], tol=1e-13)


def test_boxes_match_roots_singular_root():
    """A singular (double) root at the origin.

    The solver cannot separate the two coincident roots here, so it reports both out
    of a single bounding interval. That interval must be listed once per root. Before
    the fix this returned 6 roots against 5 boxes, with only 2 roots in their own box.

    The root count is deliberately not asserted: whether the double root is reported
    once or twice is a separate open question, and this test must keep checking
    alignment either way.
    """
    f = lambda x, y: (y - 2*x)*(y + 0.5*x)
    g = lambda x, y: x*(x**2 + y**2 - 1)
    a, b = np.array([-1., -1.]), np.array([1., 1.])

    roots, boxes = yr.solve([f, g], a, b, returnBoundingBoxes=True)

    assert len(roots) > 0
    assert_boxes_match_roots(roots, boxes, funcs=[f, g], tol=1e-12)


def test_boxes_match_roots_several_duplicate_boxes():
    """Two separate intervals each hold roots the solver could not separate.

    Exercises the case where the misalignment compounds: before the fix this returned
    10 roots against 8 boxes, and only 1 of the 8 pairs was correct.
    """
    f = lambda x, y: y**4 + (-1)*y**3 + (2*x**2)*(y**2) + (3*x**2)*y + (x**4)
    h = lambda x, y: y**10 - 2*(x**8)*(y**2) + 4*(x**4)*y - 2
    g = lambda x, y: h(2*x, 2*(y + .5))
    a, b = np.array([-1., -1.]), np.array([1., 1.])

    roots, boxes = yr.solve([f, g], a, b, returnBoundingBoxes=True)

    assert len(roots) > 0
    assert_boxes_match_roots(roots, boxes, funcs=[f, g], tol=1e-9)


def test_boxes_match_roots_3d():
    """The singular root case in three dimensions."""
    f = lambda x, y, z: (y - 2*x)*(y + 0.5*x)
    g = lambda x, y, z: x*(x**2 + y**2 - 1)
    h = lambda x, y, z: z - 0.5
    a, b = np.array([-1., -1., -1.]), np.array([1., 1., 1.])

    roots, boxes = yr.solve([f, g, h], a, b, returnBoundingBoxes=True)

    assert len(roots) > 0
    assert_boxes_match_roots(roots, boxes, funcs=[f, g, h], tol=1e-12)


def test_boxes_match_roots_4d_two_singularities():
    """Two independent singular roots in four dimensions.

    The degeneracies multiply, so this is the worst case found: before the fix it
    returned 36 roots against 25 boxes, with only 2 roots in their own box.
    """
    f = lambda x, y, z, w: (y - 2*x)*(y + 0.5*x)
    g = lambda x, y, z, w: x*(x**2 + y**2 - 1)
    h = lambda x, y, z, w: (w - 2*z)*(w + 0.5*z)
    k = lambda x, y, z, w: z*(z**2 + w**2 - 1)
    a, b = np.array([-1.]*4), np.array([1.]*4)

    roots, boxes = yr.solve([f, g, h, k], a, b, returnBoundingBoxes=True)

    assert len(roots) > 0
    assert_boxes_match_roots(roots, boxes, funcs=[f, g, h, k], tol=1e-12)


def test_boxes_match_roots_outside_unit_box():
    """Alignment must survive the transform back from a non-unit search interval."""
    f = lambda x, y: 2*x*y*np.cos(y**2)*np.cos(2*x) - np.cos(x*y)
    g = lambda x, y: 2*np.sin(x*y**2)*np.sin(3*x*y) - np.sin(x*y)
    a, b = np.array([-2., -2.]), np.array([2., 2.])

    roots, boxes = yr.solve([f, g], a, b, returnBoundingBoxes=True)

    assert len(roots) > 0
    assert_boxes_match_roots(roots, boxes, funcs=[f, g], tol=1e-13)
