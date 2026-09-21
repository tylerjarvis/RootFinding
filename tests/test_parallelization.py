"""
Unit tests for the yroots algorithm.

For each test case the suite verifies:
  1. Serial yroots   – roots match the polished (ground-truth) roots
  2. Parallel yroots – roots match the polished (ground-truth) roots
  3. Serial vs parallel – both runs agree with each other

Comparison helpers (ported from the original test suite)
---------------------------------------------------------
  norm_pass_or_fail      – sorted-norm difference on x and y columns
  residuals              – |f(roots)| at each root
  residuals_pass_or_fail – max residual within tolerance
"""

import os
import numpy as np
import pytest
from yroots import solve
import time

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EPS          = 2.220446049250313e-16
DEFAULT_TOL  = 10000 * EPS          # ~2.22e-12

MAX_CPU        = 4
PARALLEL_DEPTH = 2

# Cases used elsewhere for stress tests that need systems heavy enough for the
# parallel driver to actually exercise the scheduling paths.
HEAVY_CASE_IDS = {"2.4", "2.5", "3.2", "4.2"}

# Cases used for the parallel-determinism stress test. Chosen for having many
# roots and wide subdivision, so the parallel driver actually exercises the
# scheduling paths (a case that solves without subdividing tells us nothing).
DETERMINISM_CASE_ID = "2.5"
DETERMINISM_ITERATIONS = 20

# Worker-count values swept in the max_cpu-sweep agreement test. Every value
# must produce the same root set as max_cpu=1, otherwise a specific worker
# count is influencing the answer instead of just when it arrives.
MAX_CPU_SWEEP = sorted({1, 2, 4, min(8, os.cpu_count() or 4)})

POLISHED_DIR = os.path.join(os.path.dirname(__file__), "./Polished_results")


# ---------------------------------------------------------------------------
# Helpers (ported from original test suite)
# ---------------------------------------------------------------------------

def load_polished(test_num):
    """Load polished roots from ./Polished_results/polished_{test_num}.npy"""
    path = os.path.join(POLISHED_DIR, f"polished_{test_num}.npy")
    roots = np.load(path)
    if roots.ndim == 1:
        roots = roots.reshape(1, -1)
    return roots


def norm_pass_or_fail(yroots, roots, tol=DEFAULT_TOL):
    """
    Sort both root arrays and compare column-wise norms.
    Returns (passed, x_norm, y_norm).
    """
    roots_sorted  = np.sort(roots,  axis=0)
    yroots_sorted = np.sort(yroots, axis=0)
    diff          = roots_sorted - yroots_sorted
    x_norm        = np.linalg.norm(diff[:, 0])
    y_norm        = np.linalg.norm(diff[:, 1])
    return x_norm < tol and y_norm < tol, x_norm, y_norm


def collapse_duplicate_roots(roots, tol):
    """Collapse roots that agree to within tol.

    A singular root can be reached from more than one recursion branch, so the solver
    reports it once per branch while the polished ground truth lists it a single time.
    Collapsing here lets the norm comparison line up against that ground truth.
    """
    collapsed = []
    for root in roots:
        if not any(np.linalg.norm(root - kept) < tol for kept in collapsed):
            collapsed.append(root)
    return np.array(collapsed)


def residuals(func, roots):
    """Absolute residuals of func at each root."""
    return np.abs(func(roots[:, 0], roots[:, 1]))


def residuals_pass_or_fail(funcs, roots, tol=DEFAULT_TOL):
    """True if max residual of every func is within tol."""
    for func in funcs:
        if np.max(residuals(func, roots)) > tol:
            return False
    return True


def run_serial(tc):
    return solve([tc["f"], tc["g"]], tc["a_min"], tc["a_max"], exact=True)


def run_parallel(tc):
    return solve([tc["f"], tc["g"]], tc["a_min"], tc["a_max"], exact=True, max_cpu=MAX_CPU, parallel_depth=PARALLEL_DEPTH)


# ---------------------------------------------------------------------------
# Test-case definitions
# ---------------------------------------------------------------------------

TEST_CASES = [
    dict(
        id    = "1.1",
        desc  = "Test 1.1 – degree-5 system, 4 roots",
        f     = lambda x, y: 144*(x**4 + y**4) - 225*(x**2 + y**2) + 350*x**2*y**2 + 81,
        g     = lambda x, y: y - x**6,
        a_min = [-1, -1],
        a_max = [ 1,  1],
        tol   = DEFAULT_TOL,
    ),
    dict(
        id    = "1.2",
        desc  = "Test 1.2 – degree-10 system, 13 roots",
        f     = lambda x, y: (
                    (y**2 - x**3) *
                    ((y - 0.7)**2 - (x - 0.3)**3) *
                    ((y + 0.2)**2 - (x + 0.8)**3) *
                    ((y + 0.2)**2 - (x - 0.8)**3)
                ),
        g     = lambda x, y: (
                    ((y + .4)**3  - (x - .4)**2) *
                    ((y + .3)**3  - (x - .3)**2) *
                    ((y - .5)**3  - (x + .6)**2) *
                    ((y + 0.3)**3 - (2*x - 0.8)**3)
                ),
        a_min = [-1, -1],
        a_max = [ 1,  1],
        tol   = 2.220446049250313e-7,
    ),
    dict(
        id    = "1.3",
        desc  = "Test 1.3 – cusp system, 5 roots",
        f     = lambda x, y: y**2 - x**3,
        g     = lambda x, y: (y + .1)**3 - (x - .1)**2,
        a_min = [-1, -1],
        a_max = [ 1,  1],
        tol   = DEFAULT_TOL,
    ),
    dict(
        id    = "1.4",
        desc  = "Test 1.4 – linear system, 1 root",
        f     = lambda x, y: x - y + .5,
        g     = lambda x, y: x + y,
        a_min = [-1, -1],
        a_max = [ 1,  1],
        tol   = DEFAULT_TOL,
    ),
    dict(
        id    = "1.5",
        desc  = "Test 1.5 – linear system, 1 root",
        f     = lambda x, y: y + x/2 + 1/10,
        g     = lambda x, y: y - 2.1*x + 2,
        a_min = [-1, -1],
        a_max = [ 1,  1],
        tol   = DEFAULT_TOL,
    ),
    dict(
        id    = "2.1",
        desc  = "Test 2.1 – cos/parabola system, 6 roots",
        f     = lambda x, y: np.cos(10*x*y),
        g     = lambda x, y: x + y**2,
        a_min = [-1, -1],
        a_max = [ 1,  1],
        tol   = DEFAULT_TOL,
    ),
    dict(
        id    = "2.2",
        desc  = "Test 2.2 – near-tangent circle, 2 roots",
        f     = lambda x, y: x,
        g     = lambda x, y: (x - 0.9999)**2 + y**2 - 1,
        a_min = [-1, -1],
        a_max = [ 1,  1],
        tol   = DEFAULT_TOL,
    ),
    dict(
        id    = "2.3",
        desc  = "Test 2.3 – sin/cos trig system, 5 roots",
        f     = lambda x, y: np.sin(4*(x + y/10 + np.pi/10)),
        g     = lambda x, y: np.cos(2*(x - 2*y + np.pi/7)),
        a_min = [-1, -1],
        a_max = [ 1,  1],
        tol   = DEFAULT_TOL,
    ),
    dict(
        id    = "2.4",
        desc  = "Test 2.4 – exp/sin system, 93 roots",
        f     = lambda x, y: np.exp(x - 2*x**2 - y**2) * np.sin(10*(x + y + x*y**2)),
        g     = lambda x, y: np.exp(-x + 2*y**2 + x*y**2) * np.sin(10*(x - y - 2*x*y**2)),
        a_min = [-1, -1],
        a_max = [ 1,  1],
        tol   = DEFAULT_TOL,
    ),
    dict(
        id    = "2.5",
        desc  = "Test 2.5 – trig system, 103 roots",
        f     = lambda x, y: 2*y*np.cos(y**2)*np.cos(2*x) - np.cos(y),
        g     = lambda x, y: 2*np.sin(y**2)*np.sin(2*x) - np.sin(x),
        a_min = [-4, -4],
        a_max = [ 4,  4],
        tol   = 2.220446049250313e-12,
    ),
    dict(
        id    = "3.1",
        desc  = "Test 3.1 – ellipse/circle system, 4 roots",
        f     = lambda x, y: (x - .3)**2 + 2*(y + 0.3)**2 - 1,
        g     = lambda x, y: (
                    ((x - .49)**2 + (y + .5)**2  - 1) *
                    ((x + 0.5)**2 + (y + 0.5)**2 - 1) *
                    ((x - 1)**2   + (y - 0.5)**2 - 1)
                ),
        a_min = [-1, -1],
        a_max = [ 1,  1],
        # Nearly tangent conics, so the roots are located far less accurately than the
        # residuals suggest. Measured column-norm error 1.8e-11. The previous
        # 2.220446049250313e-11 passes, but leaves only 1.2x headroom -- by far the
        # thinnest in this file, where every other case has 2.3x or more.
        tol   = 1e-10,
    ),
    dict(
        id    = "3.2",
        desc  = "Test 3.2 – product of ellipses, 45 roots",
        f     = lambda x, y: (
                    ((x - 0.1)**2  + 2*(y - 0.1)**2  - 1) *
                    ((x + 0.3)**2  + 2*(y - 0.2)**2  - 1) *
                    ((x - 0.3)**2  + 2*(y + 0.15)**2 - 1) *
                    ((x - 0.13)**2 + 2*(y + 0.15)**2 - 1)
                ),
        g     = lambda x, y: (
                    (2*(x + 0.1)**2 + (y + 0.1)**2  - 1) *
                    (2*(x + 0.1)**2 + (y - 0.1)**2  - 1) *
                    (2*(x - 0.3)**2 + (y - 0.15)**2 - 1) *
                    ((x - 0.21)**2  + 2*(y - 0.15)**2 - 1)
                ),
        a_min = [-1, -1],
        a_max = [ 1,  1],
        # The same tangency problem as 3.1, across 45 roots. Measured column-norm error
        # 9.5e-12, so this leaves 2.3x headroom -- now the thinnest margin in the file,
        # and the case most likely to fail spuriously on another machine or numpy build.
        tol   = 2.220446049250313e-11,
    ),
    dict(
        id    = "4.1",
        desc  = "Test 4.1 – sin system, 5 roots",
        f     = lambda x, y: np.sin(3*(x + y)),
        g     = lambda x, y: np.sin(3*(x - y)),
        a_min = [-1, -1],
        a_max = [ 1,  1],
        tol   = DEFAULT_TOL,
    ),
    dict(
        id    = "4.2",
        desc  = "Test 4.2 – high-degree polynomial system, 2 roots",
        f     = lambda x, y: (
                    90000*y**10 - 1440000*y**9 +
                    (360000*x**4 + 720000*x**3 + 504400*x**2 + 144400*x + 9971200)*y**8 +
                    (-4680000*x**4 - 9360000*x**3 - 6412800*x**2 - 1732800*x - 39554400)*y**7 +
                    (540000*x**8 + 2160000*x**7 + 3817600*x**6 + 3892800*x**5 + 27577600*x**4 +
                     51187200*x**3 + 34257600*x**2 + 8952800*x + 100084400)*y**6 +
                    (-5400000*x**8 - 21600000*x**7 - 37598400*x**6 - 37195200*x**5 - 95198400*x**4 -
                     153604800*x**3 - 100484000*x**2 - 26280800*x - 169378400)*y**5 +
                    (360000*x**12 + 2160000*x**11 + 6266400*x**10 + 11532000*x**9 + 34831200*x**8 +
                     93892800*x**7 + 148644800*x**6 + 141984000*x**5 + 206976800*x**4 + 275671200*x**3 +
                     176534800*x**2 + 48374000*x + 194042000)*y**4 +
                    (-2520000*x**12 - 15120000*x**11 - 42998400*x**10 - 76392000*x**9 - 128887200*x**8 -
                     223516800*x**7 - 300675200*x**6 - 274243200*x**5 - 284547200*x**4 - 303168000*x**3 -
                     190283200*x**2 - 57471200*x - 147677600)*y**3 +
                    (90000*x**16 + 720000*x**15 + 3097600*x**14 + 9083200*x**13 + 23934400*x**12 +
                     58284800*x**11 + 117148800*x**10 + 182149600*x**9 + 241101600*x**8 + 295968000*x**7 +
                     320782400*x**6 + 276224000*x**5 + 236601600*x**4 + 200510400*x**3 + 123359200*x**2 +
                     43175600*x + 70248800)*y**2 +
                    (-360000*x**16 - 2880000*x**15 - 11812800*x**14 - 32289600*x**13 - 66043200*x**12 -
                     107534400*x**11 - 148807200*x**10 - 184672800*x**9 - 205771200*x**8 - 196425600*x**7 -
                     166587200*x**6 - 135043200*x**5 - 107568800*x**4 - 73394400*x**3 - 44061600*x**2 -
                     18772000*x - 17896000)*y +
                    (144400*x**18 + 1299600*x**17 + 5269600*x**16 + 12699200*x**15 + 21632000*x**14 +
                     32289600*x**13 + 48149600*x**12 + 63997600*x**11 + 67834400*x**10 + 61884000*x**9 +
                     55708800*x**8 + 45478400*x**7 + 32775200*x**6 + 26766400*x**5 + 21309200*x**4 +
                     11185200*x**3 + 6242400*x**2 + 3465600*x + 1708800)
                ),
        g     = lambda x, y: 1e-4 * (
                    y**7 - 3*y**6 +
                    (2*x**2 - x + 2)*y**5 +
                    (x**3 - 6*x**2 + x + 2)*y**4 +
                    (x**4 - 2*x**3 + 2*x**2 + x - 3)*y**3 +
                    (2*x**5 - 3*x**4 + x**3 + 10*x**2 - x + 1)*y**2 +
                    (-x**5 + 3*x**4 + 4*x**3 - 12*x**2)*y +
                    (x**7 - 3*x**5 - x**4 - 4*x**3 + 4*x**2)
                ),
        a_min = [-1, -1],
        a_max = [ 1,  1],
        tol   = 1e-6,
    ),
    dict(
        id    = "5.1",
        desc  = "Test 5.1 – trig system, 10 roots",
        f     = lambda x, y: 2*x*y*np.cos(y**2)*np.cos(2*x) - np.cos(x*y),
        g     = lambda x, y: 2*np.sin(x*y**2)*np.sin(3*x*y) - np.sin(x*y),
        a_min = [-2, -2],
        a_max = [ 2,  2],
        tol   = DEFAULT_TOL,
    )
]

_ids = [tc["id"] for tc in TEST_CASES]


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture(params=TEST_CASES, ids=_ids)
def test_case(request):
    tc = request.param
    polished = load_polished(tc["id"])
    return {**tc, "polished": polished}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSerialRoots:
    """Serial solve() must match polished ground-truth roots."""

    def test_root_count(self, test_case):
        tc = test_case
        print(tc["polished"])
        try:
            roots = run_serial(tc)
        except RecursionError:
            pytest.fail(f"{tc['desc']}: serial solve() hit maximum recursion depth.")
        roots = np.atleast_2d(roots)
        expected = len(tc["polished"]) + tc.get("dup_roots", 0)
        assert len(roots) == expected, (
            f"{tc['desc']}: expected {expected} roots, "
            f"got {len(roots)} (serial)."
        )

    def test_norm(self, test_case):
        tc = test_case
        try:
            roots = run_serial(tc)
        except RecursionError:
            pytest.fail(f"{tc['desc']}: serial solve() hit maximum recursion depth.")
        roots = np.atleast_2d(roots)
        passed, x_norm, y_norm = norm_pass_or_fail(roots, tc["polished"], tol=tc["tol"])
        assert passed, (
            f"{tc['desc']} (serial): norm test failed. "
            f"x_norm={x_norm:.2e}, y_norm={y_norm:.2e} (tol={tc['tol']:.2e})."
        )

    def test_residuals(self, test_case):
        tc = test_case
        try:
            roots = run_serial(tc)
        except RecursionError:
            pytest.fail(f"{tc['desc']}: serial solve() hit maximum recursion depth.")
        roots = np.atleast_2d(roots)
        assert residuals_pass_or_fail([tc["f"], tc["g"]], roots, tol=tc["tol"]), (
            f"{tc['desc']} (serial): residual test failed. "
            f"Max f residual={np.max(residuals(tc['f'], roots)):.2e}, "
            f"Max g residual={np.max(residuals(tc['g'], roots)):.2e}."
        )


class TestParallelRoots:
    """Parallel solve() must match polished ground-truth roots."""

    def test_root_count(self, test_case):
        tc = test_case
        try:
            roots = run_parallel(tc)
        except RecursionError:
            pytest.fail(f"{tc['desc']}: parallel solve() hit maximum recursion depth.")
        roots = np.atleast_2d(roots)
        assert len(roots) == len(tc["polished"]), (
            f"{tc['desc']}: expected {len(tc['polished'])} roots, "
            f"got {len(roots)} (parallel)."
        )

    def test_norm(self, test_case):
        tc = test_case
        try:
            roots = run_parallel(tc)
        except RecursionError:
            pytest.fail(f"{tc['desc']}: parallel solve() hit maximum recursion depth.")
        roots = np.atleast_2d(roots)
        if tc.get("dup_roots", 0):
            roots = collapse_duplicate_roots(roots, tc["tol"])
        passed, x_norm, y_norm = norm_pass_or_fail(roots, tc["polished"], tol=tc["tol"])
        assert passed, (
            f"{tc['desc']} (parallel): norm test failed. "
            f"x_norm={x_norm:.2e}, y_norm={y_norm:.2e} (tol={tc['tol']:.2e})."
        )

    def test_residuals(self, test_case):
        tc = test_case
        try:
            roots = run_parallel(tc)
        except RecursionError:
            pytest.fail(f"{tc['desc']}: parallel solve() hit maximum recursion depth.")
        roots = np.atleast_2d(roots)
        assert residuals_pass_or_fail([tc["f"], tc["g"]], roots, tol=tc["tol"]), (
            f"{tc['desc']} (parallel): residual test failed. "
            f"Max f residual={np.max(residuals(tc['f'], roots)):.2e}, "
            f"Max g residual={np.max(residuals(tc['g'], roots)):.2e}."
        )


class TestSerialVsParallel:
    """Serial and parallel runs must agree with each other."""

    def test_root_count_agreement(self, test_case):
        tc = test_case
        try:
            serial = np.atleast_2d(run_serial(tc))
        except RecursionError:
            pytest.fail(f"{tc['desc']}: serial solve() hit maximum recursion depth.")
        try:
            parallel = np.atleast_2d(run_parallel(tc))
        except RecursionError:
            pytest.fail(f"{tc['desc']}: parallel solve() hit maximum recursion depth.")
        assert len(serial) == len(parallel), (
            f"{tc['desc']}: serial found {len(serial)} roots, "
            f"parallel found {len(parallel)}."
        )

    def test_norm_agreement(self, test_case):
        tc = test_case
        try:
            serial = np.atleast_2d(run_serial(tc))
        except RecursionError:
            pytest.fail(f"{tc['desc']}: serial solve() hit maximum recursion depth.")
        try:
            parallel = np.atleast_2d(run_parallel(tc))
        except RecursionError:
            pytest.fail(f"{tc['desc']}: parallel solve() hit maximum recursion depth.")
        passed, x_norm, y_norm = norm_pass_or_fail(serial, parallel, tol=tc["tol"])
        assert passed, (
            f"{tc['desc']}: serial and parallel roots diverge. "
            f"x_norm={x_norm:.2e}, y_norm={y_norm:.2e} (tol={tc['tol']:.2e})."
        )

    def test_speedup(self, test_case, capsys):
        tc = test_case

        # Warm Numba JIT so the first timed call isn't dominated by compile time.
        run_serial(tc)
        run_parallel(tc)

        t0 = time.perf_counter()
        serial = np.atleast_2d(run_serial(tc))
        t_serial = time.perf_counter() - t0

        t0 = time.perf_counter()
        parallel = np.atleast_2d(run_parallel(tc))
        t_parallel = time.perf_counter() - t0

        speedup = t_serial / t_parallel if t_parallel > 0 else float("inf")

        with capsys.disabled():
            print(
                f"\n{tc['desc']}: "
                f"serial={t_serial:.3f}s  parallel={t_parallel:.3f}s  "
                f"speedup={speedup:.2f}x  "
                f"(roots: {len(serial)} vs {len(parallel)})"
            )


# ---------------------------------------------------------------------------
# Concurrent determinism stress test
# ---------------------------------------------------------------------------

class TestParallelDeterminism:
    """
    Repeatedly running the same solve under the parallel driver must produce
    the same root set every time. Different worker-completion orders change
    the sequence in which subdivided regions are queued, so if any of that
    ordering leaked into the results this test would flake.
    """

    @pytest.mark.parametrize("case_id", sorted(HEAVY_CASE_IDS))
    def test_repeated_parallel_solves_agree(self, case_id, capsys):
        tc = next(t for t in TEST_CASES if t["id"] == case_id)

        baseline = None
        for i in range(DETERMINISM_ITERATIONS):
            try:
                roots = np.atleast_2d(run_parallel(tc))
            except RecursionError:
                pytest.fail(
                    f"{tc['desc']}: parallel solve() hit maximum recursion "
                    f"depth on iteration {i}."
                )

            if baseline is None:
                baseline = roots
                continue

            assert len(roots) == len(baseline), (
                f"{tc['desc']}: iteration {i} found {len(roots)} roots, "
                f"baseline had {len(baseline)}."
            )

            passed, x_norm, y_norm = norm_pass_or_fail(roots, baseline, tol=tc["tol"])
            assert passed, (
                f"{tc['desc']}: iteration {i} diverged from baseline. "
                f"x_norm={x_norm:.2e}, y_norm={y_norm:.2e} (tol={tc['tol']:.2e})."
            )

        with capsys.disabled():
            print(
                f"\n{tc['desc']}: {DETERMINISM_ITERATIONS} parallel solves "
                f"agreed on {len(baseline)} roots."
            )


# ---------------------------------------------------------------------------
# max_cpu sweep — different worker counts must produce identical root sets
# ---------------------------------------------------------------------------

class TestMaxCpuSweep:
    """
    A specific worker count is a scheduling choice, not a mathematical one.
    Solving the same system with max_cpu = 1, 2, 4, 8 must produce the same
    root set. If it doesn't, worker count is influencing the answer (leaked
    ordering, unstable subdivision decisions, race in a supposedly-pure code
    path) — all classes of bug worth catching.
    """

    def _solve_with(self, tc, max_cpu):
        kwargs = dict(exact=True)
        if max_cpu > 1:
            kwargs["max_cpu"] = max_cpu
            kwargs["parallel_depth"] = PARALLEL_DEPTH
        return np.atleast_2d(
            solve([tc["f"], tc["g"]], tc["a_min"], tc["a_max"], **kwargs)
        )

    def test_max_cpu_sweep_agreement(self, test_case, capsys):
        tc = test_case

        # max_cpu=1 is the reference — no scheduler in the loop at all.
        try:
            baseline = self._solve_with(tc, max_cpu=1)
        except RecursionError:
            pytest.fail(f"{tc['desc']}: serial solve() hit maximum recursion depth.")

        for max_cpu in MAX_CPU_SWEEP:
            if max_cpu == 1:
                continue
            try:
                roots = self._solve_with(tc, max_cpu=max_cpu)
            except RecursionError:
                pytest.fail(
                    f"{tc['desc']}: parallel solve() with max_cpu={max_cpu} "
                    f"hit maximum recursion depth."
                )

            assert len(roots) == len(baseline), (
                f"{tc['desc']}: max_cpu={max_cpu} found {len(roots)} roots, "
                f"baseline (max_cpu=1) had {len(baseline)}."
            )

            passed, x_norm, y_norm = norm_pass_or_fail(roots, baseline, tol=tc["tol"])
            assert passed, (
                f"{tc['desc']}: max_cpu={max_cpu} diverged from max_cpu=1 baseline. "
                f"x_norm={x_norm:.2e}, y_norm={y_norm:.2e} (tol={tc['tol']:.2e})."
            )

        with capsys.disabled():
            print(
                f"\n{tc['desc']}: max_cpu sweep {MAX_CPU_SWEEP} agreed on "
                f"{len(baseline)} roots."
            )