"""Where does the ~400 us fixed cost of a small FINUFFT type-3 call go? (n=8, 1 thread)."""
import time
import numpy as np
import finufft

n = 8
modes = np.arange(-n, n + 1.0)
phi = np.arccos(0.5 * np.cos(np.pi * np.arange(n + 1) / n) - 0.5)
c = np.ones(2 * n + 1, complex)

def t(f, r=300):
    f()
    ts = []
    for _ in range(r):
        s = time.perf_counter(); f(); ts.append(time.perf_counter() - s)
    return np.median(ts) * 1e6

for eps in [1e-15, 1e-12, 1e-9, 1e-6]:
    print(f'type3 nufft1d3 n={n} eps={eps:g}: {t(lambda: finufft.nufft1d3(modes, c, phi, eps=eps, isign=1, nthreads=1)):.1f} us')
x = np.linspace(-3, 3, n + 1)
print(f'type1 small eps=1e-15: {t(lambda: finufft.nufft1d1(x, c[:n + 1], (2 * n + 1,), eps=1e-15, nthreads=1)):.1f} us')
print(f'type2 small eps=1e-15: {t(lambda: finufft.nufft1d2(x, c, eps=1e-15, nthreads=1)):.1f} us')
# Reused plan: only execute (valid only if the nodes phi are unchanged, i.e. same alpha, beta, n)
p = finufft.Plan(3, 1, eps=1e-15, isign=1, nthreads=1)
p.setpts(modes, s=phi)
print(f'type3 execute on existing plan: {t(lambda: p.execute(c)):.1f} us')
print('--- FINUFFT debug timing for one type-3 call ---', flush=True)
finufft.nufft1d3(modes, c, phi, eps=1e-15, isign=1, nthreads=1, debug=1)
