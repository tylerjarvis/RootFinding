"""Re-run of the nthreads comparison WITHOUT OMP_NUM_THREADS=1 in the environment
(bench_1d.py was launched with OMP_NUM_THREADS=1, which caps FINUFFT's nthreads=0 'auto' to 1 thread).
Run: uv run --no-sync python experiments/fast_transform/e1_timing_1d/threads_rerun.py"""
import os, sys, csv, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', '..')))
import yroots.FastTransform as FT
print('OMP_NUM_THREADS =', os.environ.get('OMP_NUM_THREADS'), ' cpu_count =', os.cpu_count())
def bench(fn, target=0.5):
    fn(); t0 = time.perf_counter(); fn(); first = time.perf_counter() - t0
    nrep = int(np.clip(target / first, 7, 500)); ts = []
    for _ in range(nrep):
        s = time.perf_counter(); fn(); ts.append(time.perf_counter() - s)
    return np.median(ts), np.percentile(ts, 10), np.percentile(ts, 90), nrep
rng = np.random.default_rng(1)
rows = []
for N in [1024, 4096, 16384, 65536]:
    c = rng.uniform(-1, 1, N)
    for eps in [1e-15, 1e-12, 1e-9]:
        for nt in [1, 2, 4, 0]:
            med, p10, p90, nrep = bench(lambda: FT.cheb_affine_fast_axis0(c, 0.5, -0.5, eps=eps, nthreads=nt))
            rows.append(dict(N=N, eps=eps, nthreads=nt, median_s=med, p10_s=p10, p90_s=p90, nrep=nrep))
            print(f'N={N} eps={eps:g} nthreads={nt}: {med*1e3:.3f} ms [{p10*1e3:.3f}, {p90*1e3:.3f}]', flush=True)
with open(os.path.join(HERE, 'threads_rerun.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
