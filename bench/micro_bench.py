"""Micro-benchmark of the two hot kernels in a nine-parameter fit.

Measures, on the Python/NumPy side, what ``bench/csharp/Micro9`` measures on
the MathNet side:

  1. one residual evaluation (m points, 3 exp calls per point)
  2. one economy SVD of the augmented Jacobian  ((m + n) x n)
  3. one finite-difference Jacobian  (n+1 residual evaluations, 2-point)

Run with the same m list on both sides to see where the constant factors go.
"""

from __future__ import annotations

import json
import platform
import time

import numpy as np
import scipy

MS = (32, 64, 128, 256, 512, 1024)
N = 9
ITERS = 400

P = np.array([1.0, 1.5, 0.55, 0.65, 4.0, 0.90, 0.40, 6.8, 0.45])


def residual(t, y, p):
    g1 = p[0] * np.exp(-0.5 * ((t - p[1]) / p[2]) ** 2)
    g2 = p[3] * np.exp(-0.5 * ((t - p[4]) / p[5]) ** 2)
    g3 = p[6] * np.exp(-0.5 * ((t - p[7]) / p[8]) ** 2)
    return (g1 + g2 + g3) - y


def fd_jacobian(t, y, p):
    J = np.empty((t.size, N))
    f0 = residual(t, y, p)
    for k in range(N):
        h = 1e-8 * max(1.0, abs(p[k]))
        pp = p.copy()
        pp[k] += h
        J[:, k] = (residual(t, y, pp) - f0) / h
    return J


def best(fn, iters):
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[0], times[len(times) // 2]


rows = []
for m in MS:
    rng = np.random.default_rng(1000 + m)
    t = np.linspace(-1.0, 9.0, m)
    y = residual(t, np.zeros(m), P) + 0.004 * rng.standard_normal(m)

    # 1. one residual evaluation
    r_min, r_med = best(lambda: residual(t, y, P), ITERS)

    # 2. one economy SVD of the augmented Jacobian
    J = fd_jacobian(t, y, P)
    A = np.zeros((m + N, N))
    A[:m] = J
    s_min, s_med = best(lambda: np.linalg.svd(A, full_matrices=False), ITERS)

    # 3. one finite-difference Jacobian
    f_min, f_med = best(lambda: fd_jacobian(t, y, P), max(20, ITERS // 5))

    rows.append({
        "m": m, "n": N,
        "residual_min": r_min, "residual_med": r_med,
        "svd_min": s_min, "svd_med": s_med,
        "fdjac_min": f_min, "fdjac_med": f_med,
    })
    print(f"  m={m:<5} residual {r_min * 1e6:8.2f} us   "
          f"svd({m + N}x{N}) {s_min * 1e6:9.2f} us   "
          f"fd-jac {f_min * 1e6:9.2f} us")

json.dump({
    "rows": rows,
    "meta": {
        "side": "python",
        "iters": ITERS,
        "scipy_version": scipy.__version__,
        "numpy_version": np.__version__,
        "python": platform.python_version(),
        "blas": np.__config__.show(mode="dicts").get("Build Dependencies", {})
                .get("blas", {}).get("name", "?"),
    },
}, open("results_micro_python.json", "w", encoding="utf-8"), indent=1)
print("wrote results_micro_python.json")
