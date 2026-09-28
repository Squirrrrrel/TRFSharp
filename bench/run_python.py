"""Run the shared benchmark problem set through scipy.optimize.least_squares.

Reads ``problems.json`` (produced by ``gen_problems.py``) and writes
``results_python.json`` with the solution, the counters, the full iteration
trajectory and wall-clock timings.
"""

from __future__ import annotations

import base64
import json
import platform
import statistics
import sys
import time

import numpy as np
import scipy
from scipy.optimize import least_squares

MAX_TRAJ = 400


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def unb64(d):
    if d is None:
        return None
    raw = base64.b64decode(d["b64"])
    return np.frombuffer(raw, dtype="<f8").reshape(d["shape"])


SQRT5 = np.sqrt(5.0)
SQRT10 = np.sqrt(10.0)
SQRT90 = np.sqrt(90.0)


# --------------------------------------------------------------------------
# problem definitions -- MUST match bench/csharp/BenchRunner/Problems.cs
# --------------------------------------------------------------------------
def make(name: str, data: dict):
    """Return (fun, jac_or_None) for the named problem."""

    if name == "rosenbrock_n2":
        def fun(p):
            return np.array([10.0 * (p[1] - p[0] ** 2), 1.0 - p[0]])

        def jac(p):
            return np.array([[-20.0 * p[0], 10.0], [-1.0, 0.0]])

        return fun, jac

    if name == "beale_n2":
        def fun(p):
            return np.array([1.5 - p[0] * (1 - p[1]),
                             2.25 - p[0] * (1 - p[1] ** 2),
                             2.625 - p[0] * (1 - p[1] ** 3)])

        def jac(p):
            return np.array([[-(1 - p[1]), p[0]],
                             [-(1 - p[1] ** 2), 2 * p[0] * p[1]],
                             [-(1 - p[1] ** 3), 3 * p[0] * p[1] ** 2]])

        return fun, jac

    if name == "powell_singular_n4":
        def fun(p):
            return np.array([p[0] + 10 * p[1],
                             SQRT5 * (p[2] - p[3]),
                             (p[1] - 2 * p[2]) ** 2,
                             SQRT10 * (p[0] - p[3]) ** 2])

        def jac(p):
            return np.array([[1.0, 10.0, 0.0, 0.0],
                             [0.0, 0.0, SQRT5, -SQRT5],
                             [0.0, 2 * (p[1] - 2 * p[2]),
                              -4 * (p[1] - 2 * p[2]), 0.0],
                             [2 * SQRT10 * (p[0] - p[3]), 0.0, 0.0,
                              -2 * SQRT10 * (p[0] - p[3])]])

        return fun, jac

    if name == "wood_n4":
        def fun(p):
            return np.array([10 * (p[1] - p[0] ** 2),
                             1 - p[0],
                             SQRT90 * (p[3] - p[2] ** 2),
                             1 - p[2],
                             SQRT10 * (p[1] + p[3] - 2),
                             (p[1] - p[3]) / SQRT10])

        def jac(p):
            return np.array([[-20 * p[0], 10.0, 0.0, 0.0],
                             [-1.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, -2 * SQRT90 * p[2], SQRT90],
                             [0.0, 0.0, -1.0, 0.0],
                             [0.0, SQRT10, 0.0, SQRT10],
                             [0.0, 1 / SQRT10, 0.0, -1 / SQRT10]])

        return fun, jac

    if name == "box3d_n3":
        t = 0.1 * np.arange(1, 11)

        def fun(p):
            return (np.exp(-t * p[0]) - np.exp(-t * p[1])
                    - p[2] * (np.exp(-t) - np.exp(-10 * t)))

        def jac(p):
            J = np.empty((10, 3))
            J[:, 0] = -t * np.exp(-t * p[0])
            J[:, 1] = t * np.exp(-t * p[1])
            J[:, 2] = -(np.exp(-t) - np.exp(-10 * t))
            return J

        return fun, jac

    if name.startswith("dbv_"):
        n = int(name.split("_n")[1])
        h = 1.0 / (n + 1)
        t = np.linspace(h, 1 - h, n)

        def fun(p):
            q = np.concatenate(([0.0], p, [0.0]))
            return (2 * q[1:-1] - q[:-2] - q[2:]
                    + h ** 2 * (q[1:-1] + t + 1) ** 3 / 2)

        def jac(p):
            J = np.zeros((n, n))
            idx = np.arange(n)
            J[idx, idx] = 2 + 1.5 * h ** 2 * (p + t + 1) ** 2
            J[idx[1:], idx[1:] - 1] = -1.0
            J[idx[:-1], idx[:-1] + 1] = -1.0
            return J

        return fun, jac

    if name.startswith("broyden_"):
        n = int(name.split("_n")[1])

        def fun(p):
            f = (3 - 2 * p) * p + 1
            f[1:] -= p[:-1]
            f[:-1] -= 2 * p[1:]
            return f

        def jac(p):
            J = np.zeros((n, n))
            idx = np.arange(n)
            J[idx, idx] = 3 - 4 * p
            J[idx[1:], idx[1:] - 1] = -1.0
            J[idx[:-1], idx[:-1] + 1] = -2.0
            return J

        return fun, jac

    if name in ("linear_rank_deficient", "underdetermined_m6_n12",
                "linear_illcond_n40"):
        A = unb64(data["A"])
        b = unb64(data["b"])

        def fun(p):
            return A @ p - b

        def jac(_p):
            return A

        return fun, jac

    if name.startswith("sinbig_"):
        A = unb64(data["A"])
        b = unb64(data["b"])

        def fun(p):
            return A @ np.sin(p) - b

        def jac(p):
            return A * np.cos(p)

        return fun, jac

    if name in ("exp_fit_m80", "exp_fit_outliers_m80"):
        t = unb64(data["t"])
        y = unb64(data["y"])

        def fun(p):
            return p[0] + p[1] * np.exp(p[2] * t) - y

        def jac(p):
            e = np.exp(p[2] * t)
            return np.column_stack((np.ones_like(t), e, p[1] * t * e))

        return fun, jac

    if name == "gauss2_m200":
        t = unb64(data["t"])
        y = unb64(data["y"])

        def fun(p):
            return (p[0] * np.exp(-0.5 * ((t - p[1]) / p[2]) ** 2)
                    + p[3] * np.exp(-0.5 * ((t - p[4]) / p[5]) ** 2) - y)

        return fun, None

    if name == "damped_osc_m150":
        t = unb64(data["t"])
        y = unb64(data["y"])

        def fun(p):
            return p[0] * np.exp(-p[1] * t) * np.sin(p[2] * t + p[3]) + p[4] - y

        return fun, None

    if name.startswith("gauss3") or name.startswith("mscan_gauss3"):
        # p = [A1, mu1, s1, A2, mu2, s2, A3, mu3, s3]   n = 9
        t = unb64(data["t"])
        y = unb64(data["y"])

        def fun(p):
            g1 = p[0] * np.exp(-0.5 * ((t - p[1]) / p[2]) ** 2)
            g2 = p[3] * np.exp(-0.5 * ((t - p[4]) / p[5]) ** 2)
            g3 = p[6] * np.exp(-0.5 * ((t - p[7]) / p[8]) ** 2)
            return (g1 + g2 + g3) - y

        def jac(p):
            J = np.empty((t.size, 9))
            for k in range(3):
                a, mu, sg = p[3 * k], p[3 * k + 1], p[3 * k + 2]
                z = (t - mu) / sg
                e = np.exp(-0.5 * z * z)
                J[:, 3 * k] = e
                J[:, 3 * k + 1] = a * e * z / sg
                J[:, 3 * k + 2] = a * e * z * z / sg
            return J

        return fun, jac

    if name == "decay_peak_n9":
        # p = [A1, tau1, A2, tau2, Ag, mu, sigma, c0, c1]   n = 9
        t = unb64(data["t"])
        y = unb64(data["y"])

        def fun(p):
            g1 = p[0] * np.exp(-t / p[1])
            g2 = p[2] * np.exp(-t / p[3])
            g3 = p[4] * np.exp(-0.5 * ((t - p[5]) / p[6]) ** 2)
            lin = p[7] + p[8] * t
            return (g1 + g2 + g3 + lin) - y

        def jac(p):
            J = np.empty((t.size, 9))
            e1 = np.exp(-t / p[1])
            J[:, 0] = e1
            J[:, 1] = p[0] * e1 * t / (p[1] * p[1])
            e2 = np.exp(-t / p[3])
            J[:, 2] = e2
            J[:, 3] = p[2] * e2 * t / (p[3] * p[3])
            z = (t - p[5]) / p[6]
            eg = np.exp(-0.5 * z * z)
            J[:, 4] = eg
            J[:, 5] = p[4] * eg * z / p[6]
            J[:, 6] = p[4] * eg * z * z / p[6]
            J[:, 7] = 1.0
            J[:, 8] = t
            return J

        return fun, jac

    raise ValueError(f"unknown problem: {name}")


# --------------------------------------------------------------------------
def run_case(prob, conf):
    fun, analytic_jac = make(prob["name"], prob["data"])
    n = prob["n"]

    if conf["jac"] == "analytic":
        jac_arg = analytic_jac
    elif conf["jac"] == "fd2":
        jac_arg = "2-point"
    else:
        jac_arg = "3-point"

    lb = None if conf["lb"] is None else np.array(conf["lb"], float)
    ub = None if conf["ub"] is None else np.array(conf["ub"], float)
    # NOTE: scipy turns `None` inside the bounds tuple into NaN, so the
    # unbounded side must be given explicitly as +/- inf.
    bounds = (lb if lb is not None else -np.inf,
              ub if ub is not None else np.inf)

    if conf["x_scale"] == "jac":
        x_scale = "jac"
    elif conf["x_scale"] is None:
        x_scale = 1.0
    else:
        x_scale = np.array(conf["x_scale"], float)

    traj = []
    ncb = 0

    # NOTE: naming the parameter `intermediate_result` makes SciPy's
    # `_wrap_callback` hand over the full OptimizeResult (x, nit, nfev)
    # instead of just the ndarray x.
    def cb(intermediate_result):
        nonlocal ncb
        ncb += 1
        if len(traj) < MAX_TRAJ:
            traj.append({
                "x": [float(v) for v in
                      np.asarray(intermediate_result.x).ravel()],
                "nit": int(intermediate_result.nit),
                "nfev": int(intermediate_result.nfev),
            })

    kwargs = dict(
        jac=jac_arg,
        bounds=bounds,
        method="trf",
        ftol=conf["ftol"] if conf["ftol"] is not None else 1e-8,
        xtol=conf["xtol"] if conf["xtol"] is not None else 1e-8,
        gtol=conf["gtol"] if conf["gtol"] is not None else 1e-8,
        x_scale=x_scale,
        loss=conf["loss"] or "linear",
        f_scale=1.0 if conf["f_scale"] is None else conf["f_scale"],
        tr_solver=conf["tr_solver"],
        diff_step=conf["diff_step"],
        max_nfev=None,
        callback=cb,
    )

    x0 = np.array(prob["x0"], float)

    # warm-up (also validates the case)
    res = least_squares(fun, x0, **kwargs)
    traj.clear()

    reps = 9 if n * prob["m"] <= 4000 else (5 if n * prob["m"] <= 200_000 else 3)
    times = []
    for _ in range(reps):
        traj.clear()
        ncb = 0
        t0 = time.perf_counter()
        res = least_squares(fun, x0, **kwargs)
        times.append(time.perf_counter() - t0)

    return {
        "problem": prob["name"],
        "config": conf["id"],
        "x": [float(v) for v in np.asarray(res.x).ravel()],
        "cost": float(res.cost),
        "optimality": float(res.optimality),
        "nfev": int(res.nfev),
        "njev": int(res.njev) if res.njev is not None else -1,
        # scipy >= 1.16 no longer reports `nit` on the trf result; the callback
        # is invoked exactly once per completed iteration, so counting it gives
        # the same number (and is not affected by MAX_TRAJ).
        "nit": int(ncb),
        "status": int(res.status),
        "success": bool(res.success),
        "message": str(res.message),
        "active_mask": [int(v) for v in np.asarray(res.active_mask).ravel()],
        "fun_norm": float(np.linalg.norm(res.fun)),
        "grad_inf": float(np.max(np.abs(res.grad))),
        "trajectory": traj,
        "t_min": min(times),
        "t_median": statistics.median(times),
        "t_mean": statistics.fmean(times),
        "reps": reps,
    }


def main():
    doc = json.load(open("problems.json", encoding="utf-8"))
    out = []
    for prob in doc["problems"]:
        for conf in prob["configs"]:
            key = f"{prob['name']}/{conf['id']}"
            t0 = time.perf_counter()
            try:
                rec = run_case(prob, conf)
                rec["error"] = None
            except Exception as exc:  # pragma: no cover
                rec = {"problem": prob["name"], "config": conf["id"],
                       "error": f"{type(exc).__name__}: {exc}"}
            out.append(rec)
            flag = "ERR " if rec.get("error") else "    "
            print(f"{flag}{key:<44} "
                  f"{time.perf_counter() - t0:8.3f}s", flush=True)

    meta = {
        "engine": "scipy.optimize.least_squares(method='trf')",
        "scipy_version": scipy.__version__,
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "blas": np.__config__.show(mode="dicts").get("Build Dependencies", {})
        .get("blas", {}).get("name", "n/a"),
    }
    json.dump({"meta": meta, "results": out},
              open("results_python.json", "w", encoding="utf-8"))
    print(f"\nwrote results_python.json ({len(out)} cases)")


if __name__ == "__main__":
    main()
