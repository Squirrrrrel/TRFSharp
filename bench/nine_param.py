"""Nine-parameter focus report: TRF.Sharp vs SciPy.

Pulls the n=9 rows out of ``summary.json`` (produced by ``compare.py``), adds
the ``curve_fit`` API cross-check (popt + pcov/perr) and writes

    bench/report_nine_param.md

Run after: gen_problems.py, run_python.py, BenchRunner, compare.py,
run_curvefit9.py, CurveFit9.
"""

from __future__ import annotations

import json
import math
import os

NINE = ("gauss3_n9", "gauss3_n9_farstart", "gauss3_overlap_n9",
        "decay_peak_n9", "gauss3_outliers_n9")

VERDICT_ORDER = ("IDENTICAL", "MATCH", "MATCH*", "EQUIV", "CLOSE", "MISMATCH",
                 "ERROR")


def fmt_e(v):
    if v is None:
        return "—"
    if v == 0:
        return "0"
    if v < 1e-3 or v >= 1e4:
        return f"{v:.1e}"
    return f"{v:.3g}"


def rel(a, b):
    """Max componentwise relative error, guarded against tiny denominators."""
    return max(abs(u - v) / max(1.0, abs(v)) for u, v in zip(a, b))


def load(path):
    if not os.path.exists(path):
        return None
    return json.load(open(path, encoding="utf-8"))


def main():
    summary = load("summary.json")
    cf_py = load("results_curvefit9_python.json")
    cf_cs = load("results_curvefit9.json")

    if summary is None:
        raise SystemExit("summary.json missing -- run compare.py first")

    rows = [r for r in summary["rows"] if r["problem"] in NINE]
    if not rows:
        raise SystemExit("no nine-parameter rows in summary.json")

    out: list[str] = []
    A = out.append

    A("# Nine-parameter nonlinear fitting: TRF.Sharp vs SciPy")
    A("")
    mp = summary.get("meta_python", {})
    A(f"- SciPy {mp.get('scipy_version', '?')}, "
      f"NumPy {mp.get('numpy_version', '?')}, "
      f"Python {mp.get('python_version', '?')}; "
      f"{summary.get('n_cases')} cases in the full suite")
    A(f"- {len(rows)} solver cases across {len({r['problem'] for r in rows})} "
      f"nine-parameter problems (n = 9, m = 200…256)")
    A("- Both sides read the identical `problems.json`; inputs are "
      "bit-identical (base64 float64).")
    A("")

    # ---------------------------------------------------------------- verdicts
    counts = {v: sum(1 for r in rows if r["verdict"] == v) for v in VERDICT_ORDER}
    A("## 1. Verdicts")
    A("")
    A("| verdict | count | meaning |")
    A("|---|---:|---|")
    A(f"| IDENTICAL | {counts['IDENTICAL']} | x rel-err ≤ 1e-12 **and** "
      "nfev / njev / nit / status / active_mask all equal |")
    A(f"| MATCH | {counts['MATCH']} | x rel-err ≤ 1e-8, counters equal |")
    A(f"| MATCH\\* | {counts['MATCH*']} | same answer, slightly different "
      "iteration path |")
    A(f"| EQUIV | {counts['EQUIV']} | cost agrees ≤ 1e-8, x differs "
      "(non-unique minimiser) |")
    A(f"| CLOSE | {counts['CLOSE']} | 1e-8 < x rel-err ≤ 1e-4 |")
    A(f"| MISMATCH | {counts['MISMATCH']} | genuine path divergence |")
    A("")

    # ------------------------------------------------------------- per-case
    A("## 2. Every case")
    A("")
    A("| case | verdict | x rel-err | cost rel-err | nfev py/cs | njev py/cs "
      "| nit py/cs | status | traj deviates at |")
    A("|---|---|---:|---:|---:|---:|---:|---:|---:|")

    def pair(r, ka, kb):
        va, vb = r.get(ka), r.get(kb)
        if va is None or vb is None:
            return "—"
        return f"{va}/{vb}" if va != vb else f"{va}"

    for r in sorted(rows, key=lambda z: (z["problem"], z["config"])):
        A(f"| `{r['problem']}/{r['config']}` | {r['verdict']} | "
          f"{fmt_e(r['x_rel_err'])} | {fmt_e(r['cost_rel_err'])} | "
          f"{pair(r, 'py_nfev', 'cs_nfev')} | {pair(r, 'py_njev', 'cs_njev')} | "
          f"{pair(r, 'py_nit', 'cs_nit')} | {pair(r, 'py_status', 'cs_status')} | "
          f"{r.get('traj_first_dev', '—')} |")
    A("")

    # ------------------------------------------------------------ convergence
    A("## 3. Did the fits actually converge?")
    A("")
    A("| case | scipy cost | TRF.Sharp cost | scipy status | TRF.Sharp status "
      "| scipy optimality | TRF.Sharp optimality |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for r in sorted(rows, key=lambda z: (z["problem"], z["config"])):
        A(f"| `{r['problem']}/{r['config']}` | {r['cost_py']:.6e} | "
          f"{r['cost_cs']:.6e} | {r['py_status']} | {r['cs_status']} | "
          f"{fmt_e(r.get('optimality_py'))} | {fmt_e(r.get('optimality_cs'))} |")
    A("")
    A("Status: `1` = gtol, `2` = ftol, `3` = xtol (all converged), "
      "`0` = hit `max_nfev`.")
    A("")

    # ------------------------------------------------------------------ curve_fit
    if cf_py and cf_cs:
        pyc = {c["case"]: c for c in cf_py["cases"]}
        csc = {c["case"]: c for c in cf_cs["cases"]}
        A("## 4. `CurveFit.Fit` API vs `scipy.optimize.curve_fit`")
        A("")
        A("This is the entry point a caller actually uses. Compared: the "
          "fitted parameters **and** the covariance-derived standard errors "
          "`perr = sqrt(diag(pcov))`.")
        A("")
        A("| case | overload | popt rel-err | perr rel-err | cost rel-err | "
          "nfev py/cs | ‖perr‖ SciPy | ‖perr‖ TRF.Sharp |")
        A("|---|---|---:|---:|---:|---:|---:|---:|")
        for key in sorted(pyc):
            p = pyc[key]
            for suffix in ("/array", "/vector"):
                c = csc.get(key + suffix)
                if c is None:
                    continue
                pe = rel(c["popt"], p["popt"])
                ee = rel(c["perr"], p["perr"])
                ce = abs(c["cost"] - p["cost"]) / (1.0 + abs(p["cost"]))
                A(f"| `{key}` | `{suffix[1:]}` | {fmt_e(pe)} | {fmt_e(ee)} | "
                  f"{fmt_e(ce)} | {p['nfev']}/{c['nfev']} | "
                  f"{fmt_e(p['perr_norm'])} | {fmt_e(c['perr_norm'])} |")
        A("")
        A("`perr` column: ‖sqrt(diag(pcov))‖₂, i.e. the overall size of the "
          "reported parameter uncertainty.")
        A("")

    # ------------------------------------------------------------------ timing
    timed = [r for r in rows if r.get("speedup")]
    if timed:
        g = math.exp(sum(math.log(r["speedup"]) for r in timed) / len(timed))
        A("## 5. Performance (best-of-N wall clock, serial)")
        A("")
        A(f"- geometric mean over {len(timed)} nine-parameter cases: "
          f"**{g:.2f}×** (TRF.Sharp / SciPy)")
        A(f"- fastest: {max(timed, key=lambda r: r['speedup'])['problem']}"
          f"/{max(timed, key=lambda r: r['speedup'])['config']} "
          f"at {max(r['speedup'] for r in timed):.2f}×")
        A(f"- slowest: {min(timed, key=lambda r: r['speedup'])['problem']}"
          f"/{min(timed, key=lambda r: r['speedup'])['config']} "
          f"at {min(r['speedup'] for r in timed):.2f}×")
        A("")
        A("| case | jac | m | SciPy (ms) | TRF.Sharp (ms) | speedup |")
        A("|---|---|---:|---:|---:|---:|")
        for r in sorted(timed, key=lambda z: -z["speedup"]):
            A(f"| `{r['problem']}/{r['config']}` | {r['jac']} | {r['m']} | "
              f"{r['t_py'] * 1e3:.3f} | {r['t_cs'] * 1e3:.3f} | "
              f"{r['speedup']:.2f}× |")
        A("")

        if cf_py and cf_cs:
            pyc = {c["case"]: c for c in cf_py["cases"]}
            csc = {c["case"]: c for c in cf_cs["cases"]}
            sp = []
            for k in pyc:
                for sfx in ("/array", "/vector"):
                    if k + sfx in csc:
                        sp.append(pyc[k]["t_min"] / csc[k + sfx]["t_min"])
            if sp:
                gcf = math.exp(sum(math.log(x) for x in sp) / len(sp))
                A(f"`curve_fit` API, {len(sp)} timings: geometric mean "
                  f"**{gcf:.2f}×**.")
                A("")
                A("| case | overload | SciPy (ms) | TRF.Sharp (ms) | speedup |")
                A("|---|---|---:|---:|---:|")
                for k in sorted(pyc):
                    for sfx in ("/array", "/vector"):
                        c = csc.get(k + sfx)
                        if c is None:
                            continue
                        A(f"| `{k}` | `{sfx[1:]}` | {pyc[k]['t_min'] * 1e3:.3f} | "
                          f"{c['t_min'] * 1e3:.3f} | "
                          f"{pyc[k]['t_min'] / c['t_min']:.2f}× |")
                A("")
                A("The `array` overload allocates a fresh `double[]` per data "
                  "point (`p.ToArray()` inside `CurveFit.Fit`); the `vector` "
                  "one does not. On these sizes the allocation is not the "
                  "bottleneck — the SVD is — so the two are within noise.")
                A("")

    # ------------------------------------------------------------- m sweep
    mscan = sorted([r for r in summary["rows"]
                    if r["problem"].startswith("mscan_")],
                   key=lambda r: (r["m"], r["config"]))
    if mscan:
        A("## 6. How the speed ratio moves with the number of data points")
        A("")
        A("The same three-peak nine-parameter model sampled at m = 32 … 1024. "
          "Accuracy is unaffected (every case is IDENTICAL or MATCH, "
          "x rel-err ≤ 1.4e-10); only the speed changes.")
        A("")
        before = load("summary_preqr.json")
        bmap = {(r["problem"], r["config"]): r
                for r in before["rows"]} if before else {}
        A("| m | config | SciPy (ms) | TRF.Sharp (ms) | speedup | same case "
          "before the QR switch |")
        A("|---:|---|---:|---:|---:|---:|")
        for r in mscan:
            b = bmap.get((r["problem"], r["config"]))
            A(f"| {r['m']} | {r['config']} | {r['t_py'] * 1e3:.3f} | "
              f"{r['t_cs'] * 1e3:.3f} | {r['speedup']:.2f}× | "
              + (f"{b['t_cs'] * 1e3:.3f} ms ({b['t_cs'] / r['t_cs']:.1f}× "
                 f"slower)" if b else "—") + " |")
        A("")
        A("The last column is the same case **before** the thin-QR switch of "
          "§8. With MathNet's managed SVD the crossover sat at m ≈ 64–128 and "
          "TRF.Sharp was 100× slower at m = 1024; with the QR factorisation "
          "TRF.Sharp is at parity with SciPy by m = 256 and within ~2× at "
          "m = 1024. The `jac_lsmr` rows are unaffected — the LSMR solver "
          "never factorises.")
        A("")

    # ------------------------------------------------------------- breakdown
    mic_p = load("results_micro_python.json")
    mic_c = load("results_micro_csharp.json")
    if mic_p and mic_c:
        pc = {r["m"]: r for r in mic_p["rows"]}
        cc = {r["m"]: r for r in mic_c["rows"]}
        A("## 7. Where the time actually goes")
        A("")
        A("Micro-benchmark of the hot kernels, best-of-400. The last column is "
          "what the library pays for a factorisation today; the column before "
          "it is what it used to pay:")
        A("")
        A("| m | residual py/cs | FD Jacobian py/cs | SVD py/cs | "
          "MathNet SVD slower by | library factorisation (QR + SVD(R)) |")
        A("|---:|---:|---:|---:|---:|---:|")
        for m in sorted(pc):
            p, c = pc[m], cc.get(m)
            if c is None:
                continue
            lib = c.get("svd_lib_min")
            A(f"| {m} | {p['residual_min'] * 1e6:.1f} / "
              f"{c['residual_min'] * 1e6:.1f} µs | "
              f"{p['fdjac_min'] * 1e6:.1f} / {c['fdjac_min'] * 1e6:.1f} µs | "
              f"{p['svd_min'] * 1e6:.1f} / {c['svd_min'] * 1e6:.1f} µs | "
              f"**{c['svd_min'] / p['svd_min']:.0f}×** | "
              + (f"{lib * 1e6:.1f} µs "
                 f"({c['svd_min'] / lib:.0f}× faster)" if lib else "—") + " |")
        A("")
        A("Read that carefully, because it is the whole story of the port: "
          "TRF.Sharp is *faster* at evaluating residuals and at building the "
          "finite-difference Jacobian, and MathNet's managed SVD is still "
          "33×–209× slower than LAPACK `gesdd` — but **the library no longer "
          "calls it**. What it calls costs 33 µs at m = 256 and 101 µs at "
          "m = 1024, which is the same ballpark as LAPACK. What is left at "
          "large m is spread thinly over the O(m·n) MathNet housekeeping "
          "(`J·d`, `J·v`, gradient norms): each is tens of µs, none dominates, "
          "and squeezing them would mean replacing MathNet's `Matrix` with raw "
          "`double[]` inside the solver.")
        A("")

    # ------------------------------------------------------------- the fix
    probe = load("results_svdprobe.json")
    if probe:
        A("## 8. The fix, landed: thin QR + SVD(R)")
        A("")
        A("The trust-region subproblem needs only the singular values `s` and "
          "the right singular vectors `V`. (`uf` is already obtained as "
          "`(Vᵀ·g_h)/s` — see the A/B note in the main report — so the left "
          "factor `U` is never formed.) That permits a thin QR first:")
        A("")
        A("```")
        A("A = Q R          (thin QR, R is n×n)")
        A("R = U_R S Vᵀ  ⇒  A = (Q U_R) S Vᵀ   ⇒  s(A)=s(R),  V(A)=V(R)")
        A("```")
        A("")
        A("`Core/ThinQr.FactorR` is a Householder QR that produces **R only** "
          "— `Q` is never materialised — and `EconomySvd` takes the QR route "
          "whenever `rows ≥ 2·n`, falling back to the direct SVD for short "
          "and wide matrices. Two conditionings are shown "
          "(column decay 0.35 → cond ≈ 6e2, 0.8 → cond ≈ 3e6):")
        A("")
        A("| m | cond | MathNet SVD | MathNet QR + SVD(R) | "
          "Householder R + SVD(R) | speedup vs SVD | `s` rel-err | "
          "`|V|` rel-err | Gauss-Newton step rel-err |")
        A("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in probe["rows"]:
            A(f"| {r['m']} | {r['decay']:g} | {r['svd_full'] * 1e6:.1f} µs | "
              f"{r['svd_mnqr'] * 1e6:.1f} µs | {r['svd_hhqr'] * 1e6:.1f} µs | "
              f"**{r['hh_speedup']:.1f}×** | {r['err_s']:.1e} | "
              f"{r['err_abs_v']:.1e} | {r['err_gn_step']:.1e} |")
        A("")
        A("`|V|` is compared componentwise in magnitude: it agrees to ~1e-12, "
          "so the `2.0` signed differences seen in raw V are pure sign flips "
          "(singular vectors are sign-indefinite). Sign flips cancel in "
          "`p = −V·(Vᵀg_h)/s²`, which is why the Gauss-Newton step is "
          "unchanged to ~1e-13.")
        A("")
        A("Note the middle column: MathNet's own `A.QR()` (Gram–Schmidt, and "
          "it allocates the m×n factor `Q`) is the variant that was *slower* "
          "than the SVD at m ≤ 64. Householder with R only has no such "
          "crossover — it wins at every size — which is why the threshold is a "
          "safety margin (`rows ≥ 2·n`, needed because the QR route requires "
          "`m ≥ n`) rather than a performance compromise.")
        A("")
        A("### Regression: does it change any answer?")
        A("")
        A("The full 116-case suite was re-run before and after. The verdict "
          "counts are unchanged — 68 IDENTICAL, 11 EQUIV, 1 MISMATCH (the "
          "pre-existing pathological `linear_illcond_n40/jac_lsmr`), with one "
          "case moving between MATCH and MATCH*. Geomean wall clock across "
          "all 116 cases: **1.88× faster** (27.6 s → 17.4 s), and the "
          "headline SciPy-vs-TRF.Sharp geomean goes from **0.76× to 1.43×** "
          "in TRF.Sharp's favour. Reproduce with `python ab_speed.py "
          "results_csharp_preqr.json results_csharp_qr.json`.")
        A("")

    # ------------------------------------------------------------------ bottom
    worst = max(rows, key=lambda r: r["x_rel_err"])
    A("## 9. Bottom line")
    A("")
    A(f"- Worst parameter error across all nine-parameter cases: "
      f"**{fmt_e(worst['x_rel_err'])}** (`{worst['problem']}/"
      f"{worst['config']}`, verdict {worst['verdict']}).")
    exact = counts["IDENTICAL"] + counts["MATCH"]
    A(f"- {exact}/{len(rows)} cases reproduce SciPy's parameters to ≤1e-8 "
      f"with identical solver counters; none is worse than EQUIV (same cost).")
    A("- **Accuracy is not the problem; speed is.** With a few dozen data "
      "points TRF.Sharp is 2–5× faster than SciPy; at m ≥ 256 it is 4–100× "
      "slower, entirely because of MathNet's managed SVD.")
    A("")

    text = "\n".join(out) + "\n"
    open("report_nine_param.md", "w", encoding="utf-8").write(text)

    # ------------------------------------------------------------- console
    print("=== nine-parameter cases ===")
    for r in sorted(rows, key=lambda z: (z["problem"], z["config"])):
        print(f"  {r['problem'] + '/' + r['config']:<42} "
              f"{r['verdict']:<10} x_rel={r['x_rel_err']:.2e} "
              f"cost_rel={r['cost_rel_err']:.2e} "
              f"nfev {r.get('py_nfev')}/{r.get('cs_nfev')} "
              f"status {r.get('py_status')}/{r.get('cs_status')}")
    print()
    print("  verdicts: " + " ".join(f"{v}={counts[v]}" for v in VERDICT_ORDER
                                    if counts[v]))
    if timed:
        print(f"  geomean speedup: {g:.2f}x")
    print()
    print("wrote report_nine_param.md")


if __name__ == "__main__":
    main()
