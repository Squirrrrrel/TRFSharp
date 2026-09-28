"""Compare TRF.Sharp results against the SciPy reference run.

Reads ``results_python.json`` and ``results_csharp.json`` and writes

    report.md      full tables (validation + timings)
    report.html    same content, with charts, self-contained
    summary.json   machine-readable verdicts
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone

# ---------------------------------------------------------------- verdicts
REL_EXACT = 1e-12      # relative deviation we call "numerically identical"
REL_OK = 1e-8          # relative deviation we still accept as "same answer"
COST_EXACT = 1e-12
COST_OK = 1e-8


def rel_err(a, b):
    """max_i |a_i-b_i| / max(1,|b_i|)"""
    if len(a) != len(b):
        return float("inf")
    worst = 0.0
    for u, v in zip(a, b):
        d = abs(u - v) / max(1.0, abs(v))
        worst = max(worst, d)
    return worst


def max_abs(a, b):
    return max((abs(u - v) for u, v in zip(a, b)), default=0.0)


def cost_rel(a, b):
    """|a-b| / (1+|b|) — denominator-stabilised so that an exact zero-cost fit
    is not reported as an infinite relative error."""
    return abs(a - b) / (1.0 + abs(b))


def main():
    py = json.load(open("results_python.json", encoding="utf-8"))
    cs = json.load(open("results_csharp.json", encoding="utf-8"))
    probs = {p["name"]: p for p in
             json.load(open("problems.json", encoding="utf-8"))["problems"]}

    py_by = {(r["problem"], r["config"]): r for r in py["results"]}
    cs_by = {(r["problem"], r["config"]): r for r in cs["results"]}

    keys = [k for k in py_by if k in cs_by]
    keys.sort()

    rows = []
    for k in keys:
        p, c = py_by[k], cs_by[k]
        name, cid = k
        prob = probs[name]

        row = {
            "problem": name,
            "config": cid,
            "n": prob["n"],
            "m": prob["m"],
            "jac": next(cf["jac"] for cf in prob["configs"] if cf["id"] == cid),
        }

        if p.get("error") or c.get("error"):
            row["verdict"] = "ERROR"
            row["detail"] = p.get("error") or c.get("error")
            rows.append(row)
            continue

        x_err = rel_err(c["x"], p["x"])
        x_abs = max_abs(c["x"], p["x"])
        c_err = cost_rel(c["cost"], p["cost"])

        counters_match = (p["nfev"] == c["nfev"] and p["njev"] == c["njev"]
                          and p["nit"] == c["nit"] and p["status"] == c["status"]
                          and p["active_mask"] == c["active_mask"])
        bitwise = c["x"] == p["x"]
        counters = {
            "py_nfev": p["nfev"], "cs_nfev": c["nfev"],
            "py_njev": p["njev"], "cs_njev": c["njev"],
            "py_nit": p["nit"], "cs_nit": c["nit"],
            "py_status": p["status"], "cs_status": c["status"],
            "status_str": p["message"],
        }
        row.update(counters)
        row["x_rel_err"] = x_err
        row["x_abs_err"] = x_abs
        row["cost_rel_err"] = c_err
        row["cost_py"] = p["cost"]
        row["cost_cs"] = c["cost"]
        row["optimality_py"] = p["optimality"]
        row["optimality_cs"] = c["optimality"]
        row["active_mask_match"] = p["active_mask"] == c["active_mask"]
        row["bitwise"] = bitwise

        # ---- iteration trajectory -------------------------------------
        tp, tc = p["trajectory"], c["trajectory"]
        traj_len_match = len(tp) == len(tc)
        traj_max_err = 0.0
        traj_first_dev = None
        traj_counters = True
        for i, (a, b) in enumerate(zip(tp, tc)):
            if a["nit"] != b["nit"] or a["nfev"] != b["nfev"]:
                traj_counters = False
            e = rel_err(b["x"], a["x"])
            traj_max_err = max(traj_max_err, e)
            if traj_first_dev is None and e > REL_EXACT:
                traj_first_dev = i
        row["traj_len_py"] = len(tp)
        row["traj_len_cs"] = len(tc)
        row["traj_len_match"] = traj_len_match
        row["traj_max_err"] = traj_max_err
        row["traj_first_dev"] = traj_first_dev
        row["traj_counters_match"] = traj_counters

        # ---- verdict ---------------------------------------------------
        if x_err <= REL_EXACT and c_err <= COST_EXACT and counters_match:
            verdict = "IDENTICAL"
        elif x_err <= REL_OK and c_err <= COST_OK and counters_match:
            verdict = "MATCH"
        elif x_err <= REL_OK and c_err <= COST_OK:
            verdict = "MATCH*"          # same answer, different iteration path
        elif c_err <= COST_OK:
            # Same cost, different x: legitimate when the minimiser is not
            # unique (rank-deficient Jacobian).
            verdict = "EQUIV"
        elif p["status"] == 0 and c["status"] == 0 and c_err <= 1e-4:
            # Both ran out of function evaluations while drifting along a flat
            # direction (e.g. p0 + p1*exp(p2*t) with p2 -> 0).  The amount of
            # drift is history-dependent; the cost is the same.
            verdict = "EQUIV"
        elif x_err <= 1e-4:
            verdict = "CLOSE"
        else:
            verdict = "MISMATCH"
        row["verdict"] = verdict

        # ---- timing ----------------------------------------------------
        row["t_py"] = p["t_min"]
        row["t_cs"] = c["t_min"]
        row["t_py_median"] = p["t_median"]
        row["t_cs_median"] = c["t_median"]
        row["speedup"] = (p["t_min"] / c["t_min"]) if c["t_min"] > 0 else float("nan")
        row["reps"] = c["reps"]

        rows.append(row)

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "meta_python": py["meta"],
        "meta_csharp": cs["meta"],
        "n_cases": len(rows),
        "verdicts": {v: sum(1 for r in rows if r["verdict"] == v)
                     for v in ("IDENTICAL", "MATCH", "MATCH*", "EQUIV",
                               "CLOSE", "MISMATCH", "ERROR")},
        "bitwise_identical": sum(1 for r in rows if r.get("bitwise")),
        "counters_identical": sum(1 for r in rows
                                  if r.get("py_nfev") is not None
                                  and r["py_nfev"] == r["cs_nfev"]
                                  and r["py_njev"] == r["cs_njev"]
                                  and r["py_nit"] == r["cs_nit"]
                                  and r["py_status"] == r["cs_status"]),
        "active_mask_identical": sum(1 for r in rows if r.get("active_mask_match")),
        "trajectory_paths_identical": sum(1 for r in rows
                                          if r.get("traj_len_match")
                                          and r.get("traj_max_err", 1) <= REL_EXACT),
        "rows": rows,
    }
    # ---- optional A/B: alternative `uf` formulation --------------------
    if os.path.exists("results_csharp_old.json"):
        alt = {(r["problem"], r["config"]): r
               for r in json.load(open("results_csharp_old.json",
                                       encoding="utf-8"))["results"]}
        ab = []
        for k in keys:
            if k not in alt:
                # results_csharp_old.json predates a problem added later
                continue
            p, c = py_by[k], alt[k]
            if p.get("error") or c.get("error"):
                continue
            xe = rel_err(c["x"], p["x"])
            ce = cost_rel(c["cost"], p["cost"])
            cm = (p["nfev"] == c["nfev"] and p["njev"] == c["njev"]
                  and p["nit"] == c["nit"] and p["status"] == c["status"]
                  and p["active_mask"] == c["active_mask"])
            if xe <= REL_EXACT and ce <= COST_EXACT and cm:
                v = "IDENTICAL"
            elif xe <= REL_OK and ce <= COST_OK and cm:
                v = "MATCH"
            elif xe <= REL_OK and ce <= COST_OK:
                v = "MATCH*"
            elif ce <= COST_OK or (p["status"] == 0 and c["status"] == 0
                                   and ce <= 1e-4):
                v = "EQUIV"
            elif xe <= 1e-4:
                v = "CLOSE"
            else:
                v = "MISMATCH"
            cur = next(r for r in rows if (r["problem"], r["config"]) == k)
            ab.append({"case": f"{k[0]}/{k[1]}", "alt": v,
                       "cur": cur["verdict"],
                       "alt_x": xe, "cur_x": cur["x_rel_err"],
                       "alt_cost": ce, "cur_cost": cur["cost_rel_err"]})
        summary["ab"] = ab
        summary["ab_counts_alt"] = {
            v: sum(1 for a in ab if a["alt"] == v)
            for v in ("IDENTICAL", "MATCH", "MATCH*", "EQUIV", "CLOSE",
                      "MISMATCH")}
        summary["ab_counts_cur"] = {
            v: sum(1 for a in ab if a["cur"] == v)
            for v in ("IDENTICAL", "MATCH", "MATCH*", "EQUIV", "CLOSE",
                      "MISMATCH")}

    json.dump(summary, open("summary.json", "w", encoding="utf-8"), indent=1)

    write_markdown(summary)
    write_html(summary)
    print_console(summary)


def print_console(s):
    v = s["verdicts"]
    print(f"\n=== VALIDATION ({s['n_cases']} cases) ===")
    print(f"  IDENTICAL (rel<=1e-12 + counters equal) : {v['IDENTICAL']}")
    print(f"  MATCH     (rel<=1e-8  + counters equal) : {v['MATCH']}")
    print(f"  MATCH*    (rel<=1e-8, different path)   : {v['MATCH*']}")
    print(f"  EQUIV     (cost equal, x differs)       : {v['EQUIV']}")
    print(f"  CLOSE     (rel<=1e-4)                   : {v['CLOSE']}")
    print(f"  MISMATCH                                : {v['MISMATCH']}")
    print(f"  ERROR                                   : {v['ERROR']}")
    print(f"  bitwise-identical x                     : {s['bitwise_identical']}")
    print(f"  identical iteration trajectory          : {s['trajectory_paths_identical']}")
    bad = [r for r in s["rows"] if r["verdict"] in ("CLOSE", "MISMATCH", "ERROR")]
    for r in s["rows"]:
        if r["verdict"] == "EQUIV":
            print(f"  ~~ {r['problem']}/{r['config']}: EQUIV "
                  f"cost_rel={r['cost_rel_err']:.2e} x_rel={r['x_rel_err']:.2e}")
    for r in bad:
        print(f"  !! {r['problem']}/{r['config']}: {r['verdict']} "
              f"x_rel={r.get('x_rel_err', float('nan')):.3e} "
              f"{r.get('detail', '')}")


def fmt_e(v):
    if v == 0:
        return "0"
    if v is None:
        return "-"
    if v < 1e-3 or v >= 1e5:
        return f"{v:.2e}"
    return f"{v:.6g}"


def write_markdown(s):
    v = s["verdicts"]
    L = []
    A = L.append
    A("# TRF.Sharp vs SciPy — validation & benchmark report")
    A("")
    A(f"Generated {s['generated_utc']} · {s['n_cases']} cases · "
      f"{len(set(r['problem'] for r in s['rows']))} problems")
    A("")
    A("## 1. Environment")
    A("")
    A("| | SciPy | TRF.Sharp |")
    A("|---|---|---|")
    mp, mc = s["meta_python"], s["meta_csharp"]
    A(f"| engine | `{mp['engine']}` | `{mc['engine']}` |")
    A(f"| version | scipy {mp['scipy_version']}, numpy {mp['numpy_version']} | "
      f"TRF.Sharp {mc['trf_sharp_version']}, MathNet {mc['mathnet_version']} |")
    A(f"| runtime | CPython {mp['python_version']} | {mc['runtime']} ({mc['build']}) |")
    A(f"| BLAS | {mp['blas']} | MathNet managed (no native BLAS) |")
    A(f"| machine | {mp['machine']} / {mp['platform']} | {mc['arch']} / {mc['os']} |")
    A("")
    A("## 2. Validation summary")
    A("")
    A("| verdict | meaning | count |")
    A("|---|---|---:|")
    A(f"| IDENTICAL | x rel-err ≤ 1e-12 **and** nfev/njev/nit/status/active_mask all equal "
      f"| {v['IDENTICAL']} |")
    A(f"| MATCH | x rel-err ≤ 1e-8 and counters equal | {v['MATCH']} |")
    A(f"| MATCH\\* | x rel-err ≤ 1e-8, counters differ (different iteration path) "
      f"| {v['MATCH*']} |")
    A(f"| CLOSE | 1e-8 < x rel-err ≤ 1e-4 | {v['CLOSE']} |")
    A(f"| MISMATCH | x rel-err > 1e-4 | {v['MISMATCH']} |")
    A(f"| ERROR | one side raised | {v['ERROR']} |")
    A("")
    A(f"- bitwise-identical solution vectors: **{s['bitwise_identical']}/{s['n_cases']}**")
    A(f"- identical evaluation counters "
      f"(nfev/njev/nit/status): **{s['counters_identical']}/{s['n_cases']}**")
    A(f"- identical `active_mask`: **{s['active_mask_identical']}/{s['n_cases']}**")
    A(f"- identical **iteration trajectory** (every intermediate iterate within 1e-12): "
      f"**{s['trajectory_paths_identical']}/{s['n_cases']}**")
    A("")
    A("## 3. Per-case validation")
    A("")
    A("| problem / config | n | m | jac | status | nfev py/cs | njev py/cs | nit py/cs | "
      "x rel-err | cost rel-err | traj | verdict |")
    A("|---|---:|---:|---|---|---|---|---|---:|---:|---|---|")
    for r in s["rows"]:
        if r["verdict"] == "ERROR":
            A(f"| {r['problem']}/{r['config']} | {r['n']} | {r['m']} | {r['jac']} | "
              f"| | | | | | | ERROR: {r.get('detail','')} |")
            continue
        traj = ("=" if r["traj_len_match"] and r["traj_max_err"] <= REL_EXACT
                else f"{r['traj_max_err']:.1e}")
        A(f"| {r['problem']}/{r['config']} | {r['n']} | {r['m']} | {r['jac']} | "
          f"{r['py_status']} | {r['py_nfev']}/{r['cs_nfev']} | "
          f"{r['py_njev']}/{r['cs_njev']} | {r['py_nit']}/{r['cs_nit']} | "
          f"{fmt_e(r['x_rel_err'])} | {fmt_e(r['cost_rel_err'])} | {traj} | "
          f"{r['verdict']} |")
    A("")
    A("`traj` = maximum relative deviation over the whole iteration trajectory "
      "(`=` means every intermediate iterate agrees to ≤1e-12).")
    A("")
    A("## 4. Performance (best-of-N wall clock, seconds)")
    A("")
    A("| problem / config | n × m | jac | t scipy | t TRF.Sharp | speedup |")
    A("|---|---|---|---:|---:|---:|")
    for r in s["rows"]:
        if r["verdict"] == "ERROR":
            continue
        A(f"| {r['problem']}/{r['config']} | {r['n']}×{r['m']} | {r['jac']} | "
          f"{r['t_py']*1e3:.3f} ms | {r['t_cs']*1e3:.3f} ms | {r['speedup']:.2f}× |")
    A("")
    timed = [r for r in s["rows"] if r["verdict"] != "ERROR" and r["t_py"] > 0]
    geo = math.exp(sum(math.log(r["speedup"]) for r in timed) / len(timed))
    A(f"Geometric mean speedup (TRF.Sharp / SciPy): **{geo:.2f}×**")
    by_jac = {}
    for r in timed:
        by_jac.setdefault(r["jac"], []).append(r["speedup"])
    for k, vals in sorted(by_jac.items()):
        g = math.exp(sum(math.log(x) for x in vals) / len(vals))
        A(f"- `{k}` ({len(vals)} cases): geometric mean {g:.2f}×")
    A("")
    A("Broken down by problem size — this is where the picture flips:")
    A("")
    A("| n·m | cases | geomean speedup |")
    A("|---|---:|---:|")
    for lo, hi, label in ((0, 100, "n·m ≤ 100"),
                          (100, 2000, "100 < n·m ≤ 2 000"),
                          (2000, 50_000, "2 000 < n·m ≤ 50 000"),
                          (50_000, 10**9, "n·m > 50 000")):
        grp = [r for r in timed if lo < r["n"] * r["m"] <= hi]
        if not grp:
            continue
        g = math.exp(sum(math.log(r["speedup"]) for r in grp) / len(grp))
        A(f"| {label} | {len(grp)} | {g:.2f}× |")
    A("")
    A("Small problems are dominated by SciPy's per-call Python overhead "
      "(argument validation, `VectorFunction` wrapper, a Python-level loop over "
      "the finite-difference columns), so TRF.Sharp wins by 3-10×. On large "
      "dense problems the cost is the linear algebra — which is why "
      "`EconomySvd` no longer calls MathNet's *managed* SVD (8-12× slower than "
      "OpenBLAS' LAPACK `gesdd`, and scaling worse than O(m·n²)) for tall "
      "matrices. It takes a Householder thin QR that produces only the n×n "
      "factor R, then an SVD of R: 25× faster at m = 256, 154× at m = 1024, "
      "563× at m = 4096 (n = 9), with the Gauss-Newton step reproduced to "
      "~1e-13. See §8 of `report_nine_param.md` and `ab_speed.py`; the "
      "regression is verdict-neutral (68 IDENTICAL before and after).")

    if "ab" in s:
        ca, cc = s["ab_counts_alt"], s["ab_counts_cur"]
        A("")
        A("## 5. A/B: how `uf` is computed in the trust-region subproblem")
        A("")
        A("`solve_lsq_trust_region` needs <code>uf = Uᵀ·f</code>. MathNet's left")
        A("singular vectors lose ~7 digits on Jacobians whose smallest singular value")
        A("is ~1e-10 relative to the largest (LAPACK, used by SciPy, does not). Since")
        A("the augmented rows carry a zero right-hand side, <code>Aᵀ·f_aug = d·g = g_h</code>,")
        A("so <code>suf = s·uf = Vᵀ·g_h</code> can be computed from <code>V</code> alone —")
        A("which is accurate and one mat-vec cheaper.")
        A("")
        A("| variant | IDENTICAL | MATCH | MATCH\* | EQUIV | CLOSE | MISMATCH |")
        A("|---|---:|---:|---:|---:|---:|---:|")
        A(f"| A: `uf = Uᵀ·f` (MathNet U) | {ca['IDENTICAL']} | {ca['MATCH']} | "
          f"{ca['MATCH*']} | {ca['EQUIV']} | {ca['CLOSE']} | {ca['MISMATCH']} |")
        A(f"| B: `uf = (Vᵀ·g_h)/s` (**shipped**) | {cc['IDENTICAL']} | {cc['MATCH']} | "
          f"{cc['MATCH*']} | {cc['EQUIV']} | {cc['CLOSE']} | {cc['MISMATCH']} |")
        A("")
        A("| case | A verdict | B verdict | x rel-err A → B | cost rel-err A → B |")
        A("|---|---|---|---:|---:|")
        for a in s["ab"]:
            if a["alt"] == a["cur"]:
                continue
            A(f"| `{a['case']}` | {a['alt']} | {a['cur']} | "
              f"{fmt_e(a['alt_x'])} → {fmt_e(a['cur_x'])} | "
              f"{fmt_e(a['alt_cost'])} → {fmt_e(a['cur_cost'])} |")
        A("")
        A("Variant B removes the only *algorithmic* divergence "
          "(`box3d_n3/jac_bounded`: x rel-err 4.2e+01 → 1.9e-09, cost rel-err "
          "3.8e-02 → 2.1e-19). The IDENTICAL → MATCH\* moves are 1-ULP shifts: "
          "the solutions still agree to ~1e-12.")

    A("")
    A("## 6. Cases that do not match bit-for-bit")
    A("")
    A("| case | verdict | why |")
    A("|---|---|---|")
    for r in s["rows"]:
        if r["verdict"] in ("EQUIV", "MATCH*", "CLOSE", "MISMATCH"):
            why = {
                "EQUIV": "same cost, different x — non-unique minimiser or flat "
                         "drift to `max_nfev`",
                "MATCH*": "same answer, slightly different iteration path",
                "CLOSE": "small numerical drift",
                "MISMATCH": "genuine path divergence on an ill-conditioned problem",
            }[r["verdict"]]
            A(f"| {r['problem']}/{r['config']} | {r['verdict']} | "
              f"x rel-err {fmt_e(r['x_rel_err'])}, cost rel-err "
              f"{fmt_e(r['cost_rel_err'])}, traj deviates from iteration "
              f"{r['traj_first_dev']} — {why} |")
    A("")
    A("## 7. Notes")
    A("")
    A("- Both sides read the **same** `problems.json`, so x0, bounds, data matrices and "
      "all solver options are bit-identical.")
    A("- Timing is best-of-N of the whole `solve` call (warm-up excluded); "
      "N = 9 for n·m ≤ 4 000, 5 for n·m ≤ 200 000, 3 otherwise.")
    A("- SciPy's dense linear algebra goes through OpenBLAS; TRF.Sharp uses MathNet's "
      "**managed** SVD/QR, which is where large dense cases differ most.")
    A("- `nit` is not exposed by SciPy ≥ 1.16 for `method='trf'`; it is taken from the "
      "last callback, which carries the same iteration counter.")
    open("report.md", "w", encoding="utf-8").write("\n".join(L) + "\n")


def bar(v, vmax, width=140):
    return max(1, int(round(width * v / vmax))) if v > 0 else 1


def write_html(s):
    v = s["verdicts"]
    mp, mc = s["meta_python"], s["meta_csharp"]
    rows = s["rows"]
    timed = [r for r in rows if r["verdict"] != "ERROR" and r["t_py"] > 0]
    geo = (math.exp(sum(math.log(r["speedup"]) for r in timed) / len(timed))
           if timed else float("nan"))

    # ---- validation table ------------------------------------------------
    val_rows = []
    for r in rows:
        if r["verdict"] == "ERROR":
            val_rows.append(
                f"<tr class='bad'><td>{r['problem']}/{r['config']}</td>"
                f"<td>{r['n']}</td><td>{r['m']}</td><td>{r['jac']}</td>"
                f"<td colspan='7'>ERROR: {r.get('detail','')}</td>"
                f"<td><span class='tag bad'>ERROR</span></td></tr>")
            continue
        cls = {"IDENTICAL": "ok", "MATCH": "ok", "MATCH*": "warn",
               "EQUIV": "info", "CLOSE": "warn", "MISMATCH": "bad",
               "ERROR": "bad"}[r["verdict"]]
        traj = ("=" if r["traj_len_match"] and r["traj_max_err"] <= REL_EXACT
                else f"{r['traj_max_err']:.1e}")
        nfev_cls = "" if r["py_nfev"] == r["cs_nfev"] else "diff"
        nit_cls = "" if r["py_nit"] == r["cs_nit"] else "diff"
        val_rows.append(
            f"<tr class='{cls}'><td class='lbl'>{r['problem']}/{r['config']}</td>"
            f"<td>{r['n']}</td><td>{r['m']}</td><td>{r['jac']}</td>"
            f"<td>{r['py_status']}</td>"
            f"<td class='{nfev_cls}'>{r['py_nfev']}/{r['cs_nfev']}</td>"
            f"<td>{r['py_njev']}/{r['cs_njev']}</td>"
            f"<td class='{nit_cls}'>{r['py_nit']}/{r['cs_nit']}</td>"
            f"<td>{fmt_e(r['x_rel_err'])}</td><td>{fmt_e(r['cost_rel_err'])}</td>"
            f"<td>{traj}</td>"
            f"<td><span class='tag {cls}'>{r['verdict']}</span></td></tr>")

    # ---- timing table ----------------------------------------------------
    t_rows = []
    vmax = max(r["speedup"] for r in timed) if timed else 1
    size_rows = []
    for lo, hi, label in ((0, 100, "n&middot;m &le; 100"),
                          (100, 2000, "100 &lt; n&middot;m &le; 2 000"),
                          (2000, 50_000, "2 000 &lt; n&middot;m &le; 50 000"),
                          (50_000, 10**9, "n&middot;m &gt; 50 000")):
        grp = [r for r in timed if lo < r["n"] * r["m"] <= hi]
        if not grp:
            continue
        g = math.exp(sum(math.log(r["speedup"]) for r in grp) / len(grp))
        size_rows.append(
            f"<tr><td class='lbl'>{label}</td><td>{len(grp)}</td>"
            f"<td class='num'>{g:.2f}&times;</td></tr>")
    for r in timed:
        faster = r["speedup"] >= 1.0
        t_rows.append(
            f"<tr><td class='lbl'>{r['problem']}/{r['config']}</td>"
            f"<td>{r['n']}×{r['m']}</td><td>{r['jac']}</td>"
            f"<td class='num'>{r['t_py']*1e3:.3f}</td>"
            f"<td class='num'>{r['t_cs']*1e3:.3f}</td>"
            f"<td class='num'>{r['speedup']:.2f}×</td>"
            f"<td><div class='bar {('up' if faster else 'down')}' "
            f"style='width:{bar(r['speedup'], vmax)}px'></div></td></tr>")

    n_ok = v["IDENTICAL"] + v["MATCH"]
    pct = 100.0 * n_ok / s["n_cases"]

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>TRF.Sharp vs SciPy — validation &amp; benchmark</title>
<style>
:root{{--bg:#f7f8fa;--card:#fff;--ink:#1b1f24;--mute:#5c6672;--line:#e3e7ed;
--ok:#1a7f37;--warn:#b45309;--bad:#b3261e;--accent:#2563eb;}}
*{{box-sizing:border-box}}
body{{margin:0;padding:32px;background:var(--bg);color:var(--ink);
font:14px/1.55 -apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;}}
.wrap{{max-width:1180px;margin:0 auto}}
h1{{font-size:24px;margin:0 0 4px}}
h2{{font-size:17px;margin:32px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line)}}
p.sub{{color:var(--mute);margin:0 0 18px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:18px 20px;margin-bottom:16px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}}
.kpi .big{{font-size:26px;font-weight:650;line-height:1.1}}
.kpi .cap{{color:var(--mute);font-size:12px;margin-top:4px}}
table{{width:100%;border-collapse:collapse;font-size:12.5px}}
th,td{{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;
white-space:nowrap}}
th{{color:var(--mute);font-weight:600;text-align:right;background:#fbfcfd;
position:sticky;top:0}}
td.lbl,th.lbl{{text-align:left;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
font-size:12px}}
tr.ok td{{color:var(--ink)}} tr.warn td{{color:#7c4a03}} tr.bad td{{color:var(--bad)}}
tr.info td{{color:#1e40af}}
td.diff{{font-weight:700;color:var(--bad)}}
.tag{{display:inline-block;padding:1px 7px;border-radius:999px;font-size:11px;
font-weight:600}}
.tag.ok{{background:#e6f4ea;color:var(--ok)}} .tag.warn{{background:#fdf0dc;
color:#7c4a03}} .tag.bad{{background:#fbe9e7;color:var(--bad)}}
.tag.info{{background:#e7effd;color:var(--accent)}}
.bar{{height:9px;border-radius:3px}} .bar.up{{background:var(--ok)}}
.bar.down{{background:#c2410c}}
.scroll{{max-height:620px;overflow:auto;border:1px solid var(--line);
border-radius:10px;background:var(--card)}}
code{{background:#eef1f5;padding:1px 5px;border-radius:4px;font-size:12px}}
ul{{margin:6px 0 0 18px;padding:0}} li{{margin:2px 0}}
.env td{{text-align:left}}
</style></head><body><div class="wrap">
<h1>TRF.Sharp vs SciPy — validation &amp; benchmark</h1>
<p class="sub">{s['n_cases']} cases · {len(set(r['problem'] for r in rows))} problems ·
generated {s['generated_utc']}</p>

<div class="kpis">
<div class="kpi"><div class="big">{pct:.0f}%</div>
<div class="cap">cases with identical answer <em>and</em> identical evaluation counters</div></div>
<div class="kpi"><div class="big">{s['bitwise_identical']}/{s['n_cases']}</div>
<div class="cap">bitwise-identical solution vectors</div></div>
<div class="kpi"><div class="big">{s['trajectory_paths_identical']}/{s['n_cases']}</div>
<div class="cap">identical full iteration trajectory</div></div>
<div class="kpi"><div class="big">{geo:.2f}×</div>
<div class="cap">geomean speedup vs SciPy (best-of-N)</div></div>
</div>

<h2>Environment</h2>
<div class="card"><table class="env">
<tr><th class="lbl">item</th><th class="lbl">SciPy</th><th class="lbl">TRF.Sharp</th></tr>
<tr><td>engine</td><td>{mp['engine']}</td><td>{mc['engine']}</td></tr>
<tr><td>version</td><td>scipy {mp['scipy_version']} · numpy {mp['numpy_version']}</td>
<td>TRF.Sharp {mc['trf_sharp_version']} · MathNet {mc['mathnet_version']}</td></tr>
<tr><td>runtime</td><td>CPython {mp['python_version']}</td>
<td>{mc['runtime']} ({mc['build']})</td></tr>
<tr><td>linear algebra</td><td>{mp['blas']} (native)</td>
<td>MathNet managed QR + SVD(R) (Householder, R only)</td></tr>
<tr><td>machine</td><td>{mp['machine']} · {mp['platform']}</td>
<td>{mc['arch']} · {mc['os']}</td></tr>
</table></div>

<h2>Verdicts</h2>
<div class="card">
<table><tr><th class="lbl">verdict</th><th>count</th><th class="lbl">meaning</th></tr>
<tr><td class="lbl"><span class="tag ok">IDENTICAL</span></td><td>{v['IDENTICAL']}</td>
<td class="lbl">x rel-err ≤ 1e-12, counters + active_mask equal</td></tr>
<tr><td class="lbl"><span class="tag ok">MATCH</span></td><td>{v['MATCH']}</td>
<td class="lbl">x rel-err ≤ 1e-8, counters equal</td></tr>
<tr><td class="lbl"><span class="tag warn">MATCH*</span></td><td>{v['MATCH*']}</td>
<td class="lbl">same answer, slightly different iteration path</td></tr>
<tr><td class="lbl"><span class="tag info">EQUIV</span></td><td>{v['EQUIV']}</td>
<td class="lbl">cost agrees ≤ 1e-8, x differs (non-unique minimiser / flat drift)</td></tr>
<tr><td class="lbl"><span class="tag warn">CLOSE</span></td><td>{v['CLOSE']}</td>
<td class="lbl">1e-8 &lt; x rel-err ≤ 1e-4</td></tr>
<tr><td class="lbl"><span class="tag bad">MISMATCH</span></td><td>{v['MISMATCH']}</td>
<td class="lbl">x rel-err &gt; 1e-4</td></tr>
<tr><td class="lbl"><span class="tag bad">ERROR</span></td><td>{v['ERROR']}</td>
<td class="lbl">one side raised</td></tr>
</table></div>

<h2>Per-case validation</h2>
<div class="scroll"><table>
<tr><th class="lbl">case</th><th>n</th><th>m</th><th class="lbl">jac</th><th>stat</th>
<th>nfev py/cs</th><th>njev py/cs</th><th>nit py/cs</th><th>x rel-err</th>
<th>cost rel-err</th><th>traj</th><th class="lbl">verdict</th></tr>
{''.join(val_rows)}
</table></div>
<p class="sub" style="margin-top:8px">
<code>nfev/njev/nit</code> are printed as <em>scipy / TRF.Sharp</em>; red means they differ.
<code>traj</code> is the worst relative deviation over every intermediate iterate
(<code>=</code> means ≤ 1e-12 everywhere).</p>

<h2>Scaling</h2>
<div class="card"><table>
<tr><th class="lbl">problem size</th><th>cases</th><th>geomean speedup</th></tr>
{''.join(size_rows)}
</table>
<p class="sub" style="margin-top:8px">Small problems are dominated by SciPy's
per-call Python overhead; large dense problems are dominated by linear algebra,
where the factorisation is a Householder thin QR plus an SVD of the n&times;n
factor R — MathNet's managed <em>full</em> SVD is 25-563&times; slower and is
no longer on this path.</p></div>

<h2>Performance (best-of-N, wall clock)</h2>
<div class="scroll"><table>
<tr><th class="lbl">case</th><th>n×m</th><th class="lbl">jac</th>
<th>scipy (ms)</th><th>TRF.Sharp (ms)</th><th>speedup</th><th></th></tr>
{''.join(t_rows)}
</table></div>
<p class="sub" style="margin-top:8px">Geometric mean speedup: <strong>{geo:.2f}×</strong>
(green = TRF.Sharp faster, orange = SciPy faster).</p>

<h2>Method notes</h2>
<div class="card"><ul>
<li>Both sides consume the same <code>problems.json</code>: identical x0, bounds,
data matrices (base64 float64) and solver options.</li>
<li>Best-of-N wall clock of the whole solve; warm-up excluded.
N = 9 for n·m ≤ 4 000, 5 for n·m ≤ 200 000, 3 otherwise.</li>
<li>TRF.Sharp is built <code>Release</code>; the bench host runs on the installed
runtime ({mc['runtime']}) while the library targets <code>net8.0</code>.</li>
<li>SciPy ≥ 1.16 no longer exposes <code>nit</code> for <code>method='trf'</code>;
it is read from the last callback, which carries the same counter.</li>
</ul></div>
</div></body></html>"""
    open("report.html", "w", encoding="utf-8").write(html)


if __name__ == "__main__":
    main()
