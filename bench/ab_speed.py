"""A/B the wall-clock effect of the QR factorisation switch.

Usage:  python ab_speed.py <before.json> <after.json> [name-filter]

Reads two BenchRunner result files (identical case sets), pairs them by
problem/config, and reports per-case and aggregate speedups.
"""

import json
import math
import sys

before = json.load(open(sys.argv[1], encoding="utf-8"))["results"]
after = json.load(open(sys.argv[2], encoding="utf-8"))["results"]
filt = sys.argv[3] if len(sys.argv) > 3 else ""

key = lambda r: f"{r['problem']}/{r['config']}"
b = {key(r): r for r in before}
a = {key(r): r for r in after}


def t(r):
    return r.get("t_min") or r.get("t_median") or r.get("t_mean")


print(f"{'case':34}{'before':>12}{'after':>12}{'speedup':>10}")
rows = []
for k in sorted(set(b) & set(a)):
    if filt and filt not in k:
        continue
    tb, ta = t(b[k]), t(a[k])
    if tb is None or ta is None:
        print("no timing field; keys:", [x for x in b[k] if "tim" in x.lower()
                                         or x in ("t", "seconds")])
        break
    rows.append((k, tb, ta))
    if tb > 0.05 or filt:
        print(f"{k:34}{tb * 1e3:10.1f}ms {ta * 1e3:8.1f}ms {tb / ta:9.2f}x")

sp = [tb / ta for _, tb, ta in rows]
gm = math.exp(sum(math.log(x) for x in sp) / len(sp))
tot_b = sum(tb for _, tb, _ in rows)
tot_a = sum(ta for _, _, ta in rows)
print()
print(f"cases            : {len(rows)}")
print(f"geomean speedup  : {gm:.2f}x")
print(f"total wall clock : {tot_b:.2f}s -> {tot_a:.2f}s  ({tot_b / tot_a:.2f}x)")
print(f"faster cases     : {sum(1 for x in sp if x > 1.05)}/{len(sp)}")
print(f"slower >5%       : {sum(1 for x in sp if x < 0.95)}/{len(sp)}")
