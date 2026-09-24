import csv, sys, numpy as np
from collections import defaultdict
def load(p):
    rows = list(csv.DictReader(open(p)))
    for r in rows:
        for k, v in r.items():
            if v in ("True", "False"): r[k] = (v == "True")
            else:
                try: r[k] = float(v)
                except ValueError: pass
    return rows
def pivot(rows, idx, col, val, agg, filt=lambda r: True, fmt="{:9.2e}"):
    g = defaultdict(list)
    for r in rows:
        if filt(r): g[(r[idx], r[col])].append(r[val])
    I = sorted({k[0] for k in g}, key=lambda x: (isinstance(x, str), x)); C = []
    for k in g:
        if k[1] not in C: C.append(k[1])
    out = [f"| {idx} | " + " | ".join(str(c) for c in C) + " |", "|" + "---|" * (len(C) + 1)]
    for i in I:
        vals = []
        for c in C:
            v = [x for x in g.get((i, c), []) if x == x]
            vals.append(fmt.format(agg(v)) if v else "-")
        out.append(f"| {i if not isinstance(i,float) or not i.is_integer() else int(i)} | " + " | ".join(vals) + " |")
    return "\n".join(out)
