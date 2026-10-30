"""Round 12: feature mining within buckets (esp. hot+dis: h1 -36 vs h2 +253).
Find CAUSAL splits where both halves agree. Features: |onv|, push size, overnight
range, push shape (first-120s share), prior-day pnl, imb_along."""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, ".")
import round5
from round5 import day_arrays, bounds, NS, sess, t, o, h, l, c, feat
from round10 import pnl_pts, win, day_pnl, names

side, ia, bkt = day_arrays()
n = NS

rows = []
for s, f in feat.items():
    i0, i1 = bounds[s], bounds[s + 1]
    tt, cc, hh, ll = t[i0:i1], c[i0:i1], h[i0:i1], l[i0:i1]
    ow = np.where(tt < 8 * 3600 + 30 * 60)[0]
    onv_rng = hh[ow].max() - ll[ow].min() if len(ow) else 0
    m = (tt >= 8 * 3600 + 30 * 60) & (tt < 8 * 3600 + 35 * 60)
    w = np.where(m)[0]
    shape = 0.0
    if len(w) >= 24 and f["mv"] != 0:
        shape = (cc[w[0] + 23] - cc[w[0]]) / f["mv"]
    prior = day_pnl[s - 1] if s > 0 else 0.0
    rows.append(dict(s=s, bkt=bkt[s], half=(0 if s < n // 2 else 1),
                     pnl=day_pnl[s], win=win[s],
                     abs_onv=abs(f["onv"]), push=abs(f["mv"]), onv_rng=onv_rng,
                     shape=shape, prior=prior, ia=ia[s]))
D = pd.DataFrame(rows)

def tab(df, col):
    q = df[col].quantile([0, .33, .66, 1]).values
    print(f"== {col} ==")
    for lo, hi in zip(q[:-1], q[1:]):
        sub = df[(df[col] >= lo) & (df[col] < hi)]
        line = f"  {lo:8.1f}-{hi:8.1f} "
        for hf, tag in ((0, "h1"), (1, "h2")):
            ss = sub[sub.half == hf]
            if len(ss):
                line += (f"| {tag} n={len(ss):3d} mean={ss.pnl.mean():6.0f} "
                         f"win={np.mean(ss.win > 0):.2f} ")
            else:
                line += f"| {tag} n=  0            "
        print(line)

for bk in (0, 1, 2):
    df = D[D.bkt == bk]
    print(f"\n########## bucket {names[bk]} pooled n={len(df)} "
          f"(h1 {int((df.half==0).sum())}/h2 {int((df.half==1).sum())}) ##########")
    for col in ("abs_onv", "push", "onv_rng", "shape", "prior", "ia"):
        tab(df, col)
