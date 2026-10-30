"""Push microstructure features vs day pnl: retracement from push extreme, push linearity,
time-of-push (early vs late volume), 8:30 gap."""
import numpy as np
import pandas as pd
import sweep3 as s3
from sweep3 import NS, srt, end, t, o, h, l, c, bv, sv, ONV, AGREE, PUSH, IA, PVOL, DOW
from eval.sim import run_days

# features
RETR = np.zeros(NS)      # pullback from push extreme at 8:35, in pts
PUSH_HI_T = np.zeros(NS) # fraction of push-window volume in first half
GAP = np.zeros(NS)       # 8:30 open vs midnight close
for s in range(NS):
    i0, i1 = srt[s], end[s]
    tt, cc = t[i0:i1 + 1], c[i0:i1 + 1]
    hh, ll = h[i0:i1 + 1], l[i0:i1 + 1]
    w = np.where((tt >= 8 * 3600 + 30 * 60) & (tt < 8 * 3600 + 35 * 60))[0]
    if len(w) == 0 or PUSH[s] == 0:
        continue
    mv = cc[w[-1]] - cc[w[0]]
    if mv > 0:
        ext = hh[w].max()
    else:
        ext = ll[w].min()
    RETR[s] = abs(ext - cc[w[-1]])
    vol = (bv[i0:i1 + 1][w] + sv[i0:i1 + 1][w])
    half = len(w) // 2
    PUSH_HI_T[s] = vol[:half].sum() / max(vol.sum(), 1)
    GAP[s] = cc[w[0]] - cc[0]

target = s3.cfg(imth=0.03, vol_min=12000.0)
day_pnl, day_min, trades = run_days(s3.sess, t, o, s3.h, s3.l, c, target, NS)

traded = PUSH > 0

def stat(mask, label):
    if mask.sum() == 0:
        return
    p = day_pnl[mask]
    print(f"{label:<30} n={mask.sum():>3} mean={p.mean():>7.0f} win={(p > 0).mean():.2f} tot={p.sum():>7.0f}")

print("== RETR (pullback from push extreme at 8:35) ==")
for lo, hi in ((0, 2), (2, 5), (5, 10), (10, 20), (20, 500)):
    stat(traded & (RETR >= lo) & (RETR < hi), f"retr {lo}-{hi}")

print("== PUSH_HI_T (vol front-loading) ==")
for lo, hi in ((0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 1.0)):
    stat(traded & (PUSH_HI_T >= lo) & (PUSH_HI_T < hi), f"hi_t {lo:.1f}-{hi:.1f}")

print("== GAP (midnight->8:30) ==")
for lo, hi in ((-500, -100), (-100, -25), (-25, 25), (25, 100), (100, 500)):
    stat(traded & (GAP >= lo) & (GAP < hi), f"gap {lo}-{hi}")

print("== RETR x bucket ==")
hot = (np.abs(ONV) < 50) | (np.abs(ONV) > 200)
agree = AGREE == 1
for lab, m in (("hot+agree", hot & agree), ("hot+dis", hot & ~agree), ("cold+dis", ~hot & ~agree)):
    mm = traded & m
    med = np.median(RETR[mm])
    stat(mm & (RETR < med), f"{lab} retr<{med:.0f}")
    stat(mm & (RETR >= med), f"{lab} retr>={med:.0f}")
