"""gu4_diag: failure anatomy of 0.6856 base. Classify failed attempts by cause;
find sessions responsible for MLL breaks; per-bucket pnl by half."""
import numpy as np, pandas as pd
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "eval")
import sim
from harness import DF, SESS
from strategy import signals

START0 = 8 * 3600 + 35 * 60
ON_LO, ON_HI = 50.0, 225.0
IMB_TH = 0.05

target = np.asarray(signals(DF), dtype=np.int64)
day_pnl, day_min, trades = sim.run_days(SESS, DF.ct_sec.values, DF.open.values, DF.high.values,
                                        DF.low.values, DF.close.values, target, SESS.max() + 1)
n = len(day_pnl); mid = n // 2

def classify(lo, hi):
    res = {"mll": [], "target": [], "consist": [], "pass": []}
    brk_sess = np.zeros(n, int)
    for st in range(lo, hi - 30 + 1):
        bal, peak, best, days, out = 50_000.0, 50_000.0, 0.0, 0, None
        cause = None
        for s in range(st, st + 30):
            thr = min(peak - 2000.0, 50_000.0)
            if bal + day_min[s] <= thr:
                cause = ("mll", s); brk_sess[s] += 1
                break
            bal += day_pnl[s]; days += 1
            best, peak = max(best, day_pnl[s]), max(peak, bal)
            profit = bal - 50_000.0
            if profit >= 3000.0 and days >= 2 and best < 0.5 * profit:
                cause = ("pass", days); break
        if cause is None:
            # didn't pass in 30 sessions
            bal2, peak2, best2, days2 = 50_000.0, 50_000.0, 0.0, 0
            hit = False
            for s in range(st, st + 30):
                thr = min(peak2 - 2000.0, 50_000.0)
                if bal2 + day_min[s] <= thr: hit = True; break
                bal2 += day_pnl[s]; days2 += 1
                best2, peak2 = max(best2, day_pnl[s]), max(peak2, bal2)
                if bal2 - 50_000.0 >= 3000.0 and days2 >= 2 and best2 >= 0.5 * (bal2 - 50_000.0):
                    cause = ("consist", days2); break
            if cause is None:
                cause = ("target", 30)
        res[cause[0]].append((st, cause[1]))
    return res, brk_sess

for name, lo, hi in (("h1", 0, mid), ("h2", mid, n)):
    res, brk = classify(lo, hi)
    tot = sum(len(v) for v in res.values())
    print(f"{name}: attempts={tot} pass={len(res['pass'])} mll={len(res['mll'])} "
          f"target={len(res['target'])} consist={len(res['consist'])}")
    print("  top MLL-break sessions (sess_idx: count, day_pnl, day_min):")
    for s in np.argsort(-brk)[:12]:
        if brk[s]:
            print(f"   s{s:3d} brk={brk[s]:2d} pnl={day_pnl[s]:7.0f} min={day_min[s]:7.0f} "
                  f"trades={trades[s]:3.0f} half={'h1' if s < mid else 'h2'}")

# per-session features
t, c = DF.ct_sec.values, DF.close.values
bv, sv = DF.buy_vol.values, DF.sell_vol.values
rows = []
for s_i, (_, idx) in enumerate(DF.groupby("session", sort=False).indices.items()):
    tt, cc = t[idx], c[idx]
    m = (tt >= 8 * 3600 + 30 * 60) & (tt < START0)
    if not m.any(): continue
    w = np.where(m)[0]
    onv = cc[w[0]] - cc[0]; mv = cc[w[-1]] - cc[w[0]]
    imb = (bv[idx][w].sum() - sv[idx][w].sum()) / max(bv[idx][w].sum() + sv[idx][w].sum(), 1)
    rows.append(dict(s=s_i, onv=onv, mv=mv, imb_along=imb * np.sign(mv) if mv else 0,
                     pnl=day_pnl[s_i], dmin=day_min[s_i],
                     bucket=("hot_" if not (ON_LO <= abs(onv) <= ON_HI) else "cold_") +
                            ("agree" if (np.sign(mv) == np.sign(onv) if onv != 0 else True) else "dis"),
                     half="h1" if s_i < mid else "h2"))
d = pd.DataFrame(rows)
print("\nper-bucket pnl by half:")
print(d.groupby(["bucket", "half"]).agg(n=("pnl", "size"), tot=("pnl", "sum"), avg=("pnl", "mean"),
      dll=("dmin", lambda x: int((x <= -990).sum()))).round(0))
print("\nworst 15 days:")
print(d.nsmallest(15, "pnl")[["s", "bucket", "onv", "mv", "imb_along", "pnl", "dmin", "half"]].to_string(index=False))
