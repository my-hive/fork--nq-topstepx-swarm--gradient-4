"""Round 7: (a) per-bucket stop/target, (b) martingale after loss (wins cluster after losses),
(c) vol-adaptive sizing from prior-session ranges."""
import numpy as np
from numba import njit
import sys
sys.path.insert(0, ".")
import round5
from round5 import day_arrays, bounds, NS, sess, t, o, h, l, c, START, EXIT, feat
from eval.sim import run_days, score
from round6 import build, sim

side, ia, bkt = day_arrays()
base = np.zeros(NS, dtype=np.int64)
for s in range(NS):
    if bkt[s] >= 0:
        v = (7, 9, 6, 0)[bkt[s]]
        if v > 0 and ia[s] >= 0.05:
            v = max(1, v - 3)
        base[s] = v

def ev_sz(sz_day, stop_day=None, tgt_day=None):
    stop_day = np.full(NS, 40.0) if stop_day is None else stop_day
    tgt_day = np.full(NS, 80.0) if tgt_day is None else tgt_day
    @njit(cache=False)
    def _b(t, c, bounds, side, sz_day, st, tg, start, exit_):
        out = np.zeros(len(c), dtype=np.int64)
        for s in range(len(bounds) - 1):
            if sz_day[s] <= 0:
                continue
            pos, entry, done = 0, 0.0, False
            for k in range(bounds[s], bounds[s + 1]):
                if t[k] < start or t[k] >= exit_:
                    pos = 0
                elif pos == 0 and not done:
                    pos, entry, done = side[s] * sz_day[s], c[k], True
                elif pos != 0:
                    pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                    if pnl <= -st[s] or pnl >= tg[s]:
                        pos = 0
                out[k] = pos
        return out
    target = _b(t, c, bounds, side, sz_day, stop_day, tgt_day, START, EXIT)
    day_pnl, day_min, _ = run_days(sess, t, o, h, l, c, target, NS)
    sc, rates, passed = score(day_pnl, day_min)
    return dict(sc=sc, h1=rates[0], h2=rates[1], days=(np.mean(passed) if passed else np.nan),
                pnl=day_pnl.sum(), worst=day_pnl.min())

def show(tag, r):
    print(f"{tag:46s} score={r['sc']:.4f} h1={r['h1']:.3f} h2={r['h2']:.3f} "
          f"days={r['days']:.1f} pnl={r['pnl']:.0f} worst={r['worst']:.0f}", flush=True)

if __name__ == "__main__":
    show("BASE", ev_sz(base))
    print("== per-bucket stop/target (buckets 0,1,2) ==")
    for hot in ((40, 80), (45, 90), (35, 70), (40, 90), (35, 80)):
        for cold in ((40, 80), (30, 60), (35, 70), (25, 50)):
            st = np.zeros(NS); tg = np.zeros(NS)
            for s in range(NS):
                if bkt[s] in (0, 1):
                    st[s], tg[s] = hot
                else:
                    st[s], tg[s] = cold
            show(f"hot {hot} cold {cold}", ev_sz(base, st, tg))
    print("== martingale: after loss size +d (cap 2x), reset on win ==")
    for d in (1, 2):
        for k in (1, 2):
            sd, oc = sim(base, side, bounds, t, c, 99, 1.0, 0, 0.0)  # outcomes at base size
            sz2 = base.copy()
            streak = 0
            for s in range(NS):
                if base[s] <= 0:
                    continue
                if streak >= k:
                    sz2[s] = min(10, base[s] + d * min(streak - k + 1, 2))
                if oc[s] == -1:
                    streak += 1
                elif oc[s] == 1:
                    streak = 0
            show(f"+{d} after k={k} losses", ev_sz(sz2))
    print("== vol-adaptive: scale size by prior-5-session range z ==")
    rng5 = np.zeros(NS)
    ranges = []
    for s in range(NS):
        i0, i1 = bounds[s], bounds[s + 1]
        r = np.nanmax(h[i0:i1]) - np.nanmin(l[i0:i1])
        ranges.append(r)
    ranges = np.array(ranges)
    for s in range(NS):
        if s >= 5:
            rng5[s] = np.median(ranges[max(0, s-5):s])
    med = np.median(rng5[rng5 > 0])
    for scale_hi, scale_lo in ((1.15, 0.85), (1.25, 0.75), (1.1, 0.9)):
        sz2 = base.copy()
        for s in range(NS):
            if base[s] > 0 and rng5[s] > 0:
                sz2[s] = max(1, int(round(base[s] * (scale_hi if rng5[s] > med else scale_lo))))
        show(f"vol {scale_hi}/{scale_lo}", ev_sz(sz2))
