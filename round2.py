"""Round 2 experiments on top of 0.5256 (imbalance-gated (7,9,6,0)).
1. breakeven / trailing stops (correct impl this time)
2. graduated imbalance shrink
3. dead-zone re-sweep with gate
4. limit-entry / extension-entry variants"""
import numpy as np
from numba import njit
import sweep
from sweep import bounds, t, o, h, l, c, sess, NS, START
from eval.sim import run_days, score
from gates import feat, PUSH, IMB

# per-day size under current best
def base_size(f):
    return [7, 9, 6, 0][f["bucket"]]

SZ = np.zeros(NS, dtype=np.int64)
for s, f in feat.items():
    if f is None:
        continue
    b = base_size(f)
    if b > 0 and f["imb"] * np.sign(f["mv"]) >= 0.05:
        b = max(1, b - 2)
    SZ[s] = b

@njit(cache=True)
def build3(t, c, sess, bounds, sides, sz_day, start, exit_, stop, tgt,
           be_after, trail_after, hi_lo_stops):
    out = np.zeros(len(c), dtype=np.int64)
    for s in range(len(bounds) - 1):
        sz = sz_day[s]
        if sz == 0:
            continue
        side, pos, entry, done = sides[s], 0, 0.0, False
        for k in range(bounds[s], bounds[s + 1]):
            if t[k] < start or t[k] >= exit_:
                pos = 0
            elif pos == 0 and not done:
                pos, entry, done = side * sz, c[k], True
            elif pos != 0:
                if hi_lo_stops:
                    # use this bar's adverse extreme for stop checks, close for target
                    if pos > 0:
                        adverse = (entry - l[k]) / 1.0
                    else:
                        adverse = (h[k] - entry) / 1.0
                    pnl = adverse if adverse < 0 else (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                    pnl_hi = adverse  # worst case within bar
                else:
                    pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                stp_eff = stop
                if be_after > 0 and pnl >= be_after:
                    stp_eff = 0.0
                if trail_after > 0 and pnl >= trail_after:
                    stp_eff = min(stp_eff, pnl - trail_after)
                if pnl <= -stp_eff or pnl >= tgt:
                    pos = 0
            out[k] = pos
    return out

def evaluate3(sz_day=SZ, stop=40.0, tgt=80.0, exit_=14 * 3600 + 30 * 60,
              be_after=0.0, trail_after=0.0, hi_lo_stops=False, start=START):
    target = build3(t, c, sess, bounds, sweep.sides, sz_day, start, exit_,
                    stop, tgt, be_after, trail_after, hi_lo_stops)
    day_pnl, day_min, _ = run_days(sess, t, o, h, l, c, target, NS)
    sc, rates, passed = score(day_pnl, day_min)
    return dict(sc=sc, h1=rates[0], h2=rates[1], days=(np.mean(passed) if passed else np.nan),
                pnl=day_pnl.sum(), worst=day_pnl.min())

def show(tag, r):
    print(f"{tag:48s} score={r['sc']:.4f} h1={r['h1']:.3f} h2={r['h2']:.3f} days={r['days']:.1f} pnl={r['pnl']:.0f} worst={r['worst']:.0f}", flush=True)

if __name__ == "__main__":
    show("BASE 0.5256", evaluate3())
    print("== breakeven stop ==")
    for be in (20, 30, 40, 50, 60):
        show(f"be_after={be}", evaluate3(be_after=be))
    print("== trailing stop ==")
    for tr in (20, 30, 40, 50, 60):
        show(f"trail_after={tr}", evaluate3(trail_after=tr))
    print("== hi/lo stops ==")
    show("hi_lo stops", evaluate3(hi_lo_stops=True))
    print("== graduated imbalance ==")
    for th2 in (0.08, 0.10, 0.12):
        SZ2 = np.zeros(NS, dtype=np.int64)
        for s, f in feat.items():
            if f is None:
                continue
            b = base_size(f)
            ia = f["imb"] * np.sign(f["mv"])
            if b > 0 and ia >= 0.05:
                b = max(1, b - 2)
            if b > 0 and ia >= th2:
                b = max(1, b - 2)
            SZ2[s] = b
        show(f"2nd tier th={th2}", evaluate3(sz_day=SZ2))
