"""Overnight session (0:00-0:05) family + ATR-normalized dead zone + skip-after-win + on_imb combo."""
import numpy as np
from numba import njit
import sweep
from sweep import bounds, t, o, h, l, c, sess, NS, START
from eval.sim import run_days, score
from round2b import recompute, sizes_for, ev, show
from families import run_generic, sess_feats
from gates import feat

b, sd = recompute(50, 175)
SZ0 = sizes_for(b)
OPEN830 = 8 * 3600 + 30 * 60

# --- overnight 0:00-0:05 family
sz_on = np.zeros(NS, dtype=np.int64); side_on = np.zeros(NS, dtype=np.int64)
for f in sess_feats:
    s = f["s"]
    i0, i1 = bounds[s], bounds[s + 1]
    tt, cc = t[i0:i1], c[i0:i1]
    m = (tt >= 0) & (tt < 300)
    w = np.where(m)[0]
    if not len(w):
        continue
    mv = cc[w[-1]] - cc[w[0]]
    if mv == 0:
        continue
    sz_on[s] = 7
    side_on[s] = -1 if mv > 0 else 1  # fade

if __name__ == "__main__":
    print("== overnight 0:00-0:05 fade (standalone) ==")
    for stop, tgt in ((20, 40), (30, 60), (40, 80)):
        arr = run_generic(t, c, bounds, sz_on, side_on, 300, 8 * 3600, stop, tgt)
        dp, dm, _ = run_days(sess, t, o, h, l, c, arr, NS)
        sc, rates, passed = score(dp, dm)
        print(f"on-fade 0:05->8:00 {stop}/{tgt}: score={sc:.4f} h1={rates[0]:.3f} h2={rates[1]:.3f} pnl={dp.sum():.0f}")
    print("== combo: morning trade + overnight fade 20/40 (sizes add) ==")

    @njit(cache=False)
    def build_combo(t, c, bounds, sd, SZ, sz_on, side_on):
        out = np.zeros(len(c), dtype=np.int64)
        for s in range(len(bounds) - 1):
            # overnight leg
            if sz_on[s] > 0:
                side, pos, entry, done = side_on[s], 0, 0.0, False
                for k in range(bounds[s], bounds[s + 1]):
                    if t[k] < 300 or t[k] >= 8 * 3600:
                        pos = 0
                    elif pos == 0 and not done:
                        pos, entry, done = side * sz_on[s], c[k], True
                    elif pos != 0:
                        pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                        if pnl <= -20 or pnl >= 40:
                            pos = 0
                    out[k] = pos
            # morning leg
            if SZ[s] > 0:
                side, pos, entry, done = sd[s], 0, 0.0, False
                for k in range(bounds[s], bounds[s + 1]):
                    if t[k] < 8 * 3600 + 35 * 60 or t[k] >= 14 * 3600 + 30 * 60:
                        pos = 0
                    elif pos == 0 and not done:
                        pos, entry, done = side * SZ[s], c[k], True
                    elif pos != 0:
                        pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                        if pnl <= -40 or pnl >= 80:
                            pos = 0
                    out[k] = pos
        return out

    for szo in (3, 5, 7):
        arr = build_combo(t, c, bounds, sd, SZ0, np.full(NS, szo, dtype=np.int64), side_on)
        dp, dm, _ = run_days(sess, t, o, h, l, c, arr, NS)
        sc, rates, passed = score(dp, dm)
        print(f"combo on-fade sz{szo} 20/40: score={sc:.4f} h1={rates[0]:.3f} h2={rates[1]:.3f} pnl={dp.sum():.0f} worst={dp.min():.0f}")

    print("== ATR-normalized dead zone ==")
    # 10-day ATR of session ranges (prior 10 sessions)
    rng = np.zeros(NS)
    for s in range(NS):
        i0, i1 = bounds[s], bounds[s + 1]
        rng[s] = np.nanmax(h[i0:i1]) - np.nanmin(l[i0:i1])
    atr10 = np.zeros(NS)
    for s in range(NS):
        lo = max(0, s - 10)
        atr10[s] = rng[lo:s].mean() if s > lo else rng[s]
    from round2b import recompute as rc
    for zlo, zhi in ((0.35, 1.2), (0.4, 1.3), (0.45, 1.4), (0.5, 1.5)):
        buckets = np.full(NS, -1, dtype=np.int64)
        sides = np.zeros(NS, dtype=np.int64)
        for s in range(NS):
            i0, i1 = bounds[s], bounds[s + 1]
            tt, cc = t[i0:i1], c[i0:i1]
            m = (tt >= OPEN830) & (tt < START)
            w = np.where(m)[0]
            if not len(w):
                continue
            onv = cc[w[0]] - cc[0]
            mv = cc[w[-1]] - cc[w[0]]
            if mv == 0 or atr10[s] == 0:
                continue
            zon = abs(onv) / atr10[s]
            hot = not (zlo <= zon <= zhi)
            agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
            buckets[s] = (0 if hot else 2) + (1 if agree else 0)
            sides[s] = -1 if mv >= 0 else 1
        SZ = np.zeros(NS, dtype=np.int64)
        for s, f in feat.items():
            if f is None or buckets[s] < 0:
                continue
            bb = [7, 9, 6, 0][buckets[s]]
            if bb > 0 and f["imb"] * np.sign(f["mv"]) >= 0.05:
                bb = max(1, bb - 2)
            SZ[s] = bb
        arr = run_generic(t, c, bounds, SZ, sides, START, 14 * 3600 + 30 * 60, 40.0, 80.0)
        dp, dm, _ = run_days(sess, t, o, h, l, c, arr, NS)
        sc, rates, passed = score(dp, dm)
        print(f"ATR zone ({zlo},{zhi}): score={sc:.4f} h1={rates[0]:.3f} h2={rates[1]:.3f} pnl={dp.sum():.0f}")
