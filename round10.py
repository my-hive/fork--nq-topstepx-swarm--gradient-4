"""Round 10: regime diagnostics. Why does h1 love continuation/flip/tight-hot-brackets
while h2 hates them? Per-half per-bucket stats, per-half imb effect, push-size effect.
Find causal features that separate profitable from unprofitable behavior in EACH half."""
import numpy as np
import sys
sys.path.insert(0, ".")
import round5
from round5 import day_arrays, bounds, NS, sess, t, o, h, l, c, feat
from eval.sim import run_days, score

side, ia, bkt = day_arrays()
base = np.zeros(NS, dtype=np.int64)
for s in range(NS):
    if bkt[s] >= 0:
        v = (7, 9, 6, 0)[bkt[s]]
        if v > 0 and ia[s] >= 0.05:
            v = max(1, v - 3)
        base[s] = v

# trade outcomes per session at base config (win/loss/pts at exit)
from numba import njit
@njit(cache=True)
def outcomes(t, c, bounds, side, sz_day, start, exit_, stop, tgt):
    # exit_pnl_pts[s]: pnl in points at final exit (0 if no trade); win[s]: 1/-1/0
    n = len(sz_day)
    pnl_pts = np.zeros(n); win = np.zeros(n, dtype=np.int64)
    for s in range(n):
        if sz_day[s] <= 0:
            continue
        pos, entry, done = 0, 0.0, False
        for k in range(bounds[s], bounds[s + 1]):
            if t[k] < start or t[k] >= exit_:
                if pos != 0:
                    pnl_pts[s] = (c[k - 1] - entry) * (1.0 if pos > 0 else -1.0)
                    win[s] = 1 if pnl_pts[s] > 0 else -1
                pos = 0
            elif pos == 0 and not done:
                pos, entry, done = side[s] * sz_day[s], c[k], True
            elif pos != 0:
                p = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if p <= -stop:
                    pnl_pts[s], win[s] = -stop, -1
                    pos = 0
                    break
                elif p >= tgt:
                    pnl_pts[s], win[s] = tgt, 1
                    pos = 0
                    break
    return pnl_pts, win

pnl_pts, win = outcomes(t, c, bounds, side, base, round5.START, round5.EXIT, 40.0, 80.0)
day_pnl = pnl_pts * base * 2.0  # approx $ (ignores costs)
n = NS
h1, h2 = slice(0, n // 2), slice(n // 2, n)
names = {0: "hot+dis", 1: "hot+agree", 2: "cold+dis", 3: "cold+agree"}

print("== per-half per-bucket day pnl (approx, base config) ==")
for b in range(4):
    for tag, sl in (("h1", h1), ("h2", h2)):
        days = [s for s in range(n)[sl] if bkt[s] == b]
        if not days:
            continue
        p = day_pnl[days]
        print(f"{names[b]:10s} {tag} n={len(days):3d} mean={p.mean():7.0f} win={np.mean(win[days]>0):.2f} sum={p.sum():8.0f}")

print("== per-half imb effect (approx pnl of imb>=.05 vs <.05 days) ==")
for tag, sl in (("h1", h1), ("h2", h2)):
    for cond, days in (("imb_hi", [s for s in range(n)[sl] if bkt[s]>=0 and ia[s]>=.05]),
                       ("imb_lo", [s for s in range(n)[sl] if bkt[s]>=0 and ia[s]<.05])):
        if days:
            p = day_pnl[days]
            print(f"{tag} {cond} n={len(days):3d} mean={p.mean():7.0f} win={np.mean(win[days]>0):.2f}")

print("== push size (abs 8:30-8:35 mv) buckets per half ==")
for tag, sl in (("h1", h1), ("h2", h2)):
    for lo, hi in ((0, 20), (20, 35), (35, 60), (60, 500)):
        days = [s for s in range(n)[sl] if bkt[s] >= 0 and lo <= abs(feat[s]["mv"]) < hi]
        if days:
            p = day_pnl[days]
            print(f"{tag} push {lo:3d}-{hi:3d} n={len(days):3d} mean={p.mean():7.0f} win={np.mean(win[days]>0):.2f} sum={p.sum():8.0f}")

print("== overnight size finer buckets per half ==")
for tag, sl in (("h1", h1), ("h2", h2)):
    for lo, hi in ((0, 50), (50, 100), (100, 175), (175, 300), (300, 3000)):
        days = [s for s in range(n)[sl] if bkt[s] >= 0 and lo <= abs(feat[s]["onv"]) < hi]
        if days:
            p = day_pnl[days]
            print(f"{tag} onv {lo:4d}-{hi:4d} n={len(days):3d} mean={p.mean():7.0f} win={np.mean(win[days]>0):.2f} sum={p.sum():8.0f}")

print("== MFE/MAE: how far do winners/losers go before resolution (pts) ==")
# for stopped trades: max adverse before stop; for wins: max favorable
