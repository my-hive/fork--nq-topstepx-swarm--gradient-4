"""Round 11: rolling self-referential bucket gating (causal regime adaptation).
Per bucket: track past-K sessions' approx pnl; if mean < thresh, shrink (mult) or skip.
Rationale: h1 hot+dis loses (mean -36), h2 hot+dis wins (+253) — regime shift.
Rolling recent performance is a causal detector that adapts to whichever regime is on."""
import numpy as np
from numba import njit
import sys
sys.path.insert(0, ".")
import round5
from round5 import day_arrays, bounds, NS, sess, t, o, h, l, c, START, EXIT
from eval.sim import run_days, score

side, ia, bkt = day_arrays()
base = np.zeros(NS, dtype=np.int64)
for s in range(NS):
    if bkt[s] >= 0:
        v = (7, 9, 6, 0)[bkt[s]]
        if v > 0 and ia[s] >= 0.05:
            v = max(1, v - 3)
        base[s] = v

@njit(cache=True)
def gate_sim(t, c, bounds, side, base, bkt, K, thresh, mult, minobs):
    """Sequential: size s-day = base * mult if bucket rolling mean < thresh, skip if mult==0.
    Rolling stats per bucket over past K TRADED sessions of that bucket."""
    n = len(base)
    sz_day = np.zeros(n, dtype=np.int64)
    # per-bucket ring buffers
    hist = np.zeros((4, K))          # approx pnl pts
    cnt = np.zeros(4, dtype=np.int64)
    for s in range(n):
        b = base[s]
        if b <= 0:
            continue
        bk = bkt[s]
        m = 1.0
        if cnt[bk] >= minobs:
            c0 = cnt[bk]
            acc = 0.0
            for j in range(K):
                acc += hist[bk, (c0 - 1 - j) % K]
            nb = min(c0, K)
            if acc / nb < thresh:
                m = mult
        sz = int(b * m) if mult > 0 else 0
        if sz < 1 and mult > 0:
            sz = 1
        if sz < 1:
            continue
        sz_day[s] = sz
        # replay
        pos, entry, done, res = 0, 0.0, False, 0.0
        for k in range(bounds[s], bounds[s + 1]):
            if t[k] < START or t[k] >= EXIT:
                if pos != 0:
                    res = (c[k - 1] - entry) * (1.0 if pos > 0 else -1.0)
                pos = 0
            elif pos == 0 and not done:
                pos, entry, done = side[s] * sz, c[k], True
            elif pos != 0:
                p = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if p <= -40.0:
                    res = -40.0
                    pos = 0
                    break
                elif p >= 80.0:
                    res = 80.0
                    pos = 0
                    break
        if pos != 0 and res == 0.0:
            k = bounds[s + 1] - 1
            res = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
        # dollar-ish pnl per contract for stats
        hist[bk, cnt[bk] % K] = res
        cnt[bk] += 1
    return sz_day

@njit(cache=True)
def build(t, c, bounds, side, sz_day, start, exit_, stop, tgt):
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
                p = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if p <= -stop or p >= tgt:
                    pos = 0
            out[k] = pos
    return out

def ev(sz_day):
    target = build(t, c, bounds, side, sz_day, START, EXIT, 40.0, 80.0)
    day_pnl, day_min, _ = run_days(sess, t, o, h, l, c, target, NS)
    sc, rates, passed = score(day_pnl, day_min)
    print(f"score={sc:.4f} h1={rates[0]:.3f} h2={rates[1]:.3f} "
          f"days={(np.mean(passed) if passed else float('nan')):.1f} pnl={day_pnl.sum():.0f} "
          f"worst={day_pnl.min():.0f}", flush=True)
    return sc

if __name__ == "__main__":
    print("BASE", end=" "); ev(base)
    print("== skip bucket if past-K mean pnl < thresh ==")
    for K in (8, 12, 20):
        for th in (-50.0, 0.0):
            print(f"K={K} th={th:.0f} skip", end=" ")
            ev(gate_sim(t, c, bounds, side, base, bkt, K, th, 0.0, 5))
    print("== shrink bucket to mult if past-K mean < thresh ==")
    for K in (8, 12, 20):
        for th, m in ((0.0, 0.5), (-30.0, 0.5), (0.0, 0.7), (-50.0, 0.5)):
            print(f"K={K} th={th:.0f} m={m}", end=" ")
            ev(gate_sim(t, c, bounds, side, base, bkt, K, th, m, 5))
