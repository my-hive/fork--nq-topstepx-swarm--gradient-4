"""Round 13: causal cross-half-consistent refinements from round12 mining:
hot+dis push<24 good; hot+agree push>53 good, 24-53 weak, shape<-0.1 (front-loaded
retrace) good, shape -0.1..0.5 weak; cold+dis push 24-53 weak; |onv|>175 tighter
brackets (35/70). Test shrinks/boosts individually, then combined."""
import numpy as np
from numba import njit
import sys
sys.path.insert(0, ".")
import round5
from round5 import day_arrays, bounds, NS, sess, t, o, h, l, c, feat
from eval.sim import run_days, score

side, ia, bkt = day_arrays()
base = np.zeros(NS, dtype=np.int64)
PUSH = np.zeros(NS); SHAPE = np.zeros(NS); AONV = np.zeros(NS)
for s in range(NS):
    if bkt[s] >= 0:
        v = (7, 9, 6, 0)[bkt[s]]
        if v > 0 and ia[s] >= 0.05:
            v = max(1, v - 3)
        base[s] = v
    if s in feat:
        i0, i1 = bounds[s], bounds[s + 1]
        tt, cc = t[i0:i1], c[i0:i1]
        w = np.where((tt >= 8 * 3600 + 30 * 60) & (tt < 8 * 3600 + 35 * 60))[0]
        PUSH[s] = abs(feat[s]["mv"])
        SHAPE[s] = (cc[w[0] + 23] - cc[w[0]]) / feat[s]["mv"] if len(w) >= 24 and feat[s]["mv"] != 0 else 0.0
        AONV[s] = abs(feat[s]["onv"])

@njit(cache=True)
def build(t, c, bounds, side, sz_day, st_day, tg_day, start, exit_):
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
                if p <= -st_day[s] or p >= tg_day[s]:
                    pos = 0
            out[k] = pos
    return out

def ev(tag, sz_day=None, st=None, tg=None):
    sz_day = base if sz_day is None else sz_day
    st = np.full(NS, 40.0) if st is None else st
    tg = np.full(NS, 80.0) if tg is None else tg
    target = build(t, c, bounds, side, sz_day, st, tg, round5.START, round5.EXIT)
    day_pnl, day_min, _ = run_days(sess, t, o, h, l, c, target, NS)
    sc, rates, passed = score(day_pnl, day_min)
    print(f"{tag:52s} score={sc:.4f} h1={rates[0]:.3f} h2={rates[1]:.3f} "
          f"days={(np.mean(passed) if passed else float('nan')):.1f} pnl={day_pnl.sum():.0f}", flush=True)
    return sc

def mod(fn):
    sz = base.copy()
    for s in range(NS):
        if sz[s] > 0:
            sz[s] = max(0, fn(s, sz[s]))
    return sz

if __name__ == "__main__":
    ev("BASE 0.5382")
    ev("hd: push<24 +1", mod(lambda s, z: z + 1 if bkt[s] == 0 and PUSH[s] < 24 else z))
    ev("hd: push 24-50 -2", mod(lambda s, z: z - 2 if bkt[s] == 0 and 24 <= PUSH[s] < 50 else z))
    ev("hd: push>50 -1", mod(lambda s, z: z - 1 if bkt[s] == 0 and PUSH[s] >= 50 else z))
    ev("ha: push>53 +1", mod(lambda s, z: z + 1 if bkt[s] == 1 and PUSH[s] > 53 else z))
    ev("ha: push 24-53 -2", mod(lambda s, z: z - 2 if bkt[s] == 1 and 24 <= PUSH[s] <= 53 else z))
    ev("ha: shape<-0.1 +1", mod(lambda s, z: z + 1 if bkt[s] == 1 and SHAPE[s] < -0.1 else z))
    ev("ha: shape -0.1..0.5 -2", mod(lambda s, z: z - 2 if bkt[s] == 1 and -0.1 <= SHAPE[s] <= 0.5 else z))
    ev("ha: shape>0.5 -1", mod(lambda s, z: z - 1 if bkt[s] == 1 and SHAPE[s] > 0.5 else z))
    ev("cd: push 24-53 -2", mod(lambda s, z: z - 2 if bkt[s] == 2 and 24 <= PUSH[s] <= 53 else z))
    ev("ALL hot push<24 +1 & 24-50 -2", mod(
        lambda s, z: (z + 1 if PUSH[s] < 24 else z - 2) if bkt[s] == 0 else z))
    st = np.where(AONV > 175, 35.0, 40.0)
    tg = np.where(AONV > 175, 70.0, 80.0)
    ev("onv>175: brackets 35/70", st=st, tg=tg)
    st = np.where(AONV > 200, 35.0, 40.0)
    tg = np.where(AONV > 200, 70.0, 80.0)
    ev("onv>200: brackets 35/70", st=st, tg=tg)
    st = np.where(AONV > 150, 35.0, 40.0)
    tg = np.where(AONV > 150, 70.0, 80.0)
    ev("onv>150: brackets 35/70", st=st, tg=tg)
