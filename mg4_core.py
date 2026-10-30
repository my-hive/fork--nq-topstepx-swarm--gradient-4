"""Round 4: re-tune core params (sizes, shrink depth, imb th, dead zone) around the
0.6018 disagree-flip stack. Generalized per-session params + numba state machine."""
import sys
import numpy as np
import pandas as pd
from numba import njit
sys.path.insert(0, ".")
from eval.sim import run_days, score

df = pd.read_parquet("/home/shawn/hive-data/nq/train.parquet")
sess = pd.factorize(df.session, sort=True)[0]
t, o, h, l, c = (df[k].values for k in ("ct_sec", "open", "high", "low", "close"))
bv, sv = df.buy_vol.values, df.sell_vol.values
NS = sess.max() + 1

START, EXIT = 8*3600+35*60, 14*3600+30*60

i0 = np.zeros(NS, dtype=np.int64); i1 = np.zeros(NS, dtype=np.int64)
SIDE = np.zeros(NS, dtype=np.int64); HOT = np.zeros(NS, dtype=np.int64)
AGRE = np.zeros(NS, dtype=np.int64); IMB = np.zeros(NS)
for s in range(NS):
    idx = np.where(sess == s)[0]
    i0[s], i1[s] = idx[0], idx[-1]
    tt, cc, bb, ss_ = t[idx], c[idx], bv[idx], sv[idx]
    w = np.where((tt >= 8*3600+30*60) & (tt < START))[0]
    if len(w) == 0:
        continue
    onv = cc[w[0]] - cc[0]
    mv = cc[w[-1]] - cc[w[0]]
    if mv == 0:
        continue
    HOT[s] = 0 if (50.0 <= abs(onv) <= 175.0) else 1
    AGRE[s] = 1 if (np.sign(mv) == np.sign(onv) if onv != 0 else True) else 0
    IMB[s] = (bb[w].sum() - ss_[w].sum()) / max(bb[w].sum() + ss_[w].sum(), 1) * (1 if mv >= 0 else -1)
    SIDE[s] = -1 if mv >= 0 else 1

@njit(cache=True)
def build(t, c, i0, i1, SIDE, HOT, AGRE, IMB,
          sz_ha, sz_hd, sz_cd, shrink, imb_th, flip_dsz,
          stop, tgt, f_stop, f_tgt):
    out = np.zeros(len(c), dtype=np.int64)
    for s in range(len(i0)):
        if HOT[s] == 1:
            sz = sz_ha if AGRE[s] == 1 else sz_hd
        else:
            sz = sz_cd if AGRE[s] == 0 else 0
        if sz == 0:
            continue
        if IMB[s] >= imb_th:
            sz = sz - shrink
            if sz < 1:
                sz = 1
        side = SIDE[s]
        pos, entry, state = 0, 0.0, 0
        for k in range(i0[s], i1[s] + 1):
            if t[k] < START or t[k] >= EXIT:
                pos = 0
            elif state == 0:
                pos, entry, state = side * sz, c[k], 1
            elif pos != 0 and state == 1:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if pnl <= -stop:
                    if AGRE[s] == 1:
                        pos, state = 0, 4
                    else:
                        fsz = sz + flip_dsz
                        if fsz < 1:
                            fsz = 1
                        pos, entry, state = -side * fsz, c[k], 3
                elif pnl >= tgt:
                    pos, state = 0, 4
            elif pos != 0 and state == 3:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if pnl <= -f_stop or pnl >= f_tgt:
                    pos, state = 0, 4
            out[k] = pos
    return out

def evaluate(target):
    day_pnl, day_min, trades = run_days(sess, t, o, h, l, c, target, NS)
    sc, rates, passed = score(day_pnl, day_min)
    return sc, rates, (np.mean(passed) if passed else np.nan)

def show(tag, r):
    print(f"{tag:<40} score={r[0]:.4f} h1={r[1][0]:.3f} h2={r[1][1]:.3f} days={r[2]:.1f}", flush=True)

def run(tag, **kw):
    p = dict(sz_ha=9, sz_hd=7, sz_cd=6, shrink=2, imb_th=0.05, flip_dsz=1,
             stop=40.0, tgt=80.0, f_stop=40.0, f_tgt=60.0)
    p.update(kw)
    show(tag, evaluate(build(t, c, i0, i1, SIDE, HOT, AGRE, IMB, **p)))

if __name__ == "__main__":
    run("BASE (expect 0.6018)")
    for k, v in (("sz_ha", (8, 9, 10)), ("sz_hd", (6, 7, 8)), ("sz_cd", (5, 6, 7)),
                 ("shrink", (1, 2, 3)), ("imb_th", (0.0, 0.03, 0.05, 0.08, 0.12)),
                 ("flip_dsz", (0, 1, 2)), ("stop", (35.0, 40.0, 45.0)),
                 ("tgt", (70.0, 80.0, 90.0)), ("f_stop", (35.0, 40.0, 45.0)),
                 ("f_tgt", (55.0, 60.0, 65.0))):
        for x in v:
            run(f"{k}={x}", **{k: x})
