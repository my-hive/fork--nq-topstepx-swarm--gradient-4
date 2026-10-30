"""Flip-variant sweep on top of d0cb886 (imbalance gate) + hot flip.
Per-session python param computation + numba state machine."""
import sys
import itertools
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
ON_LO, ON_HI, IMB_TH = 50.0, 175.0, 0.05
STOP0, TGT0 = 40.0, 80.0

i0 = np.zeros(NS, dtype=np.int64); i1 = np.zeros(NS, dtype=np.int64)
SIDE = np.zeros(NS, dtype=np.int64); SZ = np.zeros(NS, dtype=np.int64)
HOT = np.zeros(NS, dtype=np.int64); AGRE = np.zeros(NS, dtype=np.int64)
SHRUNK = np.zeros(NS, dtype=np.int64)
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
    hot = not (ON_LO <= abs(onv) <= ON_HI)
    agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
    sz = (9 if agree else 7) if hot else (6 if not agree else 0)
    imb = (bb[w].sum() - ss_[w].sum()) / max(bb[w].sum() + ss_[w].sum(), 1)
    shrunk = sz > 0 and imb * np.sign(mv) >= IMB_TH
    if shrunk:
        sz = max(1, sz - 2)
    HOT[s], AGRE[s], SZ[s], SIDE[s], SHRUNK[s] = int(hot), int(agree), sz, (-1 if mv >= 0 else 1), int(shrunk)

@njit(cache=True)
def build(t, c, i0, i1, SIDE, SZ, HOT, AGRE, SHRUNK,
          flip_agree, flip_dis, flip_shrunk_only, flip_unshrunk_only,
          flip_dsz, flip_stop, flip_tgt, stop, tgt):
    out = np.zeros(len(c), dtype=np.int64)
    for s in range(len(i0)):
        sz = SZ[s]
        if sz == 0:
            continue
        side = SIDE[s]
        do_flip = False
        if HOT[s] == 1:
            if flip_shrunk_only and SHRUNK[s] == 1:
                do_flip = True
            elif flip_unshrunk_only and SHRUNK[s] == 0:
                do_flip = True
            elif not flip_shrunk_only and not flip_unshrunk_only:
                if (AGRE[s] == 1 and flip_agree) or (AGRE[s] == 0 and flip_dis):
                    do_flip = True
        fsz = sz + flip_dsz
        if fsz < 1:
            fsz = 1
        pos, entry, state = 0, 0.0, 0
        for k in range(i0[s], i1[s] + 1):
            if t[k] < START or t[k] >= EXIT:
                pos = 0
            elif state == 0:
                pos, entry, state = side * sz, c[k], 1
            elif pos != 0:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if state == 1:
                    if pnl <= -stop:
                        if do_flip:
                            pos, entry, state = -side * fsz, c[k], 3
                        else:
                            pos, state = 0, 4
                    elif pnl >= tgt:
                        pos, state = 0, 4
                else:
                    if pnl <= -flip_stop or pnl >= flip_tgt:
                        pos, state = 0, 4
            out[k] = pos
    return out

def evaluate(target):
    day_pnl, day_min, trades = run_days(sess, t, o, h, l, c, target, NS)
    sc, rates, passed = score(day_pnl, day_min)
    return sc, rates, (np.mean(passed) if passed else np.nan)

def show(tag, r):
    print(f"{tag:<44} score={r[0]:.4f} h1={r[1][0]:.3f} h2={r[1][1]:.3f} days={r[2]:.1f}", flush=True)

if __name__ == "__main__":
    variants = [
        ("noflip (d0cb886 base)", dict(flip_agree=0, flip_dis=0, flip_shrunk_only=0, flip_unshrunk_only=0, flip_dsz=0, flip_stop=STOP0, flip_tgt=TGT0)),
        ("flip all hot (current stack)", dict(flip_agree=1, flip_dis=1, flip_shrunk_only=0, flip_unshrunk_only=0, flip_dsz=0, flip_stop=STOP0, flip_tgt=TGT0)),
        ("flip hot+agree only", dict(flip_agree=1, flip_dis=0, flip_shrunk_only=0, flip_unshrunk_only=0, flip_dsz=0, flip_stop=STOP0, flip_tgt=TGT0)),
        ("flip hot+dis only", dict(flip_agree=0, flip_dis=1, flip_shrunk_only=0, flip_unshrunk_only=0, flip_dsz=0, flip_stop=STOP0, flip_tgt=TGT0)),
        ("flip shrunk only", dict(flip_agree=0, flip_dis=0, flip_shrunk_only=1, flip_unshrunk_only=0, flip_dsz=0, flip_stop=STOP0, flip_tgt=TGT0)),
        ("flip unshrunk only", dict(flip_agree=0, flip_dis=0, flip_shrunk_only=0, flip_unshrunk_only=1, flip_dsz=0, flip_stop=STOP0, flip_tgt=TGT0)),
        ("flip all hot dsz-2", dict(flip_agree=1, flip_dis=1, flip_shrunk_only=0, flip_unshrunk_only=0, flip_dsz=-2, flip_stop=STOP0, flip_tgt=TGT0)),
        ("flip all hot dsz-3", dict(flip_agree=1, flip_dis=1, flip_shrunk_only=0, flip_unshrunk_only=0, flip_dsz=-3, flip_stop=STOP0, flip_tgt=TGT0)),
        ("flip all hot 30stop", dict(flip_agree=1, flip_dis=1, flip_shrunk_only=0, flip_unshrunk_only=0, flip_dsz=0, flip_stop=30.0, flip_tgt=TGT0)),
        ("flip all hot 60tgt", dict(flip_agree=1, flip_dis=1, flip_shrunk_only=0, flip_unshrunk_only=0, flip_dsz=0, flip_stop=STOP0, flip_tgt=60.0)),
        ("flip all hot 120tgt", dict(flip_agree=1, flip_dis=1, flip_shrunk_only=0, flip_unshrunk_only=0, flip_dsz=0, flip_stop=STOP0, flip_tgt=120.0)),
        ("flip all hot 30/120", dict(flip_agree=1, flip_dis=1, flip_shrunk_only=0, flip_unshrunk_only=0, flip_dsz=0, flip_stop=30.0, flip_tgt=120.0)),
        ("flip all hot 25/100", dict(flip_agree=1, flip_dis=1, flip_shrunk_only=0, flip_unshrunk_only=0, flip_dsz=0, flip_stop=25.0, flip_tgt=100.0)),
    ]
    for tag, kw in variants:
        show(tag, evaluate(build(t, c, i0, i1, SIDE, SZ, HOT, AGRE, SHRUNK, stop=STOP0, tgt=TGT0, **kw)))
