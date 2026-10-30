"""Flip-leg conditioning experiments on 0.6018 base (87cbe4a).
Diag insight: 100% of attempt failures are MLL breaches; -1008 DLL days come from
fade-stop + flip-stop double losses. Test: gate/shrink the flip leg."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "eval")
import sim
from harness import score_signals, DF, SESS

START, EXIT = 8*3600+35*60, 14*3600+30*60
STOP_PTS, TGT_PTS = 40.0, 80.0
ON_LO, ON_HI = 50.0, 175.0
IMB_TH = 0.05
F_STOP, F_TGT = 40.0, 60.0

def build(flip_mode="all", flip_sz_delta=1, flip_imb_gate=None, flip_imb_sign=1):
    """flip_mode: all | hot | cold | none. flip_imb_gate: None or threshold on imb*sign(mv)."""
    t, c = DF.ct_sec.values, DF.close.values
    bv, sv = DF.buy_vol.values, DF.sell_vol.values
    out = np.zeros(len(DF), dtype=np.int64)
    for _, idx in DF.groupby("session", sort=False).indices.items():
        tt, cc = t[idx], c[idx]
        m = (tt >= 8*3600+30*60) & (tt < START)
        if not m.any(): continue
        where = np.where(m)[0]
        onv = cc[where[0]] - cc[0]
        mv = cc[where[-1]] - cc[where[0]]
        if mv == 0: continue
        hot = not (ON_LO <= abs(onv) <= ON_HI)
        agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
        if hot: sz = 9 if agree else 7
        else:   sz = 6 if not agree else 0
        imb = (bv[idx][where].sum() - sv[idx][where].sum()) / max(bv[idx][where].sum() + sv[idx][where].sum(), 1)
        oneway = imb * np.sign(mv) >= IMB_TH
        if sz > 0 and oneway: sz = max(1, sz - 2)
        side = -1 if mv >= 0 else 1
        do_flip = {"all": True, "none": False, "hot": hot, "cold": not hot}[flip_mode]
        if flip_imb_gate is not None:
            do_flip = do_flip and (imb * np.sign(mv) * flip_imb_sign >= flip_imb_gate)
        pos, entry, state = 0, 0.0, 0
        for k in range(len(idx)):
            if tt[k] < START or tt[k] >= EXIT:
                pos = 0
            elif state == 0:
                pos, entry, state = side * sz, cc[k], 1
            elif pos != 0 and state == 1:
                pnl = (cc[k] - entry) * np.sign(pos)
                if pnl <= -STOP_PTS:
                    if agree or not do_flip:
                        pos, state = 0, 4
                    else:
                        pos, entry, state = -side * max(1, sz + flip_sz_delta), cc[k], 3
                elif pnl >= TGT_PTS:
                    pos, state = 0, 4
            elif pos != 0 and state == 3:
                pnl = (cc[k] - entry) * np.sign(pos)
                if pnl <= -F_STOP or pnl >= F_TGT:
                    pos, state = 0, 4
            out[idx[k]] = pos
    return out

if __name__ == "__main__":
    score_signals(lambda df: build(), "base repro (flip all, +1, 40/60)")
    score_signals(lambda df: build(flip_mode="none"), "no flip (fade only)")
    score_signals(lambda df: build(flip_mode="hot"), "flip hot only")
    score_signals(lambda df: build(flip_mode="cold"), "flip cold only")
    score_signals(lambda df: build(flip_sz_delta=0), "flip same size")
    score_signals(lambda df: build(flip_sz_delta=2), "flip sz+2")
    score_signals(lambda df: build(flip_sz_delta=-2), "flip sz-2")
    score_signals(lambda df: build(flip_imb_gate=0.05, flip_imb_sign=1), "flip only if push one-sided")
    score_signals(lambda df: build(flip_imb_gate=0.05, flip_imb_sign=-1), "flip only if push balanced")
