"""exp18: flip-leg trailing management, mv-dead-zone tweaks, fade tgt 100 — on flip base 0.6018."""
import numpy as np, pandas as pd
from harness import DF, SESS, score_signals

START, EXIT = 8*3600+35*60, 14*3600+30*60
ON_LO, ON_HI = 50.0, 175.0

def make(f_trail=None, f_be=None, mv_lo=20.0, mv_hi=35.0, mv_mode=None, tgt=80.0):
    """f_trail: trail flip stop by X once pnl >= 30. f_be: flip stop to +f_be after +30.
    mv_mode: 'skip'/'shrink' for |mv| in [mv_lo, mv_hi)."""
    def signals(df):
        out = np.zeros(len(df), dtype=np.int64)
        t, c = df.ct_sec.values, df.close.values
        bv, sv = df.buy_vol.values, df.sell_vol.values
        for _, idx in df.groupby("session", sort=False).indices.items():
            tt, cc = t[idx], c[idx]
            m = (tt >= 8*3600+30*60) & (tt < START)
            if not m.any(): continue
            w = np.where(m)[0]
            onv = cc[w[0]] - cc[0]
            mv = cc[w[-1]] - cc[w[0]]
            if mv == 0: continue
            hot = not (ON_LO <= abs(onv) <= ON_HI)
            agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
            sz = (9 if agree else 7) if hot else (6 if not agree else 0)
            ia = (bv[idx][w].sum() - sv[idx][w].sum()) / max(bv[idx][w].sum() + sv[idx][w].sum(), 1) * np.sign(mv)
            if sz > 0 and ia >= 0.05: sz = max(1, sz - 2)
            if mv_mode and sz > 0 and mv_lo <= abs(mv) < mv_hi:
                if mv_mode == "skip": sz = 0
                elif mv_mode == "shrink": sz = max(1, sz - 2)
            side = -1 if mv >= 0 else 1
            pos, entry, state, fst = 0, 0.0, 0, -40.0
            for k in range(len(idx)):
                if tt[k] < START or tt[k] >= EXIT: pos = 0
                elif state == 0:
                    if sz == 0: state = 4
                    else: pos, entry, state = side*sz, cc[k], 1
                elif pos != 0 and state == 1:
                    pnl = (cc[k]-entry)*np.sign(pos)
                    if pnl <= -40.0:
                        if agree: pos, state = 0, 4
                        else: pos, entry, state, fst = -side*(sz+1), cc[k], 3, -40.0
                    elif pnl >= tgt: pos, state = 0, 4
                elif pos != 0 and state == 3:
                    pnl = (cc[k]-entry)*np.sign(pos)
                    if f_trail is not None and pnl >= 30.0:
                        fst = max(fst, pnl - f_trail)
                    if f_be is not None and pnl >= 30.0:
                        fst = max(fst, f_be)
                    if pnl <= fst or pnl >= 60.0: pos, state = 0, 4
                out[idx[k]] = pos
        return out
    return signals

if __name__ == "__main__":
    score_signals(make(), "base (expect .6018)")
    score_signals(make(f_trail=15.0), "flip trail 15 after +30")
    score_signals(make(f_trail=25.0), "flip trail 25 after +30")
    score_signals(make(f_trail=40.0), "flip trail 40 after +30")
    score_signals(make(f_be=0.0), "flip BE after +30")
    score_signals(make(f_be=10.0), "flip stop +10 after +30")
    score_signals(make(mv_mode="skip"), "skip |mv| 20-35")
    score_signals(make(mv_mode="shrink"), "shrink |mv| 20-35")
    score_signals(make(tgt=100.0), "fade tgt 100")
    score_signals(make(f_trail=15.0, mv_mode="shrink"), "trail15 + mv shrink")
