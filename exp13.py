"""exp13: time-stop (slow fade = trend against you) + partial-profit variants on top of 0.5382 base."""
import numpy as np, pandas as pd
from harness import DF, SESS, score_signals

START, EXIT = 8*3600+35*60, 14*3600+30*60
STOP_PTS, TGT_PTS = 40.0, 80.0
ON_LO, ON_HI = 50.0, 175.0
IMB_TH, IMB_SH = 0.05, 3

def make(tstop=None, partial=None, part_at=40.0):
    """tstop: sec-of-day; close trade if still open at tstop. partial: at +part_at pts, cut size to partial*orig."""
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
            BV, SV = bv[idx][w].sum(), sv[idx][w].sum()
            ia = (BV - SV) / max(BV + SV, 1) * np.sign(mv)
            if sz > 0 and ia >= IMB_TH: sz = max(1, sz - 3)
            side = -1 if mv >= 0 else 1
            pos, entry, done, took = 0, 0.0, False, False
            for k in range(len(idx)):
                if tt[k] < START or tt[k] >= EXIT:
                    pos = 0
                elif pos == 0 and not done:
                    pos, entry, done = side*sz, cc[k], True
                elif pos != 0:
                    pnl = (cc[k]-entry) * np.sign(pos)
                    if partial is not None and not took and pnl >= part_at:
                        pos = int(side * max(1, round(sz*partial))); took = True
                    elif pnl <= -STOP_PTS or pnl >= TGT_PTS:
                        pos = 0
                    elif tstop is not None and tt[k] >= tstop:
                        pos = 0
                out[idx[k]] = pos
        return out
    return signals

H = lambda h, m: h*3600 + m*60
if __name__ == "__main__":
    score_signals(make(), "base (expect .5382)")
    for ts in [H(9,0), H(9,30), H(10,0), H(11,0), H(12,0)]:
        score_signals(make(tstop=ts), f"time-stop {ts//3600}:{(ts%3600)//60:02d}")
    score_signals(make(partial=0.5, part_at=40), "partial 50% at +40")
    score_signals(make(partial=0.5, part_at=30), "partial 50% at +30")
    score_signals(make(partial=0.5, part_at=50), "partial 50% at +50")
