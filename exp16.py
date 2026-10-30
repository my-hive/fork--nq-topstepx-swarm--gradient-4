"""exp16: cut the double-loss tail (all eval failures are MLL busts, -1008 DLL days): flip-size/stop/timeout sweep."""
import numpy as np, pandas as pd
from harness import DF, SESS, score_signals

START, EXIT = 8*3600+35*60, 14*3600+30*60
ON_LO, ON_HI = 50.0, 175.0

def make(f_add=1, f_stop=40.0, f_tgt=60.0, f_tstop=None, imb_sh=2):
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
            if sz > 0 and ia >= 0.05: sz = max(1, sz - imb_sh)
            side = -1 if mv >= 0 else 1
            pos, entry, state = 0, 0.0, 0
            flip_sz = max(1, sz + f_add)
            for k in range(len(idx)):
                if tt[k] < START or tt[k] >= EXIT: pos = 0
                elif state == 0: pos, entry, state = side*sz, cc[k], 1
                elif pos != 0 and state == 1:
                    pnl = (cc[k]-entry)*np.sign(pos)
                    if pnl <= -40.0:
                        if agree: pos, state = 0, 4
                        else: pos, entry, state = -side*flip_sz, cc[k], 3
                    elif pnl >= 80.0: pos, state = 0, 4
                elif pos != 0 and state == 3:
                    pnl = (cc[k]-entry)*np.sign(pos)
                    if pnl <= -f_stop or pnl >= f_tgt: pos, state = 0, 4
                    elif f_tstop is not None and tt[k] >= f_tstop: pos, state = 0, 4
                out[idx[k]] = pos
        return out
    return signals

H = lambda h, m: h*3600+m*60
if __name__ == "__main__":
    score_signals(make(), "base (expect .6018)")
    for fa in [0, -1, -2]:
        score_signals(make(f_add=fa), f"flip add {fa:+d}")
    for fs in [25.0, 30.0, 35.0]:
        score_signals(make(f_stop=fs), f"flip stop {fs:.0f}")
    for ts in [H(10,0), H(11,0), H(12,30)]:
        score_signals(make(f_tstop=ts), f"flip tstop {ts//3600}:{(ts%3600)//60:02d}")
    score_signals(make(f_add=-1, f_stop=30.0), "flip add -1 stop 30")
    score_signals(make(imb_sh=3), "shrink3 + flip (h1 play)")
