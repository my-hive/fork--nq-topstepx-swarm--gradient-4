"""exp14: on top of 0.6018 disagree-flip base — sweep flip bracket/size, flip gating, shrink interaction."""
import numpy as np, pandas as pd
from harness import DF, SESS, score_signals

START, EXIT = 8*3600+35*60, 14*3600+30*60
ON_LO, ON_HI = 50.0, 175.0

def make(imb_sh=2, imb_th=0.05, f_stop=40.0, f_tgt=60.0, f_add=1,
         flip_hot_only=False, flip_imb_only=False, f_trail=None):
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
            if sz > 0 and ia >= imb_th: sz = max(1, sz - imb_sh)
            do_flip = not agree and sz > 0
            if flip_hot_only: do_flip = do_flip and hot
            if flip_imb_only: do_flip = do_flip and ia >= imb_th
            side = -1 if mv >= 0 else 1
            pos, entry, state = 0, 0.0, 0
            flip_sz = sz + f_add if do_flip else 0
            for k in range(len(idx)):
                if tt[k] < START or tt[k] >= EXIT: pos = 0
                elif state == 0:
                    pos, entry, state = side*sz, cc[k], 1
                elif pos != 0 and state == 1:
                    pnl = (cc[k]-entry)*np.sign(pos)
                    if pnl <= -40.0:
                        if agree or not do_flip: pos, state = 0, 4
                        else: pos, entry, state = -side*flip_sz, cc[k], 3
                    elif pnl >= 80.0: pos, state = 0, 4
                elif pos != 0 and state == 3:
                    pnl = (cc[k]-entry)*np.sign(pos)
                    if f_trail is not None and pnl >= f_trail:
                        pnl = pnl - f_trail + f_trail  # trail check below
                    if pnl <= -f_stop or pnl >= f_tgt: pos, state = 0, 4
                out[idx[k]] = pos
        return out
    return signals

if __name__ == "__main__":
    score_signals(make(), "base flip 40/60 +1 sh2 (expect .6018)")
    score_signals(make(f_tgt=80.0), "flip tgt 80")
    score_signals(make(f_tgt=50.0), "flip tgt 50")
    score_signals(make(f_stop=30.0), "flip stop 30")
    score_signals(make(f_stop=50.0), "flip stop 50")
    score_signals(make(f_add=2), "flip size +2")
    score_signals(make(f_add=0), "flip size +0")
    score_signals(make(flip_hot_only=True), "flip hot days only")
    score_signals(make(flip_imb_only=True), "flip only if push imb one-sided")
    score_signals(make(imb_sh=3), "shrink-3 with flip")
    score_signals(make(imb_sh=1), "shrink-1 with flip")
