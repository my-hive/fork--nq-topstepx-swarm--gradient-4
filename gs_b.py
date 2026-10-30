"""gs-b: sweep flip brackets, flip size, imbalance params, flip bucket splits on 0.6018."""
from harness import score_signals
import numpy as np, pandas as pd

START, EXIT = 8 * 3600 + 35 * 60, 14 * 3600 + 30 * 60
STOP_PTS, TGT_PTS = 40.0, 80.0
ON_LO, ON_HI = 50.0, 175.0

def mk(f_stop=40.0, f_tgt=60.0, f_dsz=1, imb_th=0.05, imb_sh=2, hot_dis_sz=7,
       flip_hot=True, flip_cold=True):
    def signals(df):
        out = np.zeros(len(df), dtype=np.int64)
        t, c = df.ct_sec.values, df.close.values
        bv, sv = df.buy_vol.values, df.sell_vol.values
        for _, idx in df.groupby("session", sort=False).indices.items():
            tt, cc = t[idx], c[idx]
            m = (tt >= 8 * 3600 + 30 * 60) & (tt < START)
            if not m.any():
                continue
            w = np.where(m)[0]
            onv = cc[w[0]] - cc[0]
            mv = cc[w[-1]] - cc[w[0]]
            if mv == 0:
                continue
            hot = not (ON_LO <= abs(onv) <= ON_HI)
            agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
            if hot:
                sz = 9 if agree else hot_dis_sz
            else:
                sz = 6 if not agree else 0
            imb = (bv[idx][w].sum() - sv[idx][w].sum()) / max(bv[idx][w].sum() + sv[idx][w].sum(), 1)
            if sz > 0 and imb * np.sign(mv) >= imb_th:
                sz = max(1, sz - imb_sh)
            if sz == 0:
                continue
            side = -1 if mv >= 0 else 1
            pos, entry, state = 0, 0.0, 0
            for k in range(len(idx)):
                if tt[k] < START or tt[k] >= EXIT:
                    pos = 0
                elif state == 0:
                    pos, entry, state = side * sz, cc[k], 1
                elif state == 1 and pos != 0:
                    pnl = (cc[k] - entry) * np.sign(pos)
                    if pnl <= -STOP_PTS:
                        if not agree and (hot if flip_hot else True) and (not hot if not flip_cold else True):
                            pos, entry, state = -side * (sz + f_dsz), cc[k], 3
                        else:
                            pos, state = 0, 4
                    elif pnl >= TGT_PTS:
                        pos, state = 0, 4
                elif state == 3 and pos != 0:
                    pnl = (cc[k] - entry) * np.sign(pos)
                    if pnl <= -f_stop or pnl >= f_tgt:
                        pos, state = 0, 4
                out[idx[k]] = pos
        return out
    return signals

score_signals(mk(), "HEAD 0.6018")
for fs, ft in ((35, 55), (30, 50), (40, 50), (40, 55), (45, 60), (35, 60), (40, 65)):
    score_signals(mk(f_stop=fs, f_tgt=ft), f"flip {fs}/{ft}")
score_signals(mk(f_dsz=0), "flip size sz+0")
score_signals(mk(f_dsz=2), "flip size sz+2")
score_signals(mk(flip_cold=False), "flip hot+dis only")
score_signals(mk(flip_hot=False), "flip cold+dis only")
