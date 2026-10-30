"""gs-a: verify HEAD 0.6018 + combined flip variants. Uses gs_ files only (avoid teammate collisions)."""
from harness import score_signals
import numpy as np, pandas as pd
from numba import njit

START, EXIT = 8 * 3600 + 35 * 60, 14 * 3600 + 30 * 60
STOP_PTS, TGT_PTS = 40.0, 80.0
ON_LO, ON_HI = 50.0, 175.0
IMB_TH = 0.05

def mk(dis_flip=True, hot_agree_flip=False, f_stop=40.0, f_tgt=60.0, ha_stop=40.0, ha_tgt=80.0,
       imb_sh=2, hot_dis_sz=7):
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
            if sz > 0 and imb * np.sign(mv) >= IMB_TH:
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
                        if dis_flip and not agree:
                            pos, entry, state = -side * (sz + 1), cc[k], 3
                        elif hot_agree_flip and hot and agree:
                            pos, entry, state = -side * sz, cc[k], 5
                        else:
                            pos, state = 0, 4
                    elif pnl >= TGT_PTS:
                        pos, state = 0, 4
                elif state == 3 and pos != 0:
                    pnl = (cc[k] - entry) * np.sign(pos)
                    if pnl <= -f_stop or pnl >= f_tgt:
                        pos, state = 0, 4
                elif state == 5 and pos != 0:
                    pnl = (cc[k] - entry) * np.sign(pos)
                    if pnl <= -ha_stop or pnl >= ha_tgt:
                        pos, state = 0, 4
                out[idx[k]] = pos
        return out
    return signals

score_signals(mk(), "HEAD disagree flip 40/60 (0.6018?)")
score_signals(mk(hot_agree_flip=True), "+ hot+agree flip 40/80 full")
score_signals(mk(hot_agree_flip=True, ha_tgt=60), "+ hot+agree flip 40/60")
score_signals(mk(f_tgt=80), "dis flip 40/80")
score_signals(mk(f_stop=50, f_tgt=80), "dis flip 50/80")
score_signals(mk(f_tgt=70), "dis flip 40/70")
