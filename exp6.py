"""Refine flip full hot: s2 params, bucket split, tighter leg1 on hot."""
from harness import score_signals
import numpy as np, pandas as pd
from exp2 import gate, START, EXIT

def build4(s2_stop=40.0, s2_tgt=80.0, hot_stop1=None, flip_buckets=("hot+agree", "hot+dis"),
           s3=False):
    def signals(df):
        out = np.zeros(len(df), dtype=np.int64)
        for _, (idx, side, sz, mv, onv) in gate(df).items():
            if sz == 0:
                continue
            hot = not (50 <= abs(onv) <= 175)
            agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
            b = ("hot" if hot else "cold") + ("+agree" if agree else "+dis")
            do_flip = b in flip_buckets
            st1 = hot_stop1 if (hot and hot_stop1) else 40.0
            tt, cc = df.ct_sec.values[idx], df.close.values[idx]
            pos, entry, state = 0, 0.0, 0
            for k in range(len(idx)):
                if tt[k] < START or tt[k] >= EXIT:
                    pos = 0
                elif state == 0:
                    pos, entry, state = side * sz, cc[k], 1
                elif state == 1 and pos != 0:
                    pnl = (cc[k] - entry) * np.sign(pos)
                    if pnl <= -st1:
                        pos, entry, state = 0, 0.0, 2
                        if do_flip:
                            pos, entry, state = -side * sz, cc[k], 3
                    elif pnl >= 80:
                        pos, state = 0, 4
                elif state == 3 and pos != 0:
                    pnl = (cc[k] - entry) * np.sign(pos)
                    if pnl <= -s2_stop or pnl >= s2_tgt:
                        pos, state = 0, 4
                out[idx[k]] = pos
        return out
    return signals

score_signals(build4(), "flip full hot 40/80 (0.5137)")
score_signals(build4(s2_stop=30, s2_tgt=60), "flip hot 30/60")
score_signals(build4(s2_stop=50, s2_tgt=100), "flip hot 50/100")
score_signals(build4(s2_stop=40, s2_tgt=120), "flip hot 40/120")
score_signals(build4(s2_stop=60, s2_tgt=80), "flip hot 60/80")
score_signals(build4(flip_buckets=("hot+agree",)), "flip only hot+agree")
score_signals(build4(flip_buckets=("hot+dis",)), "flip only hot+dis")
score_signals(build4(hot_stop1=30), "flip hot, leg1 stop 30 on hot")
score_signals(build4(hot_stop1=25), "flip hot, leg1 stop 25 on hot")
