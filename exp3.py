"""Tune opposite-side re-entry: size frac, stop/target for leg 2, maybe third leg."""
from harness import score_signals
import numpy as np, pandas as pd
from exp2 import gate, START, EXIT

def build2(re_frac=0.5, s2_stop=40.0, s2_tgt=80.0, s2_size=None):
    def signals(df):
        out = np.zeros(len(df), dtype=np.int64)
        for _, (idx, side, sz, mv, onv) in gate(df).items():
            tt, cc = df.ct_sec.values[idx], df.close.values[idx]
            if sz == 0:
                continue
            pos, entry, state = 0, 0.0, 0
            for k in range(len(idx)):
                if tt[k] < START or tt[k] >= EXIT:
                    pos = 0
                elif state == 0:
                    pos, entry, state = side * sz, cc[k], 1
                elif state == 1 and pos != 0:
                    pnl = (cc[k] - entry) * np.sign(pos)
                    if state == 1 and pnl <= -40:
                        pos, entry, state = 0, 0.0, 2
                        sz2 = s2_size if s2_size else max(1, int(sz * re_frac))
                        pos, entry, state = -side * sz2, cc[k], 3
                    elif pnl >= 80:
                        pos = 0; state = 4
                elif state == 3 and pos != 0:
                    pnl = (cc[k] - entry) * np.sign(pos)
                    if pnl <= -s2_stop or pnl >= s2_tgt:
                        pos = 0; state = 4
                out[idx[k]] = pos
        return out
    return signals

score_signals(build2(re_frac=1.0), "flip full size 40/80")
score_signals(build2(re_frac=0.75), "flip .75 size 40/80")
score_signals(build2(re_frac=0.5), "flip .5 size 40/80 (exp2)")
score_signals(build2(re_frac=0.5, s2_tgt=120), "flip .5 tgt120")
score_signals(build2(re_frac=0.5, s2_stop=60, s2_tgt=120), "flip .5 60/120")
score_signals(build2(re_frac=0.5, s2_stop=30, s2_tgt=60), "flip .5 30/60")
score_signals(build2(s2_size=9), "flip fixed 9")
score_signals(build2(s2_size=6), "flip fixed 6")
