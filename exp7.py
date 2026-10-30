"""Flip-leg refinements: no stop, target only, exit time."""
from harness import score_signals
import numpy as np, pandas as pd
from exp2 import gate, START

def build5(s2_stop=40.0, s2_tgt=80.0, exit_t=14 * 3600 + 30 * 60, flip_exit=None):
    def signals(df):
        out = np.zeros(len(df), dtype=np.int64)
        for _, (idx, side, sz, mv, onv) in gate(df).items():
            if sz == 0:
                continue
            hot = not (50 <= abs(onv) <= 175)
            tt, cc = df.ct_sec.values[idx], df.close.values[idx]
            pos, entry, state = 0, 0.0, 0
            for k in range(len(idx)):
                if tt[k] < START or tt[k] >= exit_t:
                    pos = 0
                elif state == 0:
                    pos, entry, state = side * sz, cc[k], 1
                elif state == 1 and pos != 0:
                    pnl = (cc[k] - entry) * np.sign(pos)
                    if pnl <= -40:
                        pos, entry, state = 0, 0.0, 2
                        if hot:
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

score_signals(build5(), "flip hot 40/80 exit1430 (0.5137)")
score_signals(build5(s2_stop=1e9), "flip hot no stop, tgt 80")
score_signals(build5(s2_stop=1e9, s2_tgt=1e9), "flip hot ride to 1430")
score_signals(build5(s2_stop=40, s2_tgt=1e9), "flip hot stop40 no tgt")
score_signals(build5(exit_t=13 * 3600), "flip hot exit 13:00")
score_signals(build5(exit_t=12 * 3600), "flip hot exit 12:00")
score_signals(build5(exit_t=15 * 3600), "flip hot exit 15:00")
