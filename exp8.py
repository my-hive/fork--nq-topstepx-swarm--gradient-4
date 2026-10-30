"""Grid: stop/tgt for both legs + size scale, flip active."""
from harness import score_signals
import numpy as np, pandas as pd
from exp2 import gate, START, EXIT

def build6(stop=40.0, tgt=80.0, smult=1.0):
    def signals(df):
        out = np.zeros(len(df), dtype=np.int64)
        for _, (idx, side, sz, mv, onv) in gate(df).items():
            if sz == 0:
                continue
            hot = not (50 <= abs(onv) <= 175)
            sz = max(1, int(round(sz * smult)))
            tt, cc = df.ct_sec.values[idx], df.close.values[idx]
            pos, entry, state = 0, 0.0, 0
            for k in range(len(idx)):
                if tt[k] < START or tt[k] >= EXIT:
                    pos = 0
                elif state == 0:
                    pos, entry, state = side * sz, cc[k], 1
                elif state == 1 and pos != 0:
                    pnl = (cc[k] - entry) * np.sign(pos)
                    if pnl <= -stop:
                        pos, entry, state = 0, 0.0, 2
                        if hot:
                            pos, entry, state = -side * sz, cc[k], 3
                    elif pnl >= tgt:
                        pos, state = 0, 4
                elif state == 3 and pos != 0:
                    pnl = (cc[k] - entry) * np.sign(pos)
                    if pnl <= -stop or pnl >= tgt:
                        pos, state = 0, 4
                out[idx[k]] = pos
        return out
    return signals

for tgt in (70, 80, 90, 100):
    for stop in (30, 40, 50):
        score_signals(build6(stop, tgt), f"stop={stop:.0f} tgt={tgt:.0f}")
score_signals(build6(smult=1.2), "smult 1.2")
score_signals(build6(smult=0.8), "smult 0.8")
