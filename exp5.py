"""Flip leg 2 gated by hot/cold and mv size."""
from harness import score_signals
import numpy as np, pandas as pd
from exp2 import gate, START, EXIT

def build3(flip="hot", frac=0.5, min_mv=0.0, s2_stop=40.0, s2_tgt=80.0):
    def signals(df):
        out = np.zeros(len(df), dtype=np.int64)
        for _, (idx, side, sz, mv, onv) in gate(df).items():
            if sz == 0:
                continue
            hot = not (50 <= abs(onv) <= 175)
            do_flip = (flip == "all") or (flip == "hot" and hot) or (flip == "cold" and not hot)
            if abs(mv) < min_mv:
                do_flip = False
            tt, cc = df.ct_sec.values[idx], df.close.values[idx]
            pos, entry, state = 0, 0.0, 0
            for k in range(len(idx)):
                if tt[k] < START or tt[k] >= EXIT:
                    pos = 0
                elif state == 0:
                    pos, entry, state = side * sz, cc[k], 1
                elif state == 1 and pos != 0:
                    pnl = (cc[k] - entry) * np.sign(pos)
                    if pnl <= -40:
                        pos, entry, state = 0, 0.0, 2
                        if do_flip:
                            pos, entry, state = -side * max(1, int(sz * frac)), cc[k], 3
                    elif pnl >= 80:
                        pos, state = 0, 4
                elif state == 3 and pos != 0:
                    pnl = (cc[k] - entry) * np.sign(pos)
                    if pnl <= -s2_stop or pnl >= s2_tgt:
                        pos, state = 0, 4
                out[idx[k]] = pos
        return out
    return signals

score_signals(build3(flip="hot", frac=0.5), "flip .5 hot only")
score_signals(build3(flip="hot", frac=1.0), "flip full hot only")
score_signals(build3(flip="hot", frac=0.75), "flip .75 hot only")
score_signals(build3(flip="hot", frac=0.5, min_mv=20), "flip .5 hot + mv>=20")
score_signals(build3(flip="hot", frac=1.0, min_mv=20), "flip full hot + mv>=20")
score_signals(build3(flip="all", frac=0.5), "flip .5 all (known .4753)")
