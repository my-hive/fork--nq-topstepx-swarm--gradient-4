"""Baseline + reversed-trade experiments."""
from harness import score_signals
import numpy as np, pandas as pd
from numba import njit

START, EXIT = 8 * 3600 + 35 * 60, 14 * 3600 + 30 * 60
ON_LO, ON_HI = 50.0, 175.0

def gate(df):
    """Returns per-session arrays: side, size, entry_index (first idx >= START)."""
    out = {}
    t, c = df.ct_sec.values, df.close.values
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
        sz = (9 if agree else 8) if hot else (6 if not agree else 0)
        side = -1 if mv >= 0 else 1
        out[_] = (idx, side, sz, mv, onv)
    return out

def build(re_entry=False, re_side="same", re_frac=0.5, rev_after_stop=False):
    def signals(df):
        out = np.zeros(len(df), dtype=np.int64)
        for _, (idx, side, sz, mv, onv) in gate(df).items():
            tt, cc = df.ct_sec.values[idx], df.close.values[idx]
            if sz == 0:
                continue
            pos, entry, state = 0, 0.0, 0  # state: 0=await entry,1=in trade1,2=stopped,3=done
            re_done = False
            for k in range(len(idx)):
                if tt[k] < START or tt[k] >= EXIT:
                    pos = 0
                elif state == 0:
                    pos, entry, state = side * sz, cc[k], 1
                elif state == 1 and pos != 0:
                    pnl = (cc[k] - entry) * np.sign(pos)
                    if pnl <= -40 or pnl >= 80:
                        pos = 0
                        state = 3 if pnl >= 80 else 2
                        if state == 2 and re_entry and not re_done:
                            re_done = True
                            s2 = side if re_side == "same" else -side
                            pos, entry, state = s2 * max(1, int(sz * re_frac)), cc[k], 1
                elif state == 1 and pos == 0:
                    pass
                out[idx[k]] = pos
        return out
    return signals

if __name__ == "__main__":
    score_signals(build(), "baseline (9,8,6,0)")
    score_signals(build(re_entry=True), "re-entry same side half size")
    score_signals(build(re_entry=True, re_side="opp"), "re-entry opposite side half")
    score_signals(build(re_entry=True, re_frac=1.0), "re-entry same side full")
