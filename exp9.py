"""Single-knob variants on 0.5137: entry time, overnight window, cold+agree momentum, flow gate."""
from harness import score_signals
import numpy as np, pandas as pd

START_D, EXIT = 8 * 3600 + 30 * 60, 14 * 3600 + 30 * 60

def build(entry_min=5, on_from=0, cold_agree_mode="skip", flow_gate=None, stop=40.0, tgt=80.0):
    START = 8 * 3600 + (30 + entry_min) * 60
    def signals(df):
        out = np.zeros(len(df), dtype=np.int64)
        t, c, bv, sv = df.ct_sec.values, df.close.values, df.buy_vol.values, df.sell_vol.values
        for _, idx in df.groupby("session", sort=False).indices.items():
            tt, cc = t[idx], c[idx]
            m = (tt >= START_D) & (tt < START)
            if not m.any():
                continue
            w = np.where(m)[0]
            onv = cc[w[0]] - cc[max(np.searchsorted(tt, on_from) - 1, 0)] if on_from else cc[w[0]] - cc[0]
            mv = cc[w[-1]] - cc[w[0]]
            if mv == 0:
                continue
            hot = not (50 <= abs(onv) <= 175)
            agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
            if hot:
                sz = 9 if agree else 8
                side = -1 if mv >= 0 else 1
            else:
                if not agree:
                    sz, side = 6, (-1 if mv >= 0 else 1)
                elif cold_agree_mode == "mom":
                    sz, side = 6, (1 if mv >= 0 else -1)
                else:
                    continue
            if flow_gate is not None:
                net = (bv[idx[w]] - sv[idx[w]]).sum()
                if flow_gate == "exhaust" and np.sign(net) == np.sign(mv):
                    continue
                if flow_gate == "agree" and np.sign(net) != np.sign(mv):
                    continue
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

score_signals(build(), "baseline entry 8:35 (0.5137)")
score_signals(build(entry_min=4), "entry 8:34")
score_signals(build(entry_min=6), "entry 8:36")
score_signals(build(entry_min=10), "entry 8:40")
score_signals(build(on_from=17 * 3600), "overnight from 17:00 (session open)")
score_signals(build(on_from=2 * 3600), "overnight from 02:00")
score_signals(build(cold_agree_mode="mom"), "cold+agree momentum 6")
score_signals(build(cold_agree_mode="mom") , "cold+agree momentum dup")
score_signals(build(flow_gate="exhaust"), "fade only when flow disagrees (exhaustion)")
