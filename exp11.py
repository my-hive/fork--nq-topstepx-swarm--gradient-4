"""exp11: on top of 0.5382 shrink-3 — (a) counter-flow boost, (b) push-direction asymmetry, (c) total-volume gate."""
import numpy as np, pandas as pd
from harness import DF, SESS, score_signals

START, EXIT = 8*3600+35*60, 14*3600+30*60
STOP_PTS, TGT_PTS = 40.0, 80.0
ON_LO, ON_HI = 50.0, 175.0
IMB_TH, IMB_SH = 0.05, 3

def make(imb_sh=3, counter_boost=0, counter_th=-0.05, skip_up=False, skip_down=False,
         vol_pct=None, vol_delta=0):
    """vol_pct: skip trades when 8:30-8:35 total aggressor vol below quantile vol_pct.
    vol_delta: size change when below quantile (negative shrink)."""
    def signals(df):
        out = np.zeros(len(df), dtype=np.int64)
        t, c = df.ct_sec.values, df.close.values
        bv, sv = df.buy_vol.values, df.sell_vol.values
        vols = []
        sess_idx = list(df.groupby("session", sort=False).indices.items())
        for _, idx in sess_idx:
            tt = t[idx]
            m = (tt >= 8*3600+30*60) & (tt < START)
            if m.any():
                w = np.where(m)[0]
                vols.append(bv[idx][w].sum() + sv[idx][w].sum())
            else:
                vols.append(np.nan)
        thr = np.nanquantile(vols, vol_pct) if vol_pct is not None else None
        for i, (_, idx) in enumerate(sess_idx):
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
            if sz > 0 and ia >= IMB_TH:
                sz = max(1, sz - imb_sh)
            if counter_boost and sz > 0 and ia <= counter_th:
                sz = min(50, sz + counter_boost)
            if skip_up and mv > 0: sz = 0
            if skip_down and mv < 0: sz = 0
            if thr is not None and vols[i] < thr and sz > 0:
                sz = 0 if vol_delta == 0 else max(1, sz + vol_delta)
            side = -1 if mv >= 0 else 1
            pos, entry, done = 0, 0.0, False
            for k in range(len(idx)):
                if tt[k] < START or tt[k] >= EXIT: pos = 0
                elif pos == 0 and not done: pos, entry, done = side*sz, cc[k], True
                elif pos != 0:
                    pnl = (cc[k]-entry) * np.sign(pos)
                    if pnl <= -STOP_PTS or pnl >= TGT_PTS: pos = 0
                out[idx[k]] = pos
        return out
    return signals

if __name__ == "__main__":
    score_signals(make(), "base shrink-3 (expect .5382)")
    score_signals(make(counter_boost=2), "counter-boost +2 th-.05")
    score_signals(make(counter_boost=3), "counter-boost +3 th-.05")
    score_signals(make(counter_boost=2, counter_th=-0.03), "counter-boost +2 th-.03")
    score_signals(make(skip_up=True), "skip up-push fades")
    score_signals(make(skip_down=True), "skip down-push fades")
    score_signals(make(vol_pct=0.25, vol_delta=-2), "low-vol(25%) shrink2")
    score_signals(make(vol_pct=0.25, vol_delta=0), "low-vol(25%) skip")
