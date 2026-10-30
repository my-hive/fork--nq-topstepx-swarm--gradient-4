"""mg8 lane: cross-day loss conditioning on the 0.6770 base.
Reconstruct prev session's strategy PnL (deterministic, prefix-safe) and shrink/skip today
after big loss days. All failures of the base are MLL busts from clustered loss days.
Base = dead zone 50-225, hot+agree 8 / hot+dis 7 / cold+dis 6, shrink-2 on imbalance, flip sz+1 40/60.
"""
import numpy as np, pandas as pd
import sys
sys.path.insert(0, ".")
from harness import score_signals

START, EXIT = 8 * 3600 + 35 * 60, 14 * 3600 + 30 * 60
STOP_PTS, TGT_PTS = 40.0, 80.0
ON_LO, ON_HI = 50.0, 225.0
IMB_TH = 0.05
F_STOP_PTS, F_TGT_PTS = 40.0, 60.0


def make(variant):
    """variant: None=base, else dict(mode=..., kwargs)"""
    def signals(df: pd.DataFrame) -> np.ndarray:
        out = np.zeros(len(df), dtype=np.int64)
        t, c = df.ct_sec.values, df.close.values
        bv, sv = df.buy_vol.values, df.sell_vol.values
        prev_pnl_pts = 0.0        # pts*contracts of prev traded session (signed)
        prev_stop = False         # prev session had a stop-out (any leg)
        prev_loss_usd = 0.0
        n_consec_loss = 0
        for _, idx in df.groupby("session", sort=False).indices.items():
            tt, cc = t[idx], c[idx]
            m = (tt >= 8 * 3600 + 30 * 60) & (tt < START)
            if not m.any():
                continue
            where = np.where(m)[0]
            onv = cc[where[0]] - cc[0]
            mv = cc[where[-1]] - cc[where[0]]
            if mv == 0:
                prev_pnl_pts, prev_stop, n_consec_loss = 0.0, False, 0
                continue
            hot = not (ON_LO <= abs(onv) <= ON_HI)
            agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
            if hot:
                sz = 8 if agree else 7
            else:
                sz = 6 if not agree else 0
            imb = (bv[idx][where].sum() - sv[idx][where].sum()) / max(
                bv[idx][where].sum() + sv[idx][where].sum(), 1)
            if sz > 0 and imb * np.sign(mv) >= IMB_TH:
                sz = max(1, sz - 2)
            side = -1 if mv >= 0 else 1

            # ---- cross-day conditioning ----
            if variant is not None:
                mode = variant["mode"]
                if mode == "stop_skip" and prev_stop:
                    sz = 0
                elif mode == "stop_minus3" and prev_stop:
                    sz = max(1, sz - 3)
                elif mode == "stop_half" and prev_stop:
                    sz = max(1, sz // 2)
                elif mode == "loss400_half" and prev_loss_usd <= -400:
                    sz = max(1, sz // 2)
                elif mode == "loss400_skip" and prev_loss_usd <= -400:
                    sz = 0
                elif mode == "clamp_skip" and prev_loss_usd <= -900:
                    sz = 0
                elif mode == "anyloss_half" and prev_pnl_pts < 0:
                    sz = max(1, sz // 2)
                elif mode == "two_loss_skip" and n_consec_loss >= 2:
                    sz = 0
                elif mode == "two_loss_half" and n_consec_loss >= 2:
                    sz = max(1, sz // 2)

            # ---- per-session trade sim with pnl booking ----
            pos, entry, state = 0, 0.0, 0
            day_pnl = 0.0
            stopped = False
            for k in range(len(idx)):
                if tt[k] < START or tt[k] >= EXIT:
                    if pos != 0:
                        day_pnl += pos * (cc[k] - entry)
                        pos = 0
                elif state == 0:
                    pos, entry, state = side * sz, cc[k], 1
                elif pos != 0 and state == 1:
                    p = (cc[k] - entry) * np.sign(pos)
                    if p <= -STOP_PTS:
                        day_pnl += pos * (cc[k] - entry)
                        stopped = True
                        if agree:
                            pos, state = 0, 4
                        else:
                            pos, entry, state = -side * (sz + 1), cc[k], 3
                    elif p >= TGT_PTS:
                        day_pnl += pos * (cc[k] - entry)
                        pos, state = 0, 4
                elif pos != 0 and state == 3:
                    p = (cc[k] - entry) * np.sign(pos)
                    if p <= -F_STOP_PTS or p >= F_TGT_PTS:
                        day_pnl += pos * (cc[k] - entry)
                        if p <= -F_STOP_PTS:
                            stopped = True
                        pos, state = 0, 4
                out[idx[k]] = pos
            if pos != 0:
                day_pnl += pos * (cc[-1] - entry)
            if sz == 0:
                # no trade today; keep loss memory? reset single-day loss flags
                prev_pnl_pts, prev_stop, prev_loss_usd = 0.0, False, 0.0
            else:
                prev_pnl_pts = day_pnl
                prev_stop = stopped
                prev_loss_usd = day_pnl * 2.0
                n_consec_loss = n_consec_loss + 1 if day_pnl < 0 else 0
        return out
    return signals


DF = None
if __name__ == "__main__":
    from harness import DF as _DF
    DF = _DF
    b = make(None)
    score_signals(b, "base 0.6770 (recheck)")
    for mode in ["stop_skip", "stop_minus3", "stop_half", "loss400_half",
                 "loss400_skip", "clamp_skip", "anyloss_half", "two_loss_skip", "two_loss_half"]:
        try:
            score_signals(make({"mode": mode}), f"{mode}")
        except Exception as e:
            print(mode, "ERR", e)
