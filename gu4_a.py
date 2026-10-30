"""gu4_a: on 0.6856 base — EXIT time variants, momentum re-entry after morning win,
and a 10:00 second fade of the 9:30-10:00 move (gated on morning outcome)."""
import numpy as np, pandas as pd
import sys
sys.path.insert(0, ".")
from harness import score_signals

START0 = 8 * 3600 + 35 * 60
ON_LO, ON_HI = 50.0, 225.0
IMB_TH = 0.05

def make(exit_t=14 * 3600 + 30 * 60, cont=None, pm_fade=None):
    """cont: dict(sz_add=-3, stop=30, tgt=60) momentum continuation after morning target hit.
    pm_fade: dict(sz=4, stop=30, tgt=60, exit=14*3600, gate='win'|'any'|'skip') 10:00 fade."""
    def signals(df: pd.DataFrame) -> np.ndarray:
        out = np.zeros(len(df), dtype=np.int64)
        t, c = df.ct_sec.values, df.close.values
        bv, sv = df.buy_vol.values, df.sell_vol.values
        for _, idx in df.groupby("session", sort=False).indices.items():
            tt, cc = t[idx], c[idx]
            m = (tt >= 8 * 3600 + 30 * 60) & (tt < START0)
            if not m.any():
                continue
            where = np.where(m)[0]
            onv = cc[where[0]] - cc[0]
            mv = cc[where[-1]] - cc[where[0]]
            if mv == 0:
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
            stop, tgt = (30.0, 60.0) if abs(onv) > 200 else (40.0, 80.0)
            side = -1 if mv >= 0 else 1
            pos, entry, state = 0, 0.0, 0  # 0 wait,1 fade,3 flip,4 done,5 cont,6 pm
            pm_done = False
            for k in range(len(idx)):
                if tt[k] >= exit_t:
                    pos = 0
                elif tt[k] < START0:
                    pos = 0
                elif state == 0:
                    if sz == 0:
                        state, pm_done = 4, False
                    else:
                        pos, entry, state = side * sz, cc[k], 1
                elif pos != 0 and state == 1:
                    pnl = (cc[k] - entry) * np.sign(pos)
                    if pnl <= -stop:
                        if agree and abs(onv) <= 200:
                            pos, state = 0, 4
                        else:
                            pos, entry, state = -side * (sz + 1), cc[k], 3
                    elif pnl >= tgt:
                        pos, state = 0, 4
                        if cont is not None:   # momentum continuation entry
                            pos, entry, state = np.sign(side * -1) * 0, entry, 4  # placeholder
                            pos, entry, state = (1 if side < 0 else -1) * max(1, sz + cont.get("sz_add", -3)), cc[k], 5
                elif pos != 0 and state == 5:
                    pnl = (cc[k] - entry) * np.sign(pos)
                    if pnl <= -cont.get("stop", 30.0) or pnl >= cont.get("tgt", 60.0):
                        pos, state = 0, 4
                elif pos != 0 and state == 3:
                    pnl = (cc[k] - entry) * np.sign(pos)
                    if pnl <= -40.0 or pnl >= 60.0:
                        pos, state = 0, 4
                # 10:00 second fade of 9:30-10:00 move
                if pm_fade is not None and not pm_done and state == 4 and tt[k] >= 10 * 3600:
                    w2 = np.where((tt >= 9 * 3600 + 30 * 60) & (tt < 10 * 3600))[0]
                    if len(w2) and w2[-1] < k:
                        mv2 = cc[w2[-1]] - cc[w2[0]]
                        if mv2 != 0:
                            pm_done = True
                            side2 = -1 if mv2 >= 0 else 1
                            pos, entry, state = side2 * pm_fade["sz"], cc[k], 6
                elif pos != 0 and state == 6:
                    pnl = (cc[k] - entry) * np.sign(pos)
                    if pnl <= -pm_fade["stop"] or pnl >= pm_fade["tgt"] or tt[k] >= pm_fade.get("exit", 14 * 3600):
                        pos, state, pm_done = 0, 4, True
                out[idx[k]] = pos
        return out
    return signals

if __name__ == "__main__":
    score_signals(make(), "BASE (expect 0.6856)")
    for hh, mm in ((13, 30), (14, 0), (14, 30), (15, 0), (15, 5)):
        score_signals(make(exit_t=hh * 3600 + mm * 60), f"exit {hh}:{mm:02d}")
    for add in (-4, -3, -2):
        for st, tg in ((30, 60), (25, 50), (40, 80)):
            score_signals(make(cont=dict(sz_add=add, stop=float(st), tgt=float(tg))),
                          f"cont after win sz{add:+d} {st}/{tg}")
    for gsz in (3, 4, 5):
        for st, tg in ((30, 60), (20, 40)):
            score_signals(make(pm_fade=dict(sz=gsz, stop=float(st), tgt=float(tg))),
                          f"pm fade 10:00 sz{gsz} {st}/{tg}")
