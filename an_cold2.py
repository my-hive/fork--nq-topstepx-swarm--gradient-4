"""Cold-agree paper fade across brackets."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "eval")
from harness import DF, SESS

START, EXIT = 8 * 3600 + 35 * 60, 14 * 3600 + 30 * 60
dates = pd.to_datetime(DF.session.unique())
NS = len(dates)
t, c = DF.ct_sec.values, DF.close.values
srt = np.where(np.r_[True, SESS[1:] != SESS[:-1]])[0]
end = np.r_[srt[1:] - 1, len(SESS) - 1]

sessions = []
for s in range(NS):
    i0, i1 = srt[s], end[s]
    tt, cc = t[i0:i1 + 1], c[i0:i1 + 1]
    w = np.where((tt >= 8 * 3600 + 30 * 60) & (tt < START))[0]
    if len(w) == 0:
        continue
    onv = cc[w[0]] - cc[0]
    mv = cc[w[-1]] - cc[w[0]]
    if mv == 0:
        continue
    hot = not (50 <= abs(onv) <= 200)
    agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
    md = dates[s].month * 100 + dates[s].day
    if md >= 1210 or md <= 102:
        continue
    if not hot and agree:
        sessions.append((s, tt, cc, -1 if mv >= 0 else 1))

def paper(st, tg, trail=0.0):
    tot = h1 = h2 = n1 = n2 = 0.0
    ns1 = ns2 = 0
    for s, tt, cc, side in sessions:
        pos, entry, state, best = 0, 0.0, 0, 0.0
        pnl = None
        for k in range(len(tt)):
            if tt[k] < START or tt[k] >= EXIT:
                pos = 0
            elif state == 0:
                pos, entry, state = side, cc[k], 1
            elif pos != 0 and state == 1:
                p = (cc[k] - entry) * side
                best = max(best, p)
                stop = max(st, best - trail) if trail > 0 else st
                if p <= -stop:
                    pnl, state, pos = -stop, 4, 0
                elif p >= tg:
                    pnl, state, pos = tg, 4, 0
        if pnl is None:
            if state == 1:
                pnl = (cc[-1] - entry) * side
            else:
                continue
        tot += pnl
        if s > NS / 2:
            h2 += pnl; n2 += 1
        else:
            h1 += pnl; n1 += 1
    return tot, h1, h2, n1, n2

for st in (20, 30, 40):
    for tg in (20, 30, 40, 60, 80):
        tot, h1, h2, n1, n2 = paper(st, tg)
        print(f"stop{st:>3} tgt{tg:>3}: tot={tot:7.0f} h1={h1:6.0f} h2={h2:6.0f} (n={n1}/{n2})")
for tr in (15, 25, 40):
    tot, h1, h2, n1, n2 = paper(40, 80, trail=tr)
    print(f"stop40 tgt80 trail{tr:>3}: tot={tot:7.0f} h1={h1:6.0f} h2={h2:6.0f} (n={n1}/{n2})")
