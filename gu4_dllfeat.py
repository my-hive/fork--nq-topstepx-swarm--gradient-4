"""gu-4: feature compare: 15 DLL-hit days vs winning days. Look for a real gate."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "eval")
import sim
from harness import DF, SESS
import strategy as S

target = np.nan_to_num(np.asarray(S.signals(DF), dtype=float)).astype(np.int64)
day_pnl, day_min, trades = sim.run_days(SESS, DF.ct_sec.values, DF.open.values, DF.high.values,
                                        DF.low.values, DF.close.values, target, SESS.max() + 1)

rows = []
for _, idx in DF.groupby("session", sort=False).indices.items():
    tt, cc, oo, hh, ll = DF.ct_sec.values[idx], DF.close.values[idx], DF.open.values[idx], DF.high.values[idx], DF.low.values[idx]
    bv, sv = DF.buy_vol.values[idx], DF.sell_vol.values[idx]
    m = (tt >= 8 * 3600 + 30 * 60) & (tt < 8 * 3600 + 35 * 60)
    if not m.any():
        continue
    w = np.where(m)[0]
    onv = cc[w[0]] - cc[0]
    mv = cc[w[-1]] - cc[w[0]]
    if mv == 0:
        continue
    hot = not (50.0 <= abs(onv) <= 200.0)
    agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
    imb = (bv[w].sum() - sv[w].sum()) / max(bv[w].sum() + sv[w].sum(), 1)
    # overnight range & vol; push extension beyond 8:35 (max adverse in first 15m)
    ext15 = (hh[w[-1]:w[-1] + 900].max() if mv < 0 else ll[w[-1]:w[-1] + 900].min()) - cc[w[-1]] if len(idx) > w[-1] + 900 else 0
    on_rng = hh[:w[0]].max() - ll[:w[0]].min() if w[0] > 0 else 0
    rows.append(dict(sess=DF.session.values[idx[0]], hot=hot, agree=agree, onv=onv, mv=mv,
                     imb_along=imb * np.sign(mv), on_rng=on_rng,
                     open_vol=bv[w].sum() + sv[w].sum(), ext15=-ext15 * np.sign(mv)))

d = pd.DataFrame(rows).set_index("sess")
d["pnl"] = day_pnl[:len(d)]
d["mind"] = day_min[:len(d)]
d["dll"] = d.mind <= -1000
d = d[d.pnl != 0]  # traded days only

dll = d[d.dll]
win = d[(d.pnl > 800)]
print(f"dll days n={len(dll)}, big-win days n={len(win)}")
for c in ["onv", "mv", "imb_along", "on_rng", "open_vol", "ext15"]:
    print(f"{c:10s} dll med={dll[c].median():8.1f}  win med={win[c].median():8.1f}  "
          f"dll q10={dll[c].quantile(.1):8.1f} q90={dll[c].quantile(.9):8.1f}")
print("\ndll days detail:")
print(dll[["hot", "agree", "onv", "mv", "imb_along", "on_rng", "ext15", "pnl"]].round(1).to_string())
