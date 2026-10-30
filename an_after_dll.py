"""PnL of sessions following DLL days / big-win days under BASE CD8."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "eval")
import sim
from harness import DF, SESS
from exp_flip import signals

target = signals(DF)
day_pnl, day_min, trades = sim.run_days(SESS, DF.ct_sec.values, DF.open.values, DF.high.values,
                                        DF.low.values, DF.close.values, target, SESS.max() + 1)
dates = pd.to_datetime(DF.session.unique())
pnl = pd.Series(day_pnl, index=dates)
mind = pd.Series(day_min, index=dates)
dll = mind <= -1005  # DLL-ish days (incl near misses)
bigwin = pnl >= 1200

print("after DLL day (next session):")
nx = pnl.shift(-1)[dll]
print(nx.describe().round(0), "\n>0:", (nx > 0).sum(), "/", len(nx), "\n", nx.round(0).to_dict())
print("\nnext 2 sessions after DLL:")
nx2 = pd.DataFrame({"+1": pnl.shift(-1)[dll], "+2": pnl.shift(-2)[dll]})
print(nx2.round(0))
print("\nafter big win (>=1200) day:")
nw = pnl.shift(-1)[bigwin]
print(nw.describe().round(0), "\n>0:", (nw > 0).sum(), "/", len(nw))
print("\nDLL days dow:", [d.dayofweek for d in dates[dll.values]])
print("sessions after DLL:", [str(d)[:10] for d in dates[pnl.shift(-1)[dll].index]])
