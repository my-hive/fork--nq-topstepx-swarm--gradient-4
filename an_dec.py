import sys; sys.argv=['x']
import numpy as np, pandas as pd
sys.path.insert(0,'eval')
import sim, harness, types
mod=types.ModuleType('m'); exec(compile(open('strategy.py').read(),'s','exec'),mod.__dict__)
DF=harness.DF; SESS=harness.SESS
sig=mod.signals(DF)
day_pnl, day_min, trades = sim.run_days(SESS, DF.ct_sec.values, DF.open.values, DF.high.values, DF.low.values, DF.close.values, sig, SESS.max()+1)
sess_ids=pd.factorize(DF.session,sort=True)[1]
# daily PnL in Dec window and May window
for d,x in zip(sess_ids, day_pnl):
    if str(d.date()) in ('2025-12-08','2025-12-09','2025-12-10','2025-12-11','2025-12-12','2025-12-15','2025-12-16','2025-12-17','2026-01-30','2026-05-06','2026-05-12','2026-05-13','2026-05-18','2026-05-07'):
        print(str(d.date()), round(x), "min", round(day_min[d is not None and 0 or 0]) if False else "")
