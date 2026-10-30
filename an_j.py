import sys; sys.argv=['x']
import numpy as np, pandas as pd
sys.path.insert(0,'eval')
import sim, harness, types
mod=types.ModuleType('m'); exec(compile(open('strategy.py').read(),'s','exec'),mod.__dict__)
DF=harness.DF; SESS=harness.SESS
day_pnl, day_min, _ = sim.run_days(SESS, DF.ct_sec.values, DF.open.values, DF.high.values, DF.low.values, DF.close.values, mod.signals(DF).astype(np.int64), SESS.max()+1)
sess_ids=pd.factorize(DF.session,sort=True)[1]
for s in ('2026-01-09','2026-01-12','2026-01-13','2026-01-14','2026-01-15','2026-01-16','2026-01-20','2026-01-21','2026-01-22','2026-01-23','2026-01-26','2026-01-27','2026-01-30'):
    i=np.where([str(x.date())==s for x in sess_ids])[0]
    if len(i): print(s, "pnl",round(day_pnl[i[0]]),"min",round(day_min[i[0]]))
# Jan 12-30 window cumulative
i0=np.where([str(x.date())=='2026-01-12' for x in sess_ids])[0][0]
print("Jan12+30 cum:", np.round(np.cumsum(day_pnl[i0:i0+30])).astype(int).tolist())
