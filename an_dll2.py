import sys
sys.argv = ['x']
import numpy as np, pandas as pd
sys.path.insert(0, 'eval')
import sim, harness, types

mod = types.ModuleType('m'); exec(compile(open('strategy.py').read(), 's', 'exec'), mod.__dict__)
df = harness.DF
day_pnl, day_min, _ = sim.run_days(harness.SESS, df.ct_sec.values, df.open.values, df.high.values,
    df.low.values, df.close.values, mod.signals(df).astype(np.int64), harness.SESS.max() + 1)
sess_ids = pd.factorize(df.session, sort=True)[1]
c = df.close.values; tt = df.ct_sec.values
for _, idx in df.groupby("session", sort=False).indices.items():
    d = sess_ids[idx[0]]
    if day_pnl[d] < -900:
        cc = c[idx]; t2 = tt[idx]
        m = (t2 >= 8 * 3600 + 30 * 60) & (t2 < 8 * 3600 + 35 * 60)
        w = np.where(m)[0]
        onv = cc[w[0]] - cc[0]; mv = cc[w[-1]] - cc[w[0]]
        print(f"{d.date()} dow={d.dayofweek} pnl={day_pnl[d]:.0f} onv={onv:.0f} mv={mv:.0f} "
              f"hot={not (50 <= abs(onv) <= 200)} agree={np.sign(mv) == np.sign(onv)}")
