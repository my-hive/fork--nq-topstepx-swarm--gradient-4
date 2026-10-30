"""Anatomy of DLL-hit days (-1000ish): features + which legs fired."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, 'eval'); sys.path.insert(0, '.')
import sim, harness, types

mod = types.ModuleType('m'); exec(compile(open('strategy.py').read(), 's', 'exec'), mod.__dict__)
df = harness.DF
day_pnl, day_min, _ = sim.run_days(harness.SESS, df.ct_sec.values, df.open.values, df.high.values,
                                    df.low.values, df.close.values, mod.signals(df).astype(np.int64),
                                    harness.SESS.max() + 1)
sess_ids = pd.factorize(df.session, sort=True)[1]
dows = pd.to_datetime(df.session.values).dayofweek

for _, idx in df.groupby("session", sort=False).indices.items():
    i = idx[0]
    d = sess_ids[np.searchsorted(harness.SESS, i)]
    if day_pnl[d] < -900:
        tt, cc = df.ct_sec.values[idx], df.close.values[idx]
        m = (tt >= 8 * 3600 + 30 * 60) & (tt < 8 * 3600 + 35 * 60)
        w = np.where(m)[0]
        onv = cc[w[0]] - cc[0]; mv = cc[w[-1]] - cc[w[0]]
        print(f"{sess_ids[d].date()} dow={dows[i]} pnl={day_pnl[d]:.0f} onv={onv:.0f} mv={mv:.0f} "
              f"{'hot' if not (50 <= abs(onv) <= 200) else 'cold'} {'agree' if np.sign(mv)==np.sign(onv) else 'dis'}")
