"""Shift robustness: drop first k sessions, rescore (like gu-2's off check)."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, 'eval'); sys.path.insert(0, '.')
import sim, harness, types

mod = types.ModuleType('m'); exec(compile(open('strategy.py').read(), 's', 'exec'), mod.__dict__)
df = harness.DF; sess = harness.SESS
tg = mod.signals(df).astype(np.int64)
dp, dm, _ = sim.run_days(sess, df.ct_sec.values, df.open.values, df.high.values,
                          df.low.values, df.close.values, tg, sess.max() + 1)
scores = []
for k in (0, 5, 10, 20, 30, 40, 50, 60, 70):
    d, m = dp[k:], dm[k:]
    sc, rates, _ = sim.score(d, m)
    scores.append(sc)
    print(f"off {k:3d}: {sc:.4f} h1={rates[0]:.3f} h2={rates[1]:.3f}")
print(f"min={min(scores):.4f} mean={np.mean(scores):.4f}")
