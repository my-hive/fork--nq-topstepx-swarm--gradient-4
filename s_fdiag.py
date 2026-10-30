"""Diagnose which windows fail when shifted; compare base vs tweaks."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, 'eval'); sys.path.insert(0, '.')
import sim, harness, types

def fails(mod_name='strategy'):
    mod = types.ModuleType('m'); exec(compile(open(mod_name + '.py').read(), 's', 'exec'), mod.__dict__)
    df = harness.DF; sess = harness.SESS
    tg = mod.signals(df).astype(np.int64)
    dp, dm, _ = sim.run_days(sess, df.ct_sec.values, df.open.values, df.high.values,
                              df.low.values, df.close.values, tg, sess.max() + 1)
    sess_ids = pd.factorize(df.session, sort=True)[1]
    out = {}
    for k in (0, 40, 50, 60):
        d, m = dp[k:], dm[k:]
        n = len(d); mid = n // 2
        f = []
        for half, lo, hi in (("h1", 0, mid), ("h2", mid, n)):
            for i in range(lo, hi - 30 + 1):
                if not sim.attempts(d, m, i, i + 30)[0][0]:
                    f.append((half, str(sess_ids[k + i].date())))
        out[k] = f
        print(f"off {k}: {len(f)} fails -> {f}")
    return dp, dm, out

if __name__ == "__main__":
    fails()
