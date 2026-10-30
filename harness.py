"""Local fast harness: reuses eval/sim.py to score strategy variants without lookahead checks."""
import importlib, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, "eval")
import sim

DF = pd.read_parquet("/home/shawn/hive-data/nq/train.parquet")
SESS = pd.factorize(DF.session, sort=True)[0]

def score_signals(fn, name=""):
    t0 = time.time()
    target = np.nan_to_num(np.asarray(fn(DF), dtype=float)).astype(np.int64)
    day_pnl, day_min, trades = sim.run_days(SESS, DF.ct_sec.values, DF.open.values, DF.high.values,
                                            DF.low.values, DF.close.values, target, SESS.max() + 1)
    sc, rates, passed = sim.score(day_pnl, day_min)
    print(f"{name:44s} score={sc:.4f} h1={rates[0]:.3f} h2={rates[1]:.3f} "
          f"days={np.mean(passed) if passed else float('nan'):.1f} pnl={day_pnl.sum():.0f} "
          f"win={(day_pnl>0).mean():.3f} t={time.time()-t0:.0f}s")
    return sc, rates, day_pnl
