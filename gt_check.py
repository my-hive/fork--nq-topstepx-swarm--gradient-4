"""Run gu-3's actual 7107298 strategy.py through the local harness for exact score."""
import sys
import numpy as np
sys.path.insert(0, ".")
import gt_v1 as g
import importlib.util
spec = importlib.util.spec_from_file_location("s7107298", "/tmp/s710.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
tgt = m.signals(g.df)
from eval.sim import run_days, score
dp, dm, tr = run_days(g.SESS, g.T, g.O, g.H, g.L, g.C, np.asarray(tgt, dtype=np.int64), g.NS)
sc, rates, passed = score(dp, dm)
print(f"their ACTUAL: score={sc:.4f} h1={rates[0]:.3f} h2={rates[1]:.3f} days={np.mean(passed):.1f}")
