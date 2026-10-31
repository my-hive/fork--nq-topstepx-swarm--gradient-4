import numpy as np, pandas as pd, sys, os, importlib
sys.path.insert(0, "."); sys.path.insert(0, "eval")
import strategy, sim
df = pd.read_parquet(os.path.expanduser("~/hive-data/nq/train.parquet"))
SESS = pd.factorize(df.session, sort=True)[0]
O, H, L, C, T = df.open.values, df.high.values, df.low.values, df.close.values, df.ct_sec.values
def score_now(tag):
    target = np.nan_to_num(np.asarray(strategy.signals(df), dtype=float)).astype(np.int64)
    dp, dmin, tr = sim.run_days(SESS, T, O, H, L, C, target, SESS.max()+1)
    sc, rates, passed = sim.score(dp, dmin)
    print(f"{tag:30s} score {sc:.9f} rates {rates[0]:.3f}/{rates[1]:.3f} days {np.mean(passed):.4f} pnl {dp.sum():.0f}", flush=True)
importlib.reload(strategy); score_now("base")
for k, th in [(1,-400.0),(1,-500.0),(1,-600.0),(3,-800.0),(2,-1000.0)]:
    importlib.reload(strategy)
    strategy.LOSS_K, strategy.LOSS_TH = k, th
    score_now(f"LOSS_K={k} TH={th}")
# divisor edits (inline)
base_src = open("strategy.py").read()
for d in (4, 5, 7):
    src = base_src.replace("sz = int(round(sz / 6))", f"sz = int(round(sz / {d}))")
    assert src != base_src
    open("strategy.py","w").write(src)
    importlib.reload(strategy); score_now(f"cooldown divisor {d}")
open("strategy.py","w").write(base_src)
importlib.reload(strategy); score_now("restored")
