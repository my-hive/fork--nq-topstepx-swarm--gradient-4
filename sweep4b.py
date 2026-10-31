"""Constant re-audit on 7da78be base."""
import numpy as np, pandas as pd, sys, os, importlib
sys.path.insert(0, "."); sys.path.insert(0, "eval")
import strategy, sim
df = pd.read_parquet(os.path.expanduser("~/hive-data/nq/train.parquet"))
SESS = pd.factorize(df.session, sort=True)[0]
O, H, L, C, T = df.open.values, df.high.values, df.low.values, df.close.values, df.ct_sec.values
def run(tag, src=None, **over):
    importlib.reload(strategy)
    if src: open("strategy.py","w").write(src)
    for k, v in over.items(): setattr(strategy, k, v)
    target = np.nan_to_num(np.asarray(strategy.signals(df), dtype=float)).astype(np.int64)
    dp, dmin, tr = sim.run_days(SESS, T, O, H, L, C, target, SESS.max()+1)
    sc, rates, passed = sim.score(dp, dmin)
    print(f"{tag:26s} score {sc:.9f} rates {rates[0]:.3f}/{rates[1]:.3f} days {np.mean(passed):.4f} pnl {dp.sum():.0f}", flush=True)
base_src = open("strategy.py").read()
run("base7da78be")
run("F_TGT=66", F_TGT_PTS=66.0)
run("F_TGT=70", F_TGT_PTS=70.0)
run("F_STOP=55", F_STOP_PTS=55.0)
run("LOSS div via TH-900", LOSS_TH=-900.0)
run("LOSS_TH=-750", LOSS_TH=-750.0)
# strong threshold edits (inline constant)
for v in (1700, 1900, 2000):
    src = base_src.replace("strong = sum(hist[-3:]) > 1800", f"strong = sum(hist[-3:]) > {v}")
    assert src != base_src
    run(f"strong={v}", src)
    open("strategy.py","w").write(base_src)
# flip strong tgt inline
for v in (72.0, 76.0):
    src = base_src.replace("flip_tgt = 74.0 if strong else F_TGT_PTS", f"flip_tgt = {v} if strong else F_TGT_PTS")
    assert src != base_src
    run(f"flip_strong_tgt={v}", src)
    open("strategy.py","w").write(base_src)
