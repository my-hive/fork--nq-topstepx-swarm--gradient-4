# stress-min sweep: FT (flip target) x DD on 24ff1d60 CD8 base
import sys, os, numpy as np, pandas as pd, importlib
sys.path.insert(0, 'eval'); sys.path.insert(0, '.')
import sim

def build(ft, dd):
    src = open('s_24ff1d60.py').read()
    src = src.replace("F_STOP_PTS, F_TGT_PTS = 45.0, 68.0", f"F_STOP_PTS, F_TGT_PTS = 45.0, {ft}")
    if dd == 'off':
        src = src.replace("""        eq = np.cumsum([0.0] + hist)
        if eq[-1] - eq[-DD_W:].max() < DD_TH:
            sz = int(round(sz * 0.6))  # in drawdown vs recent peak: 60% size to survive trailing MLL
""", "")
    elif dd != (20, -250):
        src = src.replace("DD_W, DD_TH = 20, -250.0", f"DD_W, DD_TH = {dd[0]}, {dd[1]}.0")
    open('_sw.py', 'w').write(src)
    m = importlib.import_module('_sw'); importlib.reload(m)
    return m

df = pd.read_parquet(os.path.expanduser('~/hive-data/nq/train.parquet'))
sess = pd.factorize(df.session, sort=True)[0]
C = dict(ct_sec=df.ct_sec.values, o=df.open.values, h=df.high.values, l=df.low.values, c=df.close.values)

def run(ft, dd):
    m = build(ft, dd)
    tg = np.asarray(m.signals(df)).astype(np.int64)
    dp, dm, _ = sim.run_days(sess, C['ct_sec'], C['o'], C['h'], C['l'], C['c'], tg, sess.max() + 1)
    base = round(sim.score(dp, dm)[0], 4)
    offs = [round(sim.score(dp[k:], dm[k:])[0], 4) for k in (10, 20, 30, 40, 50, 60)]
    tails = [round(sim.score(dp[:-k], dm[:-k])[0], 4) for k in (20, 40, 60)]
    mn = round(min(offs + tails), 4)
    print(f"FT={ft} DD={dd} base={base} stress-min={mn} offs={offs} tails={tails}", flush=True)

for ft in (64, 66, 68, 70, 72):
    for dd in ((20, -250), 'off'):
        run(ft, dd)
