# ablate HEAD robustness knobs; find which cost speed vs buys fails=0; then re-add selectively
import sys, os, numpy as np, pandas as pd, importlib
sys.path.insert(0, 'eval'); sys.path.insert(0, '.')
import sim
BASE = open('strategy.py').read()

def build(rg='58', dd='on', ddw=20, ddth=-250, ft='70.0', cdm=1):
    s = BASE
    if rg == 'off':
        s = s.replace("        if hi[idx][where].max() - lo[idx][where].min() < RNG_MIN:\n            sz = sz // 2  # quiet open: weak exhaustion signal, half size\n", "")
    elif rg != '58':
        s = s.replace("RNG_MIN = 58.0", f"RNG_MIN = {rg}")
    if dd == 'off':
        s = s.replace("""        eq = np.cumsum([0.0] + hist)
        if eq[-1] - eq[-DD_W:].max() < DD_TH:
            sz = int(round(sz * 0.6))  # in drawdown vs recent peak: 60% size to survive trailing MLL
""", "")
    elif dd != 'on':
        s = s.replace(f"DD_W, DD_TH = {DD_W_D}, {DD_TH_D}", f"DD_W, DD_TH = {ddw}, {ddth}")
    if ft != '70.0':
        s = s.replace("F_STOP_PTS, F_TGT_PTS = 45.0, 70.0", f"F_STOP_PTS, F_TGT_PTS = 45.0, {ft}")
    if cdm != 1:
        s = s.replace("sz = int(round(sz * 0.6))", f"sz = int(round(sz * {cdm}))")
    open('_sw.py', 'w').write(s)
    if '_sw' in sys.modules: del sys.modules['_sw']
    return importlib.import_module('_sw')

DD_W_D, DD_TH_D = "20", "-250.0"
df = pd.read_parquet(os.path.expanduser('~/hive-data/nq/train.parquet'))
sess = pd.factorize(df.session, sort=True)[0]
WINDOW, START, TARGET, MLL = 30, 50000.0, 3000.0, 2000.0

def allrate(dp, dm):
    n = 0
    for st in range(0, len(dp) - WINDOW + 1):
        bal, peak_eod, best, days, ok = START, START, 0.0, 0, False
        for s in range(st, st + WINDOW):
            if bal + dm[s] <= min(peak_eod - MLL, START): break
            bal += dp[s]; days += 1
            best, peak_eod = max(best, dp[s]), max(peak_eod, bal)
            p = bal - START
            if p >= TARGET and days >= 2 and best < 0.5 * p: ok = True; break
        if not ok: n += 1
    return n

def run(tag, **kw):
    m = build(**kw)
    tg = np.asarray(m.signals(df)).astype(np.int64)
    dp, dm, _ = sim.run_days(sess, df.ct_sec.values, df.open.values, df.high.values, df.low.values, df.close.values, tg, sess.max() + 1)
    sc, rates = sim.score(dp, dm)[0], sim.score(dp, dm)[1]
    print(f"{tag}: base={sc:.4f} r={rates} fails={allrate(dp, dm)} dll={(dp<=-990).sum()} pnl={dp.sum():.0f}", flush=True)

run("HEAD")
run("noRG", rg='off')
run("noDD", dd='off')
run("FT68", ft='68.0')
run("noRG-noDD", rg='off', dd='off')
run("noRG+DD", rg='off', dd='on')
run("RG66", rg='66')
run("DDth-400", dd='x', ddw=20, ddth=-400)
run("DDth-150", dd='x', ddw=20, ddth=-150)
run("DDW40", dd='x', ddw=40, ddth=-250)
