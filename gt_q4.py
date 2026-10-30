"""Why do q4 attempts fail? Anatomy on sessions 160-215 with base 30/50 params."""
import gt_v1 as g, numpy as np, collections
from eval.sim import run_days, score, attempts, WINDOW

def runparams(**kw):
    d = dict(f_stop=30.0, f_tgt=50.0)
    d.update(kw)
    tgt_arr = g.build(g.T, g.C, g.SESS, g.bounds, g.ONV, g.MV, g.IMB,
                     8*3600+35*60, 14*3600+30*60, 7, 8, 6, 0, 50.0, 225.0, 0.05, 2,
                     40.0, 80.0, 200.0, d.get("big_stop", 30.0), d.get("big_tgt", 60.0),
                     d["f_stop"], d["f_tgt"], 1,
                     0.0, 0.0, True, True, False, 0, 0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, False, 0, 0)
    return run_days(g.SESS, g.T, g.O, g.H, g.L, g.C, tgt_arr, g.NS)

dp, dm, tr = runparams()
lo, hi = 3 * g.NS // 4, g.NS
att = attempts(dp, dm, lo, hi)
print(f"q4 attempts={len(att)} passrate={np.mean([p for p,_ in att]):.3f}")
cnt = collections.Counter(); pnl_days = []
for st_i, (p, d_) in enumerate(att):
    st = lo + st_i
    bal, peak_eod, best, days, busted = 50000.0, 50000.0, 0.0, 0, -1
    for s in range(st, st + WINDOW):
        thr = min(peak_eod - 2000.0, 50000.0)
        if bal + dm[s] <= thr:
            busted = s; break
        bal += dp[s]; days += 1
        best, peak_eod = max(best, dp[s]), max(peak_eod, bal)
        profit = bal - 50000.0
        if profit >= 3000.0 and days >= 2 and best < 0.5*profit:
            break
        if profit >= 3000.0 and days >= 2 and best >= 0.5*profit:
            print(f"  attempt st={st}: consistency-fail profit={profit:.0f} best={best:.0f}")
            break
    if busted >= 0:
        o_, m_ = g.ONV[busted], g.MV[busted]
        hot = not (50 <= abs(o_) <= 225)
        agree = np.sign(m_) == np.sign(o_) if o_ != 0 else True
        big = abs(o_) > 200
        b = ("HOT" if hot else "COLD") + ("_A" if agree else "_D") + ("_BIG" if big else "")
        kind = "DLL" if dm[busted] < -900 else ("close-pos" if dp[busted] > 0 else "loss")
        cnt[(b, kind, int(dp[busted]//200)*200)] += 1
print("bust days:")
for k_, v in sorted(cnt.items()): print("  ", k_, v)
print("\nq4 day pnl stats by bucket:")
lo4 = 3 * g.NS // 4
byk = collections.defaultdict(list)
for s in range(lo4, g.NS):
    o_, m_ = g.ONV[s], g.MV[s]
    hot = not (50 <= abs(o_) <= 225)
    agree = np.sign(m_) == np.sign(o_) if o_ != 0 else True
    big = abs(o_) > 200
    b = ("HOT" if hot else "COLD") + ("_A" if agree else "_D") + ("_BIG" if big else "")
    byk[b].append(dp[s])
for b, v in sorted(byk.items()):
    print(f"  {b:12s} n={len(v):3d} pnl={sum(v):7.0f} mean={np.mean(v):7.1f} win={np.mean([x>0 for x in v]):.2f}")
