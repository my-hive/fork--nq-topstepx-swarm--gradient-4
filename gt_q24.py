"""q2 bust anatomy with current best params (hb d6a8, big 30/70, flip 30/56)."""
import gt_v1 as g, numpy as np, collections
from eval.sim import run_days, attempts, WINDOW, score

P = dict(sz_hotbig_dis=6, sz_hotbig_agree=8, big_stop=30.0, big_tgt=70.0, f_stop=30.0, f_tgt=56.0, vol_min=10000.0)
tgt_arr = g.build(g.T, g.C, g.SESS, g.bounds, g.ONV, g.MV, g.IMB, g.VOL,
                 8*3600+35*60, 14*3600+30*60, 7, 8, 6, 0, 50.0, 225.0, 0.05, 2,
                 40.0, 80.0, 200.0, P["big_stop"], P["big_tgt"], P["f_stop"], P["f_tgt"], 1,
                 0.0, 0.0, True, True, False, 0, 0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, False, P["sz_hotbig_dis"], P["sz_hotbig_agree"], 0, P["vol_min"])
dp, dm, tr = run_days(g.SESS, g.T, g.O, g.H, g.L, g.C, tgt_arr, g.NS)
sc, rates, passed = score(dp, dm)
print(f"score={sc:.4f} h1={rates[0]:.3f} h2={rates[1]:.3f} days={np.mean(passed):.1f}")

for qlo, name in ((1*g.NS//4, "q2"), (3*g.NS//4, "q4")):
    hi = min((qlo// (g.NS//4) + 1) * (g.NS//4), g.NS)
    att = attempts(dp, dm, qlo, hi)
    print(f"\n{name}: attempts={len(att)} pass={np.mean([p for p,_ in att]):.3f}")
    cnt = collections.Counter()
    for st_i, (p, d_) in enumerate(att):
        if p: continue
        st = qlo + st_i
        bal, peak_eod, best, days, busted = 50000.0, 50000.0, 0.0, 0, -1
        for s in range(st, st + WINDOW):
            thr = min(peak_eod - 2000.0, 50000.0)
            if bal + dm[s] <= thr: busted = s; break
            bal += dp[s]; days += 1
            best, peak_eod = max(best, dp[s]), max(peak_eod, bal)
            profit = bal - 50000.0
            if profit >= 3000.0 and days >= 2:
                if best >= 0.5*profit: print(f"  consistency-fail st={st} profit={profit:.0f} best={best:.0f}")
                break
        if busted >= 0:
            o_, m_ = g.ONV[busted], g.MV[busted]
            hot = not (50 <= abs(o_) <= 225)
            agree = np.sign(m_) == np.sign(o_) if o_ != 0 else True
            big = abs(o_) > 200
            b = ("HOT" if hot else "COLD") + ("_A" if agree else "_D") + ("_BIG" if big else "")
            kind = "DLL" if dm[busted] < -900 else ("closepos" if dp[busted] > 0 else "loss")
            cnt[(b, kind, int(dp[busted]//100)*100)] += 1
    for k_, v in sorted(cnt.items()): print("  bust", k_, v)
