"""Bust anatomy on 0.8539 base (current gt_v1 defaults)."""
import gt_v1 as g, numpy as np, collections
from eval.sim import run_days, score, attempts, WINDOW

tgt = g.build(g.T, g.C, g.SESS, g.bounds, g.ONV, g.MV, g.IMB, g.VOL,
              8*3600+35*60, 14*3600+30*60, 7, 9, 6, 0, 50.0, 200.0, 0.03, 2,
              40.0, 80.0, 200.0, 30.0, 60.0, 40.0, 60.0, 1,
              0.0, 0.0, True, True, False, 0, 0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, True, 0, 0, 0, 12000.0,
              g.PUSH_EXT, g.LH_MV, -1e9, -1e9, 0, g.PREV_PNL, 0, g.DOW, True, True,
              0, 0, 200.0, np.zeros(20, dtype=np.int64))
dp, dm, tr = run_days(g.SESS, g.T, g.O, g.H, g.L, g.C, tgt, g.NS)
sc, rates, passed = score(dp, dm)
print(f"score={sc:.4f} h1={rates[0]:.3f} h2={rates[1]:.3f} days={np.mean(passed):.1f} pnl={dp.sum():.0f} win={(dp>0).mean():.3f}")
mid = g.NS // 2
for lo, hi, name in ((0, mid, "H1"), (mid, g.NS, "H2")):
    att = attempts(dp, dm, lo, hi)
    fails = [i for i, (p, d) in enumerate(att) if not p]
    print(f"\n{name}: attempts={len(att)} pass={np.mean([p for p,_ in att]):.3f} fails={len(fails)}")
    cnt = collections.Counter(); kinds = collections.Counter()
    for st_i in fails:
        st = lo + st_i
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
            hot = not (50 <= abs(o_) <= 200)
            agree = np.sign(m_) == np.sign(o_) if o_ != 0 else True
            big = abs(o_) > 200
            b = ("HOT" if hot else "COLD") + ("_A" if agree else "_D") + ("_BIG" if big else "")
            kinds[f"{'DLL' if dm[busted] < -900 else ('closepos' if dp[busted] > 0 else 'loss')}"] += 1
            cnt[b] += 1
    print("  bust kinds:", dict(kinds))
    print("  bust buckets:", dict(cnt))
    # failure kind: bust vs not-reached
    print("  (busted shown above; rest = target not reached in 30)")
