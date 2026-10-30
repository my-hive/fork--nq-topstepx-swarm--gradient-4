"""H1 bust anatomy on 0.8904 config via g.ev()."""
import gt_v1 as g, numpy as np
from eval.sim import attempts, WINDOW

sc, rates, dp, dm = g.ev("BASE 0.8904")
mid = g.NS // 2
att = attempts(dp, dm, 0, mid)
fails = [i for i, (p, d) in enumerate(att) if not p]
print(f"H1 fails: {len(fails)}")
for st_i in fails:
    st = st_i
    bal, peak_eod, best, days, busted = 50000.0, 50000.0, 0.0, 0, -1
    for s in range(st, st + WINDOW):
        thr = min(peak_eod - 2000.0, 50000.0)
        if bal + dm[s] <= thr: busted = s; break
        bal += dp[s]; days += 1
        best, peak_eod = max(best, dp[s]), max(peak_eod, bal)
        profit = bal - 50000.0
        if profit >= 3000.0 and days >= 2 and best < 0.5*profit: break
    if busted < 0:
        print(f"  st={st} not-reached"); continue
    o_, m_ = g.ONV[busted], g.MV[busted]
    hot = not (50 <= abs(o_) <= 200)
    agree = np.sign(m_) == np.sign(o_) if o_ != 0 else True
    b = ("HOT" if hot else "COLD") + ("_A" if agree else "_D") + ("_BIG" if abs(o_) > 200 else "")
    kind = "DLL" if dm[busted] < -900 else "closepos"
    slack = (bal + dm[busted]) - min(peak_eod - 2000.0, 50000.0)
    print(f"  st={st} busted@{busted} {b} {kind} daymin={dm[busted]:.0f} pnl={dp[busted]:.0f} over_thr_by={-slack:.0f} "
          f"|onv|={abs(o_):.0f} mv={m_:.1f} vol={g.VOL[busted]:.0f} retr={g.RETR[busted]:.1f}")
