import gt_v1 as g, numpy as np, collections
from eval.sim import run_days, score, attempts, START, WINDOW

tgt = g.build(g.T, g.C, g.SESS, g.bounds, g.ONV, g.MV, g.IMB,
              8*3600+35*60, 14*3600+30*60, 7, 8, 6, 0, 50.0, 225.0, 0.05, 2,
              40.0, 80.0, 200.0, 30.0, 60.0, 30.0, 50.0, 1,
              0.0, 0.0, True, True, False, 0, 0, 0.0, 0.0, 0.0, 0.0)
dp, dm, tr = run_days(g.SESS, g.T, g.O, g.H, g.L, g.C, tgt, g.NS)
sc, rates, passed = score(dp, dm)
print("score", sc, rates)
mid = g.NS // 2
for lo, hi, name in ((0, mid, "H1"), (mid, g.NS, "H2")):
    att = attempts(dp, dm, lo, hi)
    npass = sum(p for p, d in att)
    fails = [i for i, (p, d) in enumerate(att) if not p]
    print(f"{name}: attempts={len(att)} pass={npass} rate={npass/len(att):.3f}")
    bust_days, notreach = [], 0
    for st_i, (p, d) in enumerate(att):
        if p: continue
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
        if busted >= 0: bust_days.append(busted)
        else: notreach += 1
    print(f"  bust-failures: {len(bust_days)}; not-reached: {notreach}")
    cnt = collections.Counter()
    for s in bust_days:
        o_, m_ = g.ONV[s], g.MV[s]
        hot = not (50 <= abs(o_) <= 225)
        agree = np.sign(m_) == np.sign(o_) if o_ != 0 else True
        big = abs(o_) > 200
        b = ("HOT" if hot else "COLD") + ("_A" if agree else "_D") + ("_BIG" if big else "")
        cnt[(b, "DLL" if dm[s] < -900 else f"pnl{int(dp[s]//100*100)}")] += 1
    for k_, v in sorted(cnt.items()): print("   bustday", k_, v)
    # how far did not-reached attempts get?
    if notreach:
        print("  (not-reached attempts exist)")
