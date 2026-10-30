"""Cold+dis days: full feature dump with current 0.8904 params."""
import gt_v1 as g, numpy as np
from eval.sim import run_days, score

sc, rates, dp, dm = g.ev("BASE")
rows = []
for s in range(g.NS):
    o_, m_ = g.ONV[s], g.MV[s]
    hot = not (50 <= abs(o_) <= 200)
    agree = np.sign(m_) == np.sign(o_) if o_ != 0 else True
    if hot or agree or g.VOL[s] < 12000 or m_ == 0:
        continue
    rows.append((s, s*4//g.NS, abs(o_), m_, g.IMB[s]*np.sign(m_), g.VOL[s], g.RETR[s], dp[s], dm[s]))
print(f"cold+dis days: {len(rows)}  pnl={sum(r[7] for r in rows):.0f}")
print(f"{'s':>4} {'q':>2} {'|onv|':>5} {'mv':>6} {'imb':>6} {'vol':>6} {'retr':>5} {'pnl':>6} {'daymin':>7}")
for r in sorted(rows, key=lambda r: r[7]):
    print(f"{r[0]:4d} {r[1]:2d} {r[2]:5.0f} {r[3]:+6.1f} {r[4]:+.2f} {r[5]:6.0f} {r[6]:5.1f} {r[7]:6.0f} {r[8]:7.0f}")
# outcome by |mv|
for lo, hi in ((0,30),(30,50),(50,70),(70,200)):
    sub = [r for r in rows if lo <= abs(r[3]) < hi]
    if sub:
        print(f"|mv| [{lo},{hi}): n={len(sub)} pnl={sum(r[7] for r in sub):6.0f} "
              f"dll={sum(1 for r in sub if r[8] < -900)} win={np.mean([r[7]>0 for r in sub]):.2f}")
for lo, hi in ((0,100),(100,140),(140,200)):
    sub = [r for r in rows if lo <= abs(r[2]) < hi]
    if sub:
        print(f"|onv| [{lo},{hi}): n={len(sub)} pnl={sum(r[7] for r in sub):6.0f} "
              f"dll={sum(1 for r in sub if r[8] < -900)} win={np.mean([r[7]>0 for r in sub]):.2f}")
