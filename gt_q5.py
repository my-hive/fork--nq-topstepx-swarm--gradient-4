import gt_v1 as g, numpy as np
from eval.sim import attempts

def qcheck(name, **kw):
    sc, rates, dp, dm = g.ev(name, **kw)
    qs = []
    for q in range(4):
        lo = q*g.NS//4; hi = min((q+1)*g.NS//4, g.NS)
        att = attempts(dp, dm, lo, hi)
        qs.append(np.mean([p for p,_ in att]))
    print(f"   q={qs[0]:.2f} {qs[1]:.2f} {qs[2]:.2f} {qs[3]:.2f}")

qcheck("base 0.8405")
qcheck("thu_scope2 0.8531", thu_scope=2)
