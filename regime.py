"""Regime filters to cut MLL busts: prior-session momentum/EMA/ATR gates."""
import numpy as np
import sweep
from sweep import bounds, t, o, h, l, c, sess, NS, START
from eval.sim import run_days, score
from round2b import recompute, sizes_for
from failmode import build

b, sd = recompute(50, 175)
SZ0 = sizes_for(b)

# daily OHLC by session (close at 14:30 as "reference close", and session last close)
day_close = np.zeros(NS); day_open830 = np.zeros(NS); day_hi = np.zeros(NS); day_lo = np.zeros(NS)
for s in range(NS):
    i0, i1 = bounds[s], bounds[s + 1]
    tt, cc = t[i0:i1], c[i0:i1]
    day_close[s] = cc[-1]
    m = tt < 8 * 3600 + 30 * 60
    day_open830[s] = cc[np.where(m)[0][0]] if m.any() else cc[0]
    day_hi[s], day_lo[s] = np.nanmax(sweep.h[i0:i1]), np.nanmin(sweep.l[i0:i1])

def ema(x, n):
    a = 2 / (n + 1)
    e = np.zeros(len(x)); e[0] = x[0]
    for i in range(1, len(x)):
        e[i] = a * x[i] + (1 - a) * e[i - 1]
    return e

ema10 = ema(day_close, 10)
ema5 = ema(day_close, 5)
ret1 = np.zeros(NS); ret5 = np.zeros(NS); ret10 = np.zeros(NS)
for s in range(1, NS):
    ret1[s] = day_close[s - 1] - day_close[s - 2] if s >= 2 else 0
    ret5[s] = day_close[s - 1] - day_close[max(0, s - 6)]
    ret10[s] = day_close[s - 1] - day_close[max(0, s - 11)]
dist_ema = day_close - ema10  # yesterday's close vs its EMA (shift handled: use s-1 values)

def run_with(sz):
    tgt = build(t, c, bounds, sd, sz, START, 14 * 3600 + 30 * 60, 40.0, 80.0)
    dp, dm, _ = run_days(sess, t, o, h, l, c, tgt, NS)
    sc, rates, passed = score(dp, dm)
    return sc, rates[0], rates[1], (np.mean(passed) if passed else np.nan), dp.sum(), dp.min()

def show(tag, r):
    print(f"{tag:52s} score={r[0]:.4f} h1={r[1]:.3f} h2={r[2]:.3f} days={r[3]:.1f} pnl={r[4]:.0f} worst={r[5]:.0f}", flush=True)

def mk_gate(cond_fn, shrink=0):
    SZ = SZ0.copy()
    n = 0
    for s in range(NS):
        if SZ[s] > 0 and cond_fn(s):
            SZ[s] = max(0, SZ[s] - shrink) if shrink else 0
            n += 1
    return SZ, n

if __name__ == "__main__":
    show("BASE", run_with(SZ0))
    print("== prior 5d momentum: skip when trending ==")
    for th in (150, 250, 350, 450):
        SZ, n = mk_gate(lambda s, th=th: abs(ret5[s]) > th)
        show(f"|ret5|>{th} skip (n={n})", run_with(SZ))
    print("== prior 5d momentum: skip only when trend AGREES with overnight ==")
    for th in (200, 300):
        SZ, n = mk_gate(lambda s, th=th: abs(ret5[s]) > th)  # placeholder replaced below
    # directional: skip when 5d trend direction == fade direction is AGAINST trend (fade with trend only)
    print("== fade-with-trend only: skip when fade fights 5d trend ==")
    import gates
    from gates import feat
    for th in (100, 200, 300):
        SZ = SZ0.copy(); n = 0
        for s, f in feat.items():
            if f is None or SZ[s] == 0:
                continue
            fade_dir = -np.sign(f["mv"])
            trend = np.sign(ret5[s])
            if trend != 0 and fade_dir * trend < 0 and abs(ret5[s]) > th:
                SZ[s] = 0; n += 1
        show(f"skip counter-trend fades |ret5|>{th} (n={n})", run_with(SZ))
    print("== dist from EMA10: skip when far ==")
    for th in (100, 150, 200):
        SZ, n = mk_gate(lambda s, th=th: abs(dist_ema[s]) > th)
        show(f"|close-ema10|>{th} skip (n={n})", run_with(SZ))
