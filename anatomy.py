"""Trade anatomy: resolution time, MFE/MAE, time-stop variants, risk-parity stops."""
import numpy as np
from numba import njit
import sweep
from sweep import bounds, t, o, h, l, c, sess, NS, START
from eval.sim import run_days, score
from round2b import recompute, sizes_for
from gates import feat

b, sd = recompute(50, 175)
SZ0 = sizes_for(b)

@njit(cache=False)
def build_ts(t, c, bounds, sides, sz_day, start, exit_, stop, tgt, tstop_sec):
    """tstop_sec: flat if trade unresolved after this many seconds past entry (0=off)."""
    out = np.zeros(len(c), dtype=np.int64)
    res = np.zeros((len(bounds) - 1, 3))  # minutes to resolve, pnl pts, resolved(1 stop/2 tgt/3 time/4 eod)
    for s in range(len(bounds) - 1):
        sz = sz_day[s]
        if sz == 0:
            continue
        side, pos, entry, done = sides[s], 0, 0.0, False
        t_entry = 0.0
        r_pnl, r_kind = 0.0, 0
        for k in range(bounds[s], bounds[s + 1]):
            if t[k] < start or t[k] >= exit_:
                if pos != 0:
                    r_pnl, r_kind = (c[k] - entry) * (1.0 if pos > 0 else -1.0), 4
                pos = 0
            elif pos == 0 and not done:
                pos, entry, done = side * sz, c[k], True
                t_entry = t[k]
            elif pos != 0:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if pnl <= -stop:
                    r_pnl, r_kind = pnl, 1; pos = 0
                elif pnl >= tgt:
                    r_pnl, r_kind = pnl, 2; pos = 0
                elif tstop_sec > 0 and t[k] - t_entry >= tstop_sec:
                    r_pnl, r_kind = pnl, 3; pos = 0
            out[k] = pos
        res[s, 0] = (t_entry % 86400 - 8 * 3600) / 60.0
        res[s, 1] = r_pnl
        res[s, 2] = r_kind
    return out, res

def ev(sz=SZ0, stop=40.0, tgt=80.0, tstop=0):
    tgt_arr, res = build_ts(t, c, bounds, sd, sz, START, 14*3600+30*60, stop, tgt, tstop)
    dp, dm, _ = run_days(sess, t, o, h, l, c, tgt_arr, NS)
    sc, rates, passed = score(dp, dm)
    return sc, rates[0], rates[1], (np.mean(passed) if passed else np.nan), dp.sum(), dp.min(), res

def show(tag, r):
    print(f"{tag:44s} score={r[0]:.4f} h1={r[1]:.3f} h2={r[2]:.3f} days={r[3]:.1f} pnl={r[4]:.0f} worst={r[5]:.0f}", flush=True)

if __name__ == "__main__":
    r = ev(); show("BASE", r)
    res = r[6]
    valid = res[res[:, 2] > 0]
    print(f"\nn trades={len(valid)}; kind: stop={np.sum(valid[:,2]==1)} tgt={np.sum(valid[:,2]==2)} time={np.sum(valid[:,2]==3)} eod={np.sum(valid[:,2]==4)}")
    print(f"win rate (pnl>0): {(valid[:,1]>0).mean():.3f}; avg pnl stop-days={valid[valid[:,2]==1][:,1].mean():.0f} tgt-days={valid[valid[:,2]==2][:,1].mean():.0f} eod={valid[valid[:,2]==4][:,1].mean():.0f}")
    print(f"avg pnl pts: stop={valid[valid[:,2]==1][:,1].mean():.1f} target={valid[valid[:,2]==2][:,1].mean():.1f} eod={valid[valid[:,2]==4][:,1].mean():.1f}")
    print(f"\n== time stops ==")
    for mins in (15, 30, 60, 120, 240):
        show(f"tstop={mins}m", ev(tstop=mins * 60))
    print(f"\n== per-bucket stop (risk parity-ish) ==")
    # emulate via separate runs is complex; approximate with global variants
    for st, tg in ((35, 80), (40, 80), (45, 85)):
        show(f"stop={st} tgt={tg}", ev(stop=float(st), tgt=float(tg)))
