"""Round 8: trade-management variants on 0.5382 (all failures are MLL busts -> cut variance).
(a) partial take at +P half, stop->entry, target T rest
(b) breakeven after +B (round2 said hurt on old base; retest on new)
(c) scratch time-stop: flat at T1 if |pnl|<X
(d) loser time-stop: flat at T1 if pnl<0
(e) hot-day tight brackets 35/70 when push small / overnight quiet (feature-gated)"""
import numpy as np
from numba import njit
import sys
sys.path.insert(0, ".")
import round5
from round5 import day_arrays, bounds, NS, sess, t, o, h, l, c, START, EXIT, feat
from eval.sim import run_days, score

side, ia, bkt = day_arrays()
base = np.zeros(NS, dtype=np.int64)
for s in range(NS):
    if bkt[s] >= 0:
        v = (7, 9, 6, 0)[bkt[s]]
        if v > 0 and ia[s] >= 0.05:
            v = max(1, v - 3)
        base[s] = v

@njit(cache=True)
def build8(t, c, bounds, side, sz_day, start, exit_, stop, tgt,
           part_at, be_after, scratch_t, scratch_w, loser_t):
    """part_at>0: close half at +part_at, stop->entry, target tgt for rest.
    be_after>0: stop to entry after pnl>=be_after (no partial).
    scratch_t>0: flat at scratch_t if |pnl|<scratch_w. loser_t>0: flat at loser_t if pnl<0."""
    out = np.zeros(len(c), dtype=np.int64)
    for s in range(len(bounds) - 1):
        sz = sz_day[s]
        if sz <= 0:
            continue
        pos, entry, done, scaled = 0, 0.0, False, False
        for k in range(bounds[s], bounds[s + 1]):
            if t[k] < start or t[k] >= exit_:
                pos = 0
            elif pos == 0 and not done:
                pos, entry, done, scaled = side[s] * sz, c[k], True, False
            elif pos != 0:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                stp = stop
                if part_at > 0 and scaled:
                    stp = 0.0
                elif part_at > 0 and pnl >= part_at and (pos > 1 or pos < -1):
                    pos = pos // 2          # scale out half
                    scaled = True
                    out[k] = pos
                    continue
                if be_after > 0 and pnl >= be_after:
                    stp = 0.0
                if scratch_t > 0 and t[k] >= scratch_t and abs(pnl) < scratch_w:
                    pos = 0
                elif loser_t > 0 and t[k] >= loser_t and pnl < 0:
                    pos = 0
                elif pnl <= -stp or pnl >= tgt:
                    pos = 0
            out[k] = pos
    return out

def ev(tag, part_at=0.0, be_after=0.0, scratch_t=0, scratch_w=0.0, loser_t=0,
       sz_day=None, stop=40.0, tgt=80.0):
    sz_day = base if sz_day is None else sz_day
    target = build8(t, c, bounds, side, sz_day, START, EXIT, stop, tgt,
                    part_at, be_after, scratch_t, scratch_w, loser_t)
    day_pnl, day_min, _ = run_days(sess, t, o, h, l, c, target, NS)
    sc, rates, passed = score(day_pnl, day_min)
    print(f"{tag:46s} score={sc:.4f} h1={rates[0]:.3f} h2={rates[1]:.3f} "
          f"days={np.mean(passed) if passed else float('nan'):.1f} pnl={day_pnl.sum():.0f} "
          f"worst={day_pnl.min():.0f}", flush=True)
    return sc

if __name__ == "__main__":
    ev("BASE 0.5382")
    print("== partial: half off at +P, stop->entry, rest to T ==")
    for P, T in ((35, 70), (40, 80), (40, 100), (30, 60), (45, 90), (35, 80)):
        ev(f"part {P}/{T}", part_at=P, tgt=T)
    print("== breakeven after +B ==")
    for B in (30, 40, 50):
        ev(f"be_after={B}", be_after=B)
    print("== scratch time-stop (flat if |pnl|<W at T) ==")
    for T1, W in ((10 * 3600 + 30 * 60, 10), (11 * 3600, 15), (11 * 3600, 25), (12 * 3600, 15)):
        ev(f"scratch {T1//3600}:{T1//60%60:02d} w={W}", scratch_t=T1, scratch_w=W)
    print("== loser time-stop (flat if pnl<0 at T) ==")
    for T1 in (10 * 3600 + 30 * 60, 11 * 3600, 12 * 3600):
        ev(f"loser cut {T1//3600}:{T1//60%60:02d}", loser_t=T1)
    print("== combo: part 40/80 + loser cut 11:00 ==")
    ev("part40/80 + cut11", part_at=40.0, tgt=80.0, loser_t=11 * 3600)
    ev("part35/70 + cut11", part_at=35.0, tgt=70.0, loser_t=11 * 3600)
