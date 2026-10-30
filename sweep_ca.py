"""Counterfactual: cold+agree (dead zone) fade at small size — does any sub-tier have edge?"""
import numpy as np
from numba import njit
import sweep3 as s3
from sweep3 import NS, srt, end, t, c, ONV, AGREE, PUSH, IA, PVOL, DOW, SIDE, evaluate, show

@njit(cache=True)
def build_cf(t, c, srt, end, SIDE, SZ, ST, TG, FLIP, FSZ, FST, FTG, start, exit_):
    out = np.zeros(len(c), dtype=np.int64)
    for s in range(len(srt)):
        sz = SZ[s]
        if sz == 0:
            continue
        side = SIDE[s]
        pos, entry, state = 0, 0.0, 0
        for k in range(srt[s], end[s] + 1):
            if t[k] < start or t[k] >= exit_:
                pos = 0
            elif state == 0:
                pos, entry, state = side * sz, c[k], 1
            elif pos != 0 and state == 1:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if pnl <= -ST[s]:
                    if FLIP[s]:
                        pos, entry, state = -side * FSZ[s], c[k], 3
                    else:
                        pos, state = 0, 4
                elif pnl >= TG[s]:
                    pos, state = 0, 4
            elif pos != 0 and state == 3:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if pnl <= -FST or pnl >= FTG:
                    pos, state = 0, 4
            out[k] = pos
    return out

def mk(ca_sz=0, ca_lo=50.0, ca_hi=200.0):
    SZ = np.zeros(NS, dtype=np.int64); ST = np.zeros(NS); TG = np.zeros(NS)
    FLIP = np.zeros(NS, dtype=np.int64); FSZ = np.zeros(NS, dtype=np.int64)
    for s in range(NS):
        onv, push = ONV[s], PUSH[s]
        if push == 0:
            continue
        aonv = abs(onv)
        hot = aonv < 50 or aonv > 200
        agree = AGREE[s] == 1
        sz = (9 if agree else 7) if hot else (6 if not agree else 0)
        if not hot and agree and ca_lo <= aonv <= ca_hi:
            sz = ca_sz  # counterfactual cold+agree fade
        if sz > 0 and IA[s] >= 0.03:
            sz = max(1, sz - 2)
        if sz > 0 and PVOL[s] < 12000.0:
            sz = 0
        dow = DOW[s]
        big_agree = agree and aonv > 200
        if big_agree and dow in (0, 4):
            sz = int(round(sz * 0.5))
        if hot and agree and dow == 3:
            sz = 0
        SZ[s] = sz
        if sz == 0:
            continue
        ST[s], TG[s] = (30.0, 60.0) if aonv > 200 else (40.0, 80.0)
        flip = (not agree) or (agree and aonv > 200)
        FLIP[s] = 1 if flip else 0
        FSZ[s] = sz + 1
    return build_cf(t, c, srt, end, SIDE, SZ, ST, TG, FLIP, FSZ, 40.0, 60.0,
                    s3.START, 14 * 3600 + 30 * 60)

if __name__ == "__main__":
    # counterfactual pnl stats for cold+agree fade at sz4 (score impact secondary)
    from eval.sim import run_days
    for lo, hi in ((50, 100), (100, 150), (150, 200)):
        tgt = mk(ca_sz=4, ca_lo=lo, ca_hi=hi - 0.01)
        dp, dm, tr = run_days(s3.sess, t, s3.o, s3.h, s3.l, c, tgt, NS)
        base_dp, _, _ = run_days(s3.sess, t, s3.o, s3.h, s3.l, c, mk(ca_sz=0), NS)
        delta = dp - base_dp
        m = delta != 0
        print(f"cold+agree fade sz4 |onv| {lo}-{hi}: n={m.sum()} mean_delta={delta[m].mean():.0f} "
              f"tot={delta[m].sum():.0f} win={(delta[m] > 0).mean():.2f}")
    for sz in (3, 4, 5):
        show(f"ca fade sz{sz} (50-200)", evaluate(mk(ca_sz=sz)))
