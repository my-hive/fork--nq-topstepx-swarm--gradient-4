"""Sweep v6 on shrink-3 base (9912823 = 0.5382).
New hypotheses: push-magnitude size scaling, pullback entry, overnight volume gate,
per-bucket stop/target, overnight onv finer tiers."""
import sys
import numpy as np
import pandas as pd
from numba import njit

sys.path.insert(0, ".")
from eval.sim import run_days, score

df = pd.read_parquet("/home/shawn/hive-data/nq/train.parquet")
sess = pd.factorize(df.session, sort=True)[0]
t, o, h, l, c = (df[k].values for k in ("ct_sec", "open", "high", "low", "close"))
bv, sv = df.buy_vol.values, df.sell_vol.values
NS = sess.max() + 1

ON_LO, ON_HI = 50.0, 175.0
START = 8 * 3600 + 35 * 60
EXIT = 14 * 3600 + 30 * 60
STOP, TGT = 40.0, 80.0
TH, SH = 0.05, 3

srt = np.where(np.r_[True, sess[1:] != sess[:-1]])[0]
end = np.r_[srt[1:] - 1, len(sess) - 1]

SIDE = np.zeros(NS, dtype=np.int64)
B = np.full(NS, 2, dtype=np.int64)
HOT = np.zeros(NS, dtype=np.int64)
IA = np.zeros(NS)
PUSH = np.zeros(NS)
ONV = np.zeros(NS)
ONVOL = np.zeros(NS)  # overnight total volume
for s in range(NS):
    i0, i1 = srt[s], end[s]
    tt, cc = t[i0:i1 + 1], c[i0:i1 + 1]
    w = np.where((tt >= 8 * 3600 + 30 * 60) & (tt < START))[0]
    on_mask = tt < 8 * 3600 + 30 * 60
    ONVOL[s] = (bv[i0:i1 + 1][on_mask] + sv[i0:i1 + 1][on_mask]).sum()
    if len(w) == 0:
        continue
    onv = cc[w[0]] - cc[0]
    mv = cc[w[-1]] - cc[w[0]]
    if mv == 0:
        continue
    bsum, ssum = bv[i0 + w].sum(), sv[i0 + w].sum()
    imb = (bsum - ssum) / max(bsum + ssum, 1.0)
    IA[s] = imb * (1 if mv > 0 else -1)
    PUSH[s] = abs(mv)
    ONV[s] = onv
    hot = abs(onv) < ON_LO or abs(onv) > ON_HI
    HOT[s] = 1 if hot else 0
    if onv == 0:
        agree = True
    else:
        agree = (mv > 0 and onv > 0) or (mv < 0 and onv < 0)
    if hot and agree:
        B[s] = 0
    elif hot:
        B[s] = 1
    elif agree:
        B[s] = 2
    else:
        B[s] = 3
    SIDE[s] = -1 if mv >= 0 else 1

BASE_SZ = np.zeros(NS, dtype=np.int64)
for s in range(NS):
    b = B[s]
    v = [9, 7, 0, 6][b]
    if v > 0 and IA[s] >= TH:
        v = max(1, v - SH)
    BASE_SZ[s] = v

@njit(cache=True)
def build(t, c, srt, end, SIDE, SZ, st, tg, start, exit_, pb):
    """pb: pullback entry pts — enter only after price moves pb pts ADVERSE to fade
    (i.e., extends the push beyond the 8:35 close by pb), else enter at first bar."""
    out = np.zeros(len(c), dtype=np.int64)
    for s in range(len(srt)):
        sz = SZ[s]
        if sz == 0:
            continue
        side = SIDE[s]
        pos, entry, done, armed = 0, 0.0, False, pb <= 0.0
        ref = -1.0
        for k in range(srt[s], end[s] + 1):
            if t[k] < start or t[k] >= exit_:
                pos = 0
            elif pos == 0 and not done:
                if not armed:
                    if ref < 0.0:
                        ref = c[k]
                    # adverse extension beyond entry bar close by pb pts
                    if (side < 0 and c[k] >= ref + pb) or (side > 0 and c[k] <= ref - pb):
                        armed = True
                if armed:
                    pos, entry, done = side * sz, c[k], True
            elif pos != 0:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if pnl <= -st[s] or pnl >= tg[s]:
                    pos = 0
            out[k] = pos
    return out

def evaluate(target):
    day_pnl, day_min, trades = run_days(sess, t, o, h, l, c, target, NS)
    sc, rates, passed = score(day_pnl, day_min)
    ad = np.mean(passed) if passed else np.nan
    return sc, rates, ad, day_pnl.sum(), day_pnl.min(), (day_pnl > 0).mean(), trades.sum()

def show(tag, r):
    sc, rates, ad, tp, wd, wr, ct_ = r
    print(f"{tag:<38} score={sc:.4f} h1={rates[0]:.3f} h2={rates[1]:.3f} days={ad:.1f} "
          f"pnl={tp:.0f} worst={wd:.0f} wr={wr:.2f} ct={ct_}", flush=True)

def mk(sz=None, st=None, tg=None, pb=0.0, exit_=EXIT):
    if sz is None:
        sz = BASE_SZ
    if st is None:
        st = np.full(NS, STOP)
    if tg is None:
        tg = np.full(NS, TGT)
    return build(t, c, srt, end, SIDE, np.asarray(sz, dtype=np.int64),
                 np.asarray(st, dtype=np.float64), np.asarray(tg, dtype=np.float64),
                 START, exit_, pb)

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    show("BASE expect 0.5382", evaluate(mk()))

    if which in ("all", "pushsz"):
        # size scales with push magnitude
        for k in (0.05, 0.1, 0.15):
            for lo, hi in ((0, 12),):
                sz = np.where(BASE_SZ > 0,
                              np.clip(np.round(BASE_SZ + k * (PUSH - 30)), 1, 12), 0).astype(np.int64)
                show(f"sz + {k}*(push-30)", evaluate(mk(sz=sz)))
        for mp in (20, 30, 40):
            sz = np.where(BASE_SZ > 0, np.where(PUSH >= mp, BASE_SZ, 0), 0).astype(np.int64)
            show(f"skip push<{mp}", evaluate(mk(sz=sz)))
    if which in ("all", "pb"):
        for pb in (5.0, 10.0, 15.0, 20.0):
            show(f"pullback entry {pb:.0f}pt", evaluate(mk(pb=pb)))
    if which in ("all", "onvol"):
        med = np.median(ONVOL)
        for q in (0.25, 0.5, 0.75):
            thr = np.quantile(ONVOL, q)
            sz = np.where(ONVOL >= thr, BASE_SZ, 0).astype(np.int64)
            show(f"only overnight vol>=q{q:.2f}", evaluate(mk(sz=sz)))
            sz2 = np.where(ONVOL < thr, BASE_SZ, 0).astype(np.int64)
            show(f"only overnight vol<q{q:.2f}", evaluate(mk(sz=sz2)))
    if which in ("all", "bkt"):
        # per-bucket stop/tgt
        for b, st_, tg_ in ((3, 30, 60), (3, 50, 100), (0, 30, 60), (0, 50, 100), (1, 30, 60), (1, 50, 100)):
            st = np.full(NS, STOP); tg = np.full(NS, TGT)
            m = B == b
            st[m] = st_; tg[m] = tg_
            show(f"bucket {b} {st_}/{tg_}", evaluate(mk(st=st, tg=tg)))
