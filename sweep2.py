"""Sweep trade-management variants on top of 0.4686 (b221a62).
Variants: breakeven stop, re-entry on re-extreme, entry time shift, min |move| filter."""
import sys
import numpy as np
import pandas as pd
from numba import njit

sys.path.insert(0, ".")
from eval.sim import run_days, score

df = pd.read_parquet("/home/shawn/hive-data/nq/train.parquet")
sess = pd.factorize(df.session, sort=True)[0]
t, o, h, l, c = (df[k].values for k in ("ct_sec", "open", "high", "low", "close"))
NS = sess.max() + 1

ON_LO, ON_HI = 50.0, 175.0
STOP, TGT, EXIT = 40.0, 80.0, 14 * 3600 + 30 * 60
SIZES = np.array([9, 8, 6, 0], dtype=np.int64)  # hot+agree, hot+dis, cold+dis, cold+agree

# per-day precompute
slices, sides, buckets, mvs, onvs = [], [], [], [], []
for s in range(NS):
    idx = np.where(sess == s)[0]
    tt, cc = t[idx], c[idx]
    m = (tt >= 8 * 3600 + 30 * 60) & (tt < 8 * 3600 + 35 * 60)
    if not m.any() or (w := np.where(m)[0]) is None or len(w) == 0:
        slices.append(idx); sides.append(0); buckets.append(3); mvs.append(0.0); onvs.append(0.0); continue
    onv = cc[w[0]] - cc[0]
    mv = cc[w[-1]] - cc[w[0]]
    slices.append(idx)
    if mv == 0:
        sides.append(0); buckets.append(3); mvs.append(0.0); onvs.append(onv); continue
    hot = not (ON_LO <= abs(onv) <= ON_HI)
    agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
    buckets.append((0 if hot else 2) + (1 if agree else 0))
    sides.append(-1 if mv >= 0 else 1)
    mvs.append(mv); onvs.append(onv)

B = np.array(buckets); SIDE = np.array(sides); MV = np.array(mvs)
# session start/end index per session
i0 = np.zeros(NS, dtype=np.int64); i1 = np.zeros(NS, dtype=np.int64)
for s in range(NS):
    idx = slices[s]
    i0[s], i1[s] = idx[0], idx[-1]

@njit(cache=True)
def build(t, c, sess, i0, i1, SIDE, B, sizes, start, exit_, stop, tgt,
          be_trig, reentry, mv_lo):
    """be_trig: pts profit after which stop moves to breakeven (0=off).
    reentry: 0=one-shot; 1=re-enter on new extreme vs first entry price (max 2 trades)
    mv_lo: min |8:30-8:35 move| in pts to trade (0=off)."""
    out = np.zeros(len(c), dtype=np.int64)
    ns = len(i0)
    for s in range(ns):
        if B[s] >= 4 or sizes[B[s]] == 0:
            continue
        if abs(0.0) < mv_lo:  # placeholder, real check below
            pass
        side = SIDE[s]
        sz = sizes[B[s]]
        pos, entry, done, ntr = 0, 0.0, False, 0
        stop_px = stop
        for k in range(i0[s], i1[s] + 1):
            if t[k] < start or t[k] >= exit_:
                pos = 0
            elif pos == 0 and not done:
                pos, entry, done = side * sz, c[k], True
                ntr = 1
            elif pos != 0:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if be_trig > 0.0 and pnl >= be_trig and stop_px == stop:
                    stop_px = 0.0
                if pnl <= -stop_px or pnl >= tgt:
                    pos = 0
                    if reentry == 1 and ntr == 1:
                        # wait for new extreme beyond entry against side, re-enter
                        if pnl <= -stop or pnl <= -stop_px or pnl >= tgt:
                            done = False if pnl <= -stop else done
                        if pnl <= -stop:
                            done = False  # allow re-entry after stop-out
                        ntr = 2
            out[k] = pos
    return out

@njit(cache=True)
def build2(t, c, sess, i0, i1, SIDE, B, MV, sizes, start, exit_, stop, tgt,
           be_trig, reentry, mv_lo):
    out = np.zeros(len(c), dtype=np.int64)
    ns = len(i0)
    for s in range(ns):
        b = B[s]
        if b >= 4 or sizes[b] == 0 or abs(MV[s]) < mv_lo:
            continue
        side = SIDE[s]
        sz = sizes[b]
        pos, entry, done, ntr = 0, 0.0, False, 0
        cur_stop = stop
        ext = 0.0  # best pnl seen
        for k in range(i0[s], i1[s] + 1):
            if t[k] < start or t[k] >= exit_:
                pos = 0
            elif pos == 0 and not done:
                pos, entry, done = side * sz, c[k], True
                ntr += 1
                cur_stop = stop
                ext = 0.0
            elif pos != 0:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if pnl > ext:
                    ext = pnl
                if be_trig > 0.0 and ext >= be_trig and cur_stop == stop:
                    cur_stop = 0.0
                if pnl <= -cur_stop or pnl >= tgt:
                    pos = 0
                    # re-entry: after stop-out, re-enter if price makes new extreme vs entry
                    if ntr == 1 and pnl <= -stop:
                        if reentry == 1:
                            # let done=False only when price extends beyond entry + (stop+1) adverse
                            # i.e., beyond the stop level
                            done = True  # keep one-shot unless price exceeds prior extreme
                            re_ext = True
                            # we handle via check: price beyond stop level in adverse direction
                        # simplified: re-enter when close beyond prior entry by >stop adverse
                        # implement below
            out[k] = pos
    return out

def evaluate(target):
    day_pnl, day_min, trades = run_days(sess, t, o, h, l, c, target, NS)
    sc, rates, passed = score(day_pnl, day_min)
    ad = np.mean(passed) if passed else np.nan
    return sc, rates, ad, day_pnl.sum(), day_pnl.min(), (day_pnl > 0).mean(), trades.sum()

def base_target():
    return build_simple(t, c, sess, i0, i1, SIDE, B, SIZES,
                        8*3600+35*60, EXIT, STOP, TGT)

@njit(cache=True)
def build_simple(t, c, sess, i0, i1, SIDE, B, sizes, start, exit_, stop, tgt):
    out = np.zeros(len(c), dtype=np.int64)
    for s in range(len(i0)):
        b = B[s]
        if b >= 4 or sizes[b] == 0:
            continue
        side, sz = SIDE[s], sizes[b]
        pos, entry, done = 0, 0.0, False
        for k in range(i0[s], i1[s] + 1):
            if t[k] < start or t[k] >= exit_:
                pos = 0
            elif pos == 0 and not done:
                pos, entry, done = side * sz, c[k], True
            elif pos != 0:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if pnl <= -stop or pnl >= tgt:
                    pos = 0
            out[k] = pos
    return out

@njit(cache=True)
def build_be(t, c, sess, i0, i1, SIDE, B, sizes, start, exit_, stop, tgt, be_trig):
    out = np.zeros(len(c), dtype=np.int64)
    for s in range(len(i0)):
        b = B[s]
        if b >= 4 or sizes[b] == 0:
            continue
        side, sz = SIDE[s], sizes[b]
        pos, entry, done, cur_stop = 0, 0.0, False, stop
        for k in range(i0[s], i1[s] + 1):
            if t[k] < start or t[k] >= exit_:
                pos = 0
            elif pos == 0 and not done:
                pos, entry, done, cur_stop = side * sz, c[k], True, stop
            elif pos != 0:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if be_trig > 0.0 and pnl >= be_trig and cur_stop == stop:
                    cur_stop = 0.0
                if pnl <= -cur_stop or pnl >= tgt:
                    pos = 0
            out[k] = pos
    return out

@njit(cache=True)
def build_reentry(t, c, sess, i0, i1, SIDE, B, sizes, start, exit_, stop, tgt):
    """After stop-out, re-enter once if price extends beyond stop price (new extreme)."""
    out = np.zeros(len(c), dtype=np.int64)
    for s in range(len(i0)):
        b = B[s]
        if b >= 4 or sizes[b] == 0:
            continue
        side, sz = SIDE[s], sizes[b]
        pos, entry, done, ntr = 0, 0.0, False, 0
        rearm = 0.0  # adverse price beyond which re-entry allowed
        for k in range(i0[s], i1[s] + 1):
            if t[k] < start or t[k] >= exit_:
                pos = 0
            elif pos == 0 and not done:
                pos, entry, done = side * sz, c[k], True
                ntr += 1
            elif pos != 0:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if pnl <= -stop or pnl >= tgt:
                    if pnl <= -stop and ntr == 1:
                        # allow one re-entry if price goes beyond the stop price by >= stop
                        rearm = entry - side * 2.0 * stop  # beyond stop by another stop
                        done = False
                        ntr += 1
                    pos = 0
            elif pos == 0 and not done and ntr == 2 and rearm != 0.0:
                pass
            out[k] = pos
        # note: re-entry condition handled crudely; refine below in python
    return out

# ---- cleaner re-entry builder (python-level, one pass per session, fast enough with numba if needed)
@njit(cache=True)
def build_reentry2(t, c, sess, i0, i1, SIDE, B, sizes, start, exit_, stop, tgt):
    """Stop-out closes trade. If price later extends beyond stop-out price by (stop-10) pts
    in the adverse direction (i.e., new low/high), re-enter same direction once."""
    out = np.zeros(len(c), dtype=np.int64)
    for s in range(len(i0)):
        b = B[s]
        if b >= 4 or sizes[b] == 0:
            continue
        side, sz = SIDE[s], sizes[b]
        pos, entry, done, ntr = 0, 0.0, False, 0
        stop_px = 0.0
        can_re = False
        for k in range(i0[s], i1[s] + 1):
            if t[k] < start or t[k] >= exit_:
                pos = 0
            elif pos == 0 and not done:
                pos, entry, done = side * sz, c[k], True
                ntr += 1
                can_re = False
            elif pos != 0:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if pnl <= -stop or pnl >= tgt:
                    if pnl <= -stop and ntr == 1:
                        can_re = True
                    pos = 0
            # re-entry trigger: flat after stop, price beyond entry-stop level
            if pos == 0 and can_re and not done:
                # price beyond the original stop level by margin
                if side > 0 and c[k] <= entry - stop - 1.0:
                    pos, entry, done = side * sz, c[k], True
                    can_re = False
                elif side < 0 and c[k] >= entry + stop + 1.0:
                    pos, entry, done = side * sz, c[k], True
                    can_re = False
            out[k] = pos
    return out

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"

    if which in ("all", "base"):
        sc, rates, ad, tp, wd, wr, ct_ = evaluate(base_target())
        print(f"BASE               score={sc:.4f} h1={rates[0]:.3f} h2={rates[1]:.3f} days={ad:.1f} pnl={tp:.0f} worst={wd:.0f} wr={wr:.2f} ct={ct_}")

    if which in ("all", "be"):
        for be in (20.0, 30.0, 40.0, 50.0, 60.0):
            tgt_ = build_be(t, c, sess, i0, i1, SIDE, B, SIZES,
                            8*3600+35*60, EXIT, STOP, TGT, be)
            sc, rates, ad, tp, wd, wr, ct_ = evaluate(tgt_)
            print(f"BE after +{be:.0f}      score={sc:.4f} h1={rates[0]:.3f} h2={rates[1]:.3f} days={ad:.1f} pnl={tp:.0f} worst={wd:.0f} wr={wr:.2f} ct={ct_}")

    if which in ("all", "re"):
        tgt_ = build_reentry2(t, c, sess, i0, i1, SIDE, B, SIZES,
                              8*3600+35*60, EXIT, STOP, TGT)
        sc, rates, ad, tp, wd, wr, ct_ = evaluate(tgt_)
        print(f"REENTRY beyond stop score={sc:.4f} h1={rates[0]:.3f} h2={rates[1]:.3f} days={ad:.1f} pnl={tp:.0f} worst={wd:.0f} wr={wr:.2f} ct={ct_}")

    if which in ("all", "entry"):
        for emin in range(5, 16, 1):
            st = 8*3600 + 30*60 + emin*60
            # rebuild sides from window 8:30..st
            sides2 = np.zeros(NS, dtype=np.int64)
            for s in range(NS):
                idx = slices[s]
                tt, cc = t[idx], c[idx]
                m = (tt >= 8*3600+30*60) & (tt < st)
                if not m.any():
                    sides2[s] = 0; continue
                w = np.where(m)[0]
                mv = cc[w[-1]] - cc[w[0]]
                sides2[s] = -1 if mv >= 0 else 1
            S2 = np.where(SIDE != 0, sides2, 0)
            tgt_ = build_simple(t, c, sess, i0, i1, S2, B, SIZES, st, EXIT, STOP, TGT)
            sc, rates, ad, tp, wd, wr, ct_ = evaluate(tgt_)
            print(f"entry 8:{30+emin:02d}        score={sc:.4f} h1={rates[0]:.3f} h2={rates[1]:.3f} days={ad:.1f} pnl={tp:.0f} worst={wd:.0f} wr={wr:.2f} ct={ct_}")
