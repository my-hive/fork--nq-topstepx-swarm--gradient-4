"""New gate candidates: push-size scaling of stop/target, volume imbalance, tiny-push skip, DOW."""
import numpy as np
import pandas as pd
from numba import njit
import sweep
from sweep import bounds, t, o, h, l, c, sess, NS, START
from eval.sim import run_days, score

df = sweep.df
buy, sell = df.buy_vol.values, df.sell_vol.values
dates = df.session.values
dayofweek = pd.to_datetime(dates).dayofweek

# per-day features
feat = {}
for s in range(NS):
    i0, i1 = bounds[s], bounds[s + 1]
    tt, cc = t[i0:i1], c[i0:i1]
    m = (tt >= 8 * 3600 + 30 * 60) & (tt < START)
    if not m.any():
        feat[s] = None
        continue
    w = np.where(m)[0]
    onv = cc[w[0]] - cc[0]
    mv = cc[w[-1]] - cc[w[0]]
    bv, sv = buy[i0:i1][w].sum(), sell[i0:i1][w].sum()
    imb = (bv - sv) / max(bv + sv, 1)
    feat[s] = dict(onv=onv, mv=mv, push=abs(mv), imb=imb, dow=dayofweek[i0],
                   bucket=sweep.buckets[s], side=sweep.sides[s])

PUSH = np.zeros(NS); IMB = np.zeros(NS); DOW = np.zeros(NS, dtype=np.int64)
for s, f in feat.items():
    if f is None:
        PUSH[s], IMB[s], DOW[s] = 0.0, 0.0, -1
    else:
        PUSH[s], IMB[s], DOW[s] = f["push"], f["imb"], f["dow"]

@njit(cache=True)
def build(t, c, sess, bounds, sides, buckets, sizes, start, exit_, stop, tgt):
    out = np.zeros(len(c), dtype=np.int64)
    for s in range(len(bounds) - 1):
        sz = sizes[buckets[s]] if buckets[s] >= 0 else 0
        if sz == 0:
            continue
        side, pos, entry, done = sides[s], 0, 0.0, False
        for k in range(bounds[s], bounds[s + 1]):
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

def evaluate(sizes, stop=40.0, tgt=80.0, exit_=14 * 3600 + 30 * 60, start=START):
    target = build(t, c, sess, bounds, sweep.sides, sweep.buckets,
                   np.array(sizes, dtype=np.int64), start, exit_, stop, tgt)
    day_pnl, day_min, _ = run_days(sess, t, o, h, l, c, target, NS)
    sc, rates, passed = score(day_pnl, day_min)
    return dict(sc=sc, h1=rates[0], h2=rates[1], days=(np.mean(passed) if passed else np.nan),
                pnl=day_pnl.sum(), worst=day_pnl.min())

def show(tag, r):
    print(f"{tag:50s} score={r['sc']:.4f} h1={r['h1']:.3f} h2={r['h2']:.3f} days={r['days']:.1f} pnl={r['pnl']:.0f} worst={r['worst']:.0f}", flush=True)

def with_sizes(fn):
    """fn(feat_dict or None) -> size; builds sizes array from buckets 0..3 semantics via direct per-day size."""
    sizes_per_day = np.zeros(NS, dtype=np.int64)
    for s in range(NS):
        f = feat.get(s)
        sizes_per_day[s] = 0 if f is None else fn(f)
    return sizes_per_day

@njit(cache=True)
def build_perday(t, c, sess, bounds, sides, sz_day, start, exit_, stop_day, tgt_day):
    out = np.zeros(len(c), dtype=np.int64)
    for s in range(len(bounds) - 1):
        sz = sz_day[s]
        if sz == 0:
            continue
        side, pos, entry, done = sides[s], 0, 0.0, False
        st, tg = stop_day[s], tgt_day[s]
        for k in range(bounds[s], bounds[s + 1]):
            if t[k] < start or t[k] >= exit_:
                pos = 0
            elif pos == 0 and not done:
                pos, entry, done = side * sz, c[k], True
            elif pos != 0:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if pnl <= -st or pnl >= tg:
                    pos = 0
            out[k] = pos
    return out

def evaluate2(sz_day, stop_day, tgt_day, exit_=14 * 3600 + 30 * 60):
    target = build_perday(t, c, sess, bounds, sweep.sides, sz_day, START, exit_,
                          np.full(NS, 40.0) if np.isscalar(stop_day) else np.asarray(stop_day, dtype=np.float64),
                          np.full(NS, 80.0) if np.isscalar(tgt_day) else np.asarray(tgt_day, dtype=np.float64))
    day_pnl, day_min, _ = run_days(sess, t, o, h, l, c, target, NS)
    sc, rates, passed = score(day_pnl, day_min)
    return dict(sc=sc, h1=rates[0], h2=rates[1], days=(np.mean(passed) if passed else np.nan),
                pnl=day_pnl.sum(), worst=day_pnl.min())

BASE = lambda f: [8, 9, 6, 0][f["bucket"]]

if __name__ == "__main__":
    # sanity
    show("BASE", evaluate([8, 9, 6, 0]))
    print("\n== push-scaled stop/target (stop=k*push, tgt=2*stop, floor 25) ==")
    for k in (0.6, 0.8, 1.0, 1.2):
        sd = np.zeros(NS); td = np.zeros(NS)
        for s in range(NS):
            f = feat.get(s)
            if f is None:
                continue
            st = max(25.0, k * f["push"])
            sd[s], td[s] = st, 2 * st
        show(f"k={k}", evaluate2(with_sizes(BASE), sd, td))
    print("\n== tiny-push skip (min push to trade) ==")
    for mp in (15, 20, 25, 30):
        show(f"min_push={mp}", evaluate2(with_sizes(lambda f: 0 if f["push"] < mp else BASE(f)), 40.0, 80.0))
    print("\n== volume-imbalance size mod ==")
    for th in (0.05, 0.10, 0.15):
        def mk(f, th=th):
            b = BASE(f)
            if f["bucket"] in (0, 1) and abs(f["imb"]) < th:  # weak push: shrink
                return b - 2
            return b
        show(f"imb<th hot shrink2 (th={th})", evaluate2(with_sizes(mk), 40.0, 80.0))
    print("\n== DOW skip ==")
    for d in range(5):
        show(f"skip dow={d}", evaluate2(with_sizes(lambda f, d=d: 0 if f["dow"] == d else BASE(f)), 40.0, 80.0))
