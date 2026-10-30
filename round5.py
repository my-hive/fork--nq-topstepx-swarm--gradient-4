"""Round 5: self-contained coordinate sweep around 0.5382 config.
Sizes (7,9,6,0) hot+dis/hot+agree/cold+dis/cold+agree, imb th=.05 shrink 3, 40/80, ON 50/175."""
import numpy as np
import pandas as pd
from numba import njit
import sys
sys.path.insert(0, ".")
from eval.sim import run_days, score

df = pd.read_parquet("/home/shawn/hive-data/nq/train.parquet")
sess = pd.factorize(df.session, sort=True)[0]
t, o, h, l, c = (df[k].values for k in ("ct_sec", "open", "high", "low", "close"))
buy, sell = df.buy_vol.values, df.sell_vol.values
NS = sess.max() + 1
START, EXIT = 8 * 3600 + 35 * 60, 14 * 3600 + 30 * 60

feat = {}
for s in range(NS):
    idx = np.where(sess == s)[0]
    tt, cc = t[idx], c[idx]
    m = (tt >= 8 * 3600 + 30 * 60) & (tt < START)
    if not m.any():
        continue
    w = np.where(m)[0]
    onv, mv = cc[w[0]] - cc[0], cc[w[-1]] - cc[w[0]]
    if mv == 0:
        continue
    bv, sv = buy[idx][w].sum(), sell[idx][w].sum()
    feat[s] = dict(onv=onv, mv=mv, imb=(bv - sv) / max(bv + sv, 1),
                   side=-1 if mv >= 0 else 1)

def day_arrays(on_lo=50.0, on_hi=175.0):
    side = np.zeros(NS, dtype=np.int64); ia = np.zeros(NS)
    bkt = np.full(NS, -1, dtype=np.int64)
    for s, f in feat.items():
        hot = not (on_lo <= abs(f["onv"]) <= on_hi)
        agree = np.sign(f["mv"]) == np.sign(f["onv"]) if f["onv"] != 0 else True
        side[s] = f["side"]; ia[s] = f["imb"] * np.sign(f["mv"])
        bkt[s] = (0 if hot else 2) + (1 if agree else 0)
    return side, ia, bkt

@njit(cache=True)
def build(t, c, sess, bounds, side, sz_day, st_day, tg_day, start, exit_):
    out = np.zeros(len(c), dtype=np.int64)
    for s in range(len(bounds) - 1):
        sz = sz_day[s]
        if sz <= 0:
            continue
        pos, entry, done = 0, 0.0, False
        for k in range(bounds[s], bounds[s + 1]):
            if t[k] < start or t[k] >= exit_:
                pos = 0
            elif pos == 0 and not done:
                pos, entry, done = side[s] * sz, c[k], True
            elif pos != 0:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if pnl <= -st_day[s] or pnl >= tg_day[s]:
                    pos = 0
            out[k] = pos
    return out

bounds = np.zeros(NS + 1, dtype=np.int64)
for s in range(NS):
    bounds[s] = np.searchsorted(sess, s)
bounds[NS] = len(sess)

def evaluate(side, ia, bkt, sz=(7, 9, 6, 0), th=0.05, sh=3,
             stop=40.0, tgt=80.0, start=START, exit_=EXIT):
    sz_day = np.zeros(NS, dtype=np.int64)
    for s in range(NS):
        if bkt[s] >= 0:
            v = sz[bkt[s]]
            if v > 0 and ia[s] >= th:
                v = max(1, v - sh)
            sz_day[s] = v
    st = np.full(NS, stop, dtype=np.float64)
    tg = np.full(NS, tgt, dtype=np.float64)
    target = build(t, c, sess, bounds, side, sz_day, st, tg, start, exit_)
    day_pnl, day_min, _ = run_days(sess, t, o, h, l, c, target, NS)
    sc, rates, passed = score(day_pnl, day_min)
    return dict(sc=sc, h1=rates[0], h2=rates[1], days=(np.mean(passed) if passed else np.nan),
                pnl=day_pnl.sum(), worst=day_pnl.min())

def show(tag, r):
    print(f"{tag:44s} score={r['sc']:.4f} h1={r['h1']:.3f} h2={r['h2']:.3f} "
          f"days={r['days']:.1f} pnl={r['pnl']:.0f} worst={r['worst']:.0f}", flush=True)

if __name__ == "__main__":
    side, ia, bkt = day_arrays()
    show("BASE 0.5382", evaluate(side, ia, bkt))
    print("== size one-dim sweep ==")
    for d, vals in ((0, range(5, 12)), (1, range(7, 13)), (2, range(4, 10))):
        for v in vals:
            if v == (7, 9, 6)[d]:
                continue
            sz = [7, 9, 6, 0]; sz[d] = v
            show(f"sz{d}={v}", evaluate(side, ia, bkt, tuple(sz)))
    print("== stop/target ==")
    for st_, tg_ in ((30, 80), (35, 80), (40, 70), (40, 90), (40, 100), (45, 80),
                     (50, 80), (35, 70), (45, 90), (50, 100), (40, 60), (40, 120)):
        show(f"st={st_} tg={tg_}", evaluate(side, ia, bkt, stop=st_, tgt=tg_))
    print("== ON bounds ==")
    for lo, hi in ((40, 175), (60, 175), (50, 150), (50, 200), (40, 200),
                   (60, 150), (30, 175), (50, 250), (70, 175), (50, 125)):
        sd2, ia2, bk2 = day_arrays(lo, hi)
        show(f"on {lo}/{hi}", evaluate(sd2, ia2, bk2))
    print("== entry/exit times ==")
    for st_ in (8 * 3600 + 36 * 60, 8 * 3600 + 40 * 60, 8 * 3600 + 45 * 60, 8 * 3600 + 50 * 60):
        show(f"entry {st_//60%60}:{st_%60:02d}", evaluate(side, ia, bkt, start=st_))
    for ex_ in (11 * 3600, 12 * 3600, 13 * 3600, 14 * 3600, 15 * 3600, 14 * 3600 + 55 * 60):
        show(f"exit {ex_//3600}:{ex_//60%60:02d}", evaluate(side, ia, bkt, exit_=ex_))
