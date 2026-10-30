"""Sweep hot-streak booster + re-fade on CD7/FT68 base (4afc9a7). Reports score, days, worst day, shift-robustness."""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval.sim as sim
import strategy

df = pd.read_parquet(os.path.expanduser("~/hive-data/nq/train.parquet"))
sess = pd.factorize(df.session, sort=True)[0]
N = len(df)

def run_signals():
    t = np.nan_to_num(np.asarray(strategy.signals(df), dtype=float)).astype(np.int64)
    return sim.run_days(sess, df.ct_sec.values, df.open.values, df.high.values,
                        df.low.values, df.close.values, t, sess.max() + 1)

def score_of(day_pnl, day_min):
    sc, rates, passed = sim.score(day_pnl, day_min)
    return sc, rates, (np.mean(passed) if passed else float("nan"))

def shift_min(day_pnl, day_min, n=4):
    """gu-4 split-shift test: min over mid±10*n of min(h1,h2) pass rates."""
    tot = len(day_pnl); out = []
    for off in range(-10 * n, 10 * n + 1, 10):
        mid = tot // 2 + off
        a1 = sim.attempts(day_pnl, day_min, 0, mid)
        a2 = sim.attempts(day_pnl, day_min, mid, tot)
        r1 = np.mean([p for p, _ in a1]); r2 = np.mean([p for p, _ in a2])
        out.append(min(r1, r2))
    return min(out)

def setp(**kw):
    for k, v in kw.items():
        setattr(strategy, k, v)

def report(tag):
    day_pnl, day_min, _ = run_signals()
    sc, rates, days = score_of(day_pnl, day_min)
    print(f"{tag:44s} score={sc:.4f} r={rates[0]:.3f}/{rates[1]:.3f} days={days:.1f} worst={day_pnl.min():.0f}")
    return sc, rates, days, day_pnl, day_min

if __name__ == "__main__":
    setp(BOOST_ADD=0, REFADE=0)
    sc, r, d, dp, dm = report("baseline (off)")
    assert abs(sc - 1.0060) < 2e-4, f"baseline broken: {sc}"
    for k, th in [(2, 300), (2, 500), (3, 400), (3, 600), (5, 500), (5, 800)]:
        for add in (1, 2):
            setp(BOOST_K=k, BOOST_TH=float(th), BOOST_ADD=add, REFADE=0)
            report(f"boost K={k} TH={th} +{add}")
    for k, th in [(2, 500), (3, 500)]:
        for add in (1, 2):
            setp(BOOST_K=k, BOOST_TH=float(th), BOOST_ADD=add, REFADE=1)
            report(f"boost K={k} TH={th} +{add} REFADE")
    setp(BOOST_ADD=0, REFADE=1)
    report("REFADE only")
