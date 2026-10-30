"""Robustness jitter: random perturbations of the 0.8905 config; distribution of scores."""
import numpy as np
import sweep3 as s3
from sweep3 import NS, ONV, AGREE, PUSH, IA, PVOL, DOW, SIDE, srt, end, t, c, evaluate
from sweep_ca import build_cf as build
from sweep_retr import RETR

rng = np.random.default_rng(7)

def mk(ha=10, hd=7, cd=6, imth=0.03, vol=12000.0, retr=5.0, fst=40.0, ftg=60.0,
       bst=30.0, btgt=70.0, fsz_d=2, on_hi=200.0, tbr=200.0):
    SZ = np.zeros(NS, dtype=np.int64); ST = np.zeros(NS); TG = np.zeros(NS)
    FLIP = np.zeros(NS, dtype=np.int64); FSZ = np.zeros(NS, dtype=np.int64)
    for s in range(NS):
        onv, push = ONV[s], PUSH[s]
        if push == 0:
            continue
        aonv = abs(onv)
        hot = aonv < 50 or aonv > on_hi
        agree = AGREE[s] == 1
        sz = (ha if agree else hd) if hot else (cd if not agree else 0)
        if sz > 0 and IA[s] >= imth:
            sz = max(1, sz - 2)
        if sz > 0 and PVOL[s] < vol:
            sz = 0
        if hot and not agree and sz > 0 and RETR[s] < retr:
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
        ST[s], TG[s] = (bst, btgt) if aonv > tbr else (40.0, 80.0)
        flip = (not agree) or (agree and aonv > tbr)
        FLIP[s] = 1 if flip else 0
        FSZ[s] = sz + fsz_d
    return build(t, c, srt, end, SIDE, SZ, ST, TG, FLIP, FSZ, fst, ftg, s3.START, 14 * 3600 + 30 * 60)

if __name__ == "__main__":
    base = evaluate(mk())[0]
    print(f"base={base:.4f}")
    scores = []
    for i in range(24):
        p = dict(
            ha=rng.integers(9, 12), hd=rng.integers(6, 9), cd=rng.integers(5, 8),
            imth=rng.choice([0.02, 0.03, 0.04]), vol=rng.choice([10000.0, 12000.0, 14000.0]),
            retr=rng.choice([4.0, 5.0, 6.0]), fst=rng.choice([38.0, 40.0, 42.0]),
            ftg=rng.choice([55.0, 60.0, 65.0]), bst=rng.choice([28.0, 30.0, 32.0]),
            btgt=rng.choice([65.0, 70.0, 75.0]), fsz_d=rng.integers(1, 4),
            on_hi=rng.choice([195.0, 200.0, 210.0]), tbr=rng.choice([195.0, 200.0, 205.0]),
        )
        sc, rates, ad, tp, wd, wr, ct_ = evaluate(mk(**p))
        scores.append(sc)
        print(f"j{i:02d} ha={p['ha']} hd={p['hd']} cd={p['cd']} imth={p['imth']} vol={p['vol']/1000:.0f}k "
              f"retr={p['retr']:.0f} fst={p['fst']:.0f} ftg={p['ftg']:.0f} btgt={p['btgt']:.0f} "
              f"fsz+{p['fsz_d']}: {sc:.4f} (h1={rates[0]:.3f} h2={rates[1]:.3f})")
    scores = np.array(scores)
    print(f"\njitter: mean={scores.mean():.4f} median={np.median(scores):.4f} "
          f"p25={np.quantile(scores, .25):.4f} max={scores.max():.4f} "
          f"frac>=0.86: {(scores >= 0.86).mean():.2f}")
