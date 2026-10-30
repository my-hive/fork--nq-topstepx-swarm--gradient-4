"""Round 3: grid around flip_dis with tgt 50-70 and dsz +1..+4."""
import sys
from mg1_flip import build, evaluate, show, STOP0, t, c, i0, i1, SIDE, SZ, HOT, AGRE, SHRUNK

def run(tag, **kw):
    show(tag, evaluate(build(t, c, i0, i1, SIDE, SZ, HOT, AGRE, SHRUNK,
                             stop=STOP0, tgt=STOP0*2, **kw)))

BASE = dict(flip_agree=0, flip_dis=1, flip_shrunk_only=0, flip_unshrunk_only=0)

if __name__ == "__main__":
    for d in (0, 1, 2, 3, 4):
        for fs in (30, 35, 40, 45):
            for ft in (50, 60, 70):
                kw = dict(BASE); kw.update(flip_dsz=d, flip_stop=float(fs), flip_tgt=float(ft))
                run(f"flip_dis dsz{d:+d} {fs}/{ft}", **kw)
