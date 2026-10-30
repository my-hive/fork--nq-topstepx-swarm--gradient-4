"""Round 2 flip sweep: refine around 'flip hot+dis only' (0.5385) and 'flip shrunk only' (0.5323)."""
import sys
import numpy as np
from mg1_flip import (build, evaluate, show, STOP0, TGT0,
                      t, c, i0, i1, SIDE, SZ, HOT, AGRE, SHRUNK)

def run(tag, **kw):
    show(tag, evaluate(build(t, c, i0, i1, SIDE, SZ, HOT, AGRE, SHRUNK,
                             stop=STOP0, tgt=TGT0, **kw)))

BASE = dict(flip_agree=0, flip_dis=1, flip_shrunk_only=0, flip_unshrunk_only=0,
            flip_dsz=0, flip_stop=STOP0, flip_tgt=TGT0)

if __name__ == "__main__":
    for fs, ft in ((40,80),(30,60),(30,80),(35,80),(40,60),(25,60),(20,50),(30,100),(45,90)):
        kw = dict(BASE); kw.update(flip_stop=float(fs), flip_tgt=float(ft))
        run(f"flip_dis {fs}/{ft}", **kw)
    for d in (-3,-2,-1,1,2):
        kw = dict(BASE); kw.update(flip_dsz=d)
        run(f"flip_dis dsz{d:+d}", **kw)
    # flip_dis but only when ALSO shrunk / NOT shrunk (interaction with imb gate)
    for name, extra in (("shrunk_only", dict(flip_shrunk_only=1, flip_dis=0)),
                        ("unshrunk_only", dict(flip_unshrunk_only=1, flip_dis=0))):
        kw = dict(BASE); kw.update(extra)
        run(f"flip_dis {name}", **kw)
    # also allow cold+dis flip via dis flag when hot flag absent is impossible;
    # approximate cold+dis flip separately below using flip on bucket 3
