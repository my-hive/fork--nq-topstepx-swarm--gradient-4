"""gu4_ad: micro-grid on verified 0.8908 base: f_tgt, big tgt, retr, imb_th interplay."""
import sys
sys.path.insert(0, ".")
from harness import score_signals
from gu4_ac import make

if __name__ == "__main__":
    # gu4_ac.make has fixed f_stop=45, big 30/70, retr 8, imb 0.02 — patch via source exec
    import re
    src = open("gu4_ac.py").read()
    src = src.replace('def make(cd_on_lo=None, cd_push_hi=None, mode="skip", ha=10, hd=7, cd=6):',
                      'def make(cd_on_lo=None, cd_push_hi=None, mode="skip", ha=10, hd=7, cd=6, f_tgt=60.0, big_tgt=70.0, retr=8.0, imb_th=0.02):')
    src = src.replace('if sz > 0 and imb * np.sign(mv) >= IMB_TH:', 'if sz > 0 and imb * np.sign(mv) >= imb_th:')
    src = src.replace('if abs(ext - cc[where[-1]]) < RETR:', 'if abs(ext - cc[where[-1]]) < retr:')
    src = src.replace('stop, tgt = (30.0, 70.0) if abs(onv) > 200 else (40.0, 80.0)',
                      'stop, tgt = (30.0, big_tgt) if abs(onv) > 200 else (40.0, 80.0)')
    src = src.replace('if pnl <= -45.0 or pnl >= 60.0:', 'if pnl <= -45.0 or pnl >= f_tgt:')
    src = src.replace('if __name__ == "__main__":', 'if False:')
    ns = {}
    exec(compile(src, "gu4_ac_patched", "exec"), ns)
    mk = ns["make"]
    score_signals(mk(), "BASE 0.8908 recheck")
    for ft in (55.0, 65.0, 70.0):
        score_signals(mk(f_tgt=ft), f"f_tgt {ft:.0f}")
    for bt in (75.0, 80.0):
        score_signals(mk(big_tgt=bt), f"big tgt {bt:.0f}")
    for r in (8.5, 9.0, 10.0):
        score_signals(mk(retr=r), f"retr {r}")
    for it in (0.015, 0.025, 0.035):
        score_signals(mk(imb_th=it), f"imb_th {it}")
    for ft, bt in ((60.0, 75.0), (65.0, 75.0)):
        score_signals(mk(f_tgt=ft, big_tgt=bt), f"f_tgt {ft:.0f} big {bt:.0f}")
