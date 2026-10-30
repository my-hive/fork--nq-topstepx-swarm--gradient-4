"""gu4_f: combos around Thursday hot_agree skip: ha size, other-day hot_agree skips (placebo),
Thursday bracket tightening instead of skip, + hot_agree 9."""
import sys
sys.path.insert(0, ".")
from harness import score_signals
from gu4_e import make

if __name__ == "__main__":
    score_signals(make(), "BASE (expect 0.6856)")
    score_signals(make(thu_ha=0.0), "Thu skip hot_agree (0.7020)")
    score_signals(make(thu_ha=0.0, ha=9), "Thu skip ha + ha sz9")
    score_signals(make(thu_ha=0.0, hd=8), "Thu skip ha + hd sz8")
    score_signals(make(thu_ha=0.0, ha=9, hd=8), "Thu skip ha + ha9 hd8")
    # placebo: hot_agree skip on other days
    for d in (0, 1, 2, 4):
        score_signals(make(dow_ha={d: 0.0}), f"dow{d} skip hot_agree (placebo)")
    # dow_ha skip on Thursday handled by thu_ha; also try skipping hot_agree everywhere EXCEPT best days
    score_signals(make(dow_ha={0: 0.0, 1: 0.0}), "Mon+Tue skip hot_agree")
