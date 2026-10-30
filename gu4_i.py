"""gu4_i: combos: fri_big size, mon_big size, bigflip off by day."""
import sys
sys.path.insert(0, ".")
from harness import score_signals
from gu4_h import make

if __name__ == "__main__":
    score_signals(make(fri_big=0.5), "cur best 0.7654 recheck")
    score_signals(make(fri_big=0.5, mon_big=0.5), "fri .5 + mon .5")
    score_signals(make(fri_big=0.5, mon_big=0.7), "fri .5 + mon .7")
    score_signals(make(fri_big=0.5, bigflip_dow_off=(4,)), "fri .5 + no bigflip Fri")
    score_signals(make(fri_big=0.5, bigflip_dow_off=(0,)), "fri .5 + no bigflip Mon")
    score_signals(make(fri_big=0.7, mon_big=0.7), "fri .7 + mon .7")
    score_signals(make(fri_big=0.5, hd=7), "sanity hd7")
    # non-big mon/fri hot_agree shrink
    score_signals(make(fri_big=0.5, mon_ha=0.7), "fri .5 + mon_ha .7")
    score_signals(make(fri_big=0.5, fri_ha=0.7), "fri .5 + fri_ha .7")
