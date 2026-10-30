"""gu4_g: fine-tune around Thu-skip-hot_agree + ha9 (0.7148): sizes, Thu hot_dis, cd variants."""
import sys
sys.path.insert(0, ".")
from harness import score_signals
from gu4_e import make

if __name__ == "__main__":
    score_signals(make(thu_ha=0.0, ha=9), "BEST 0.7148 recheck")
    for v in (10, 11, 12):
        score_signals(make(thu_ha=0.0, ha=v), f"ha {v}")
    for v in (5, 6, 7):
        score_signals(make(thu_ha=0.0, ha=9, cd=v), f"cd {v}")
    for v in (6, 7, 8):
        score_signals(make(thu_ha=0.0, ha=9, hd=v), f"hd {v}")
    score_signals(make(thu_ha=0.0, ha=9, thu_hd=0.0), "+ Thu skip hot_dis")
    score_signals(make(thu_ha=0.0, ha=9, thu_cd=0.0), "+ Thu skip cold_dis")
    score_signals(make(thu_ha=0.0, ha=9, thu_cd=0.5), "+ Thu cold_dis x0.5")
    score_signals(make(thu_ha=0.0, ha=9, dow_ha={4: 0.5}), "+ Fri hot_agree x0.5")
