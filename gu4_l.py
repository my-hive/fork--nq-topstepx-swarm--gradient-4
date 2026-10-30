"""gu4_l: fine-tune imb_th on 0.7903 base + combos with zone 50/200."""
import sys
sys.path.insert(0, ".")
from harness import score_signals
from gu4_k import make

if __name__ == "__main__":
    score_signals(make(imb_th=0.03), "BEST 0.7903 recheck")
    for v in (0.0, 0.01, 0.02, 0.025, 0.035, 0.04):
        score_signals(make(imb_th=v), f"imb_th {v}")
    for v in (0.02, 0.03):
        score_signals(make(imb_th=v, on_hi=200.0), f"imb {v} zone 50/200")
        score_signals(make(imb_th=v, big_th=225.0), f"imb {v} big_th 225")
    for sh in (2, 3):
        score_signals(make(imb_th=0.03, shrink=sh), f"imb .03 shrink {sh}")
