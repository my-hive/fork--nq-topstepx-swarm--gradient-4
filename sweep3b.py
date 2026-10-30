"""VOL_MIN fine sweep + interactions on d6eb227 base."""
from sweep3 import cfg, evaluate, show

if __name__ == "__main__":
    for v in (9000.0, 10000.0, 11000.0, 12000.0, 13000.0, 14000.0, 16000.0):
        show(f"vol{v/1000:.0f}k", evaluate(cfg(vol_min=v)))
    show("vol12k imth .03", evaluate(cfg(vol_min=12000.0, imth=0.03)))
    show("vol12k imth .05", evaluate(cfg(vol_min=12000.0, imth=0.05)))
    show("vol12k ha10", evaluate(cfg(vol_min=12000.0, ha=10)))
    show("vol12k ha8", evaluate(cfg(vol_min=12000.0, ha=8)))
    show("vol12k cd7", evaluate(cfg(vol_min=12000.0, cd=7)))
    show("vol12k hd8", evaluate(cfg(vol_min=12000.0, hd=8)))
