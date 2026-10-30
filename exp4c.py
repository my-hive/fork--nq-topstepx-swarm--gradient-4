"""Exp: bracket sweeps (fade stop/tgt, big-onv brackets, flip brackets, flip size, exit, loss throttle)."""
import exp4 as e

if __name__ == "__main__":
    for bst in (25.0, 30.0, 35.0):
        for btgt in (70.0, 80.0, 90.0):
            if (bst, btgt) != (30.0, 80.0):
                e.show(f"big {bst:.0f}/{btgt:.0f}", e.evaluate(e.mk(bst=bst, btgt=btgt)))
    for fst in (40.0, 45.0, 50.0):
        for ftg in (64.0, 66.0, 68.0, 70.0):
            if (fst, ftg) != (45.0, 68.0):
                e.show(f"flip {fst:.0f}/{ftg:.0f}", e.evaluate(e.mk(fst=fst, ftg=ftg)))
    for fsa in (1, 2, 3):
        if fsa != 2:
            e.show(f"f_sz_add={fsa}", e.evaluate(e.mk(f_sz_add=fsa)))
    for ex, nm in ((14 * 3600, "14:00"), (14 * 3600 + 15 * 60, "14:15"), (14 * 3600 + 45 * 60, "14:45")):
        e.show(f"exit {nm}", e.evaluate(e.mk(exit_=ex)))
    for k in (1, 2, 3):
        for th in (-300.0, -500.0, -700.0, -900.0):
            if (k, th) != (2, -700.0):
                e.show(f"loss k={k} th={th:.0f}", e.evaluate(e.mk(loss_k=k, loss_th=th)))
