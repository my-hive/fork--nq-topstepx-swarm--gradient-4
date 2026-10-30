"""Exp: combos of marginal winners (big 30/90, flip 40/68) + robustness."""
import exp4 as e

if __name__ == "__main__":
    combos = [
        dict(btgt=90.0),
        dict(fst=40.0),
        dict(btgt=90.0, fst=40.0),
        dict(btgt=90.0, fst=40.0, ftg=70.0),
        dict(btgt=90.0, fst=42.0),
        dict(btgt=85.0, fst=40.0),
        dict(btgt=90.0, fst=38.0),
        dict(btgt=95.0, fst=40.0),
    ]
    for c in combos:
        e.robust("+".join(f"{k}={v}" for k, v in c.items()), **c)
