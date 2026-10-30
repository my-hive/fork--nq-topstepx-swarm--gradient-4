"""Variant: after moderate-agree fade stop (stand-aside), re-enter second fade if price
extends R pts beyond stop price, half size. Bounded to moderate-agree days."""
import harness, types

src = open('strategy.py').read()
A = """                if pnl <= -stop:
                    if agree and abs(onv) <= 200:   # moderate agree: stand aside after stop
                        pos, state = 0, 4
                    else:                           # counters drift / big agree: real momentum
                        pos, entry, state = -side * (sz + 2), cc[k], 3"""


def run(name, R=20.0, SZF=0.5, TS=40.0, TT=80.0):
    s = src.replace(A, """                if pnl <= -stop:
                    if agree and abs(onv) <= 200:   # moderate agree: stand aside, maybe re-enter
                        pos, state, sprice = 0, 2, cc[k]
                    else:
                        pos, entry, state = -side * (sz + 2), cc[k], 3""", 1)
    s = s.replace("""            elif pos != 0 and state == 3:""", """            elif state == 2:
                if (cc[k] - sprice) * -side >= %s:   # extended R pts past stop price
                    pos, entry, state = side * max(1, int(round(sz * %s))), cc[k], 5
            elif pos != 0 and state == 5:
                pnl = (cc[k] - entry) * np.sign(pos)
                if pnl <= -%s or pnl >= %s:
                    pos, state = 0, 4
            elif pos != 0 and state == 3:""" % (R, SZF, TS, TT), 1)
    mod = types.ModuleType('m'); exec(compile(s, 's', 'exec'), mod.__dict__)
    harness.score_signals(mod.signals, name)


if __name__ == "__main__":
    run("base")
    for R in (15.0, 25.0, 40.0):
        for SZF in (0.5, 1.0):
            run(f"reenter R{R} szx{SZF}", R, SZF)
