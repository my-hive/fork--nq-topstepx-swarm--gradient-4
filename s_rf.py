"""Variant: allow ONE re-flip after flip-leg stop (state 5), on big-agree days only."""
import harness, types

src = open('strategy.py').read()
A = """            elif pos != 0 and state == 3:
                pnl = (cc[k] - entry) * np.sign(pos)
                if pnl <= -F_STOP_PTS or pnl >= F_TGT_PTS:
                    pos, state = 0, 4"""


def run(name, big_only=True, szd=0, fstop=45.0, ftgt=68.0):
    cond = "agree and abs(onv) > 200" if big_only else "True"
    s = src.replace(A, """            elif pos != 0 and state == 3:
                pnl = (cc[k] - entry) * np.sign(pos)
                if pnl >= F_TGT_PTS:
                    pos, state = 0, 4
                elif pnl <= -F_STOP_PTS:
                    if %s:
                        pos, entry, state = -side * (sz + 2 + %d), cc[k], 5
                    else:
                        pos, state = 0, 4
            elif pos != 0 and state == 5:
                pnl = (cc[k] - entry) * np.sign(pos)
                if pnl <= -%s or pnl >= %s:
                    pos, state = 0, 4""" % (cond, szd, fstop, ftgt), 1)
    mod = types.ModuleType('m'); exec(compile(s, 's', 'exec'), mod.__dict__)
    harness.score_signals(mod.signals, name)


if __name__ == "__main__":
    run("base(no reflip)")
    for szd in (0, 2):
        for fs, ft in ((45.0, 68.0), (60.0, 100.0)):
            run(f"reflip big szd{szd} {fs}/{ft}", True, szd, fs, ft)
