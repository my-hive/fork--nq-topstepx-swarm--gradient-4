import harness, types
src=open('strategy.py').read()
# state 4 = done. Variant: after fade stop on moderate-agree (state->4 at 97), re-enter a 2nd fade at better price if price extends further by R1 pts
A="""                if pnl <= -stop:
                    if agree and abs(onv) <= 200:   # moderate agree: stand aside after stop
                        pos, state = 0, 4
                    else:                           # counters drift / big agree: real momentum
                        pos, entry, state = -side * (sz + 2), cc[k], 3"""
def run(rep,name):
    s=src
    for a,b in rep: assert a in s; s=s.replace(a,b,1)
    mod=types.ModuleType('m'); exec(compile(s,'s','exec'),mod.__dict__)
    harness.score_signals(mod.signals,name)
run([],"base")
for R in [10,20,30]:
    run([(A,"""                if pnl <= -stop:
                    if agree and abs(onv) <= 200:
                        pos, state, ext = 0, 2, 0.0
                    else:
                        pos, entry, state = -side * (sz + 2), cc[k], 3"""),
     ("            elif pos != 0 and state == 1:","""            elif state == 2:
                ext = max(ext, (cc[k] - entry) * -side * -1 * np.sign(side))
                ext = max(ext, (entry - cc[k]) * side if side>0 else 0)  # placeholder
            elif pos != 0 and state == 1:""")],f"reprobe {R}")
