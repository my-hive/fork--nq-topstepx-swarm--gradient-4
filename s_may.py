import harness, types
src=open('strategy.py').read()
A="        if sum(hist[-LOSS_K:]) < LOSS_TH:"
def run(rep,name):
    s=src
    for a,b in rep: assert a in s; s=s.replace(a,b,1)
    mod=types.ModuleType('m'); exec(compile(s,'s','exec'),mod.__dict__)
    harness.score_signals(mod.signals,name)
run([],"base")
# skip day entirely after prior session was a big loss (day_min proxy unavailable; use hist < -700 pts*ct)
run([(A,"""        if len(hist) and hist[-1] < -700:
            sz = 0""")],"skip after big-loss session")
run([(A,"""        if len(hist) and hist[-1] < -400:
            sz = 0""")],"skip after -400 session")
# quarter-size after loss instead of half
for f in [0.25,0.34,0.5]:
    run([("            sz = int(round(sz * 0.5))\n        if sz <= 0:","            sz = int(round(sz * %s))\n        if sz <= 0:"%f)],f"throttle {f}")
