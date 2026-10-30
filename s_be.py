import harness, types
src=open('strategy.py').read()
A="""                pnl = (cc[k] - entry) * np.sign(pos)
                if pnl <= -stop:"""
def run(rep,name):
    s=src
    for a,b in rep: assert a in s; s=s.replace(a,b,1)
    mod=types.ModuleType('m'); exec(compile(s,'s','exec'),mod.__dict__)
    harness.score_signals(mod.signals,name)
for be in [30,40,50,60]:
  for lk in [-10,0,10,20]:
    run([(A,"""                pnl = (cc[k] - entry) * np.sign(pos)
                mfe = max(mfe, pnl)
                if mfe >= %d and pnl <= %d:
                    pos, state = 0, 4
                elif pnl <= -stop:"""%(be,lk)),("pos, entry, state = side * sz, cc[k], 1","pos, entry, state, mfe = side * sz, cc[k], 1, 0.0")],f"be{be} lock{lk}")
