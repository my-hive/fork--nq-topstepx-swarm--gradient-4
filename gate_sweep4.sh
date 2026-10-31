#!/bin/bash
# usage: bash gate_sweep4.sh
set -e
cp strategy.py strategy_backup4.py
declare -A pats
# 1) remove retrace gate
python3 - << 'PY'
s = open("strategy.py").read()
a = '''        if hot and not agree and sz > 0:            # hot disagree: require retrace
            ext = hi[idx][where].max() if mv > 0 else lo[idx][where].min()
            if abs(ext - cc[where[-1]]) < RETR_MIN:
                sz = 0  # push still at its extreme: momentum, don't fade
'''
assert a in s
open("strategy.py","w").write(s.replace(a, ""))
PY
python3 exact4.py 2>/dev/null | head -1 | sed 's/^/NO_RETRACE: /'
cp strategy_backup4.py strategy.py
# 2) remove news gate
python3 - << 'PY'
s = open("strategy.py").read()
a = '''        if hot and not agree and abs(onv) < 25 and abs(mv) > 70:
            sz = 0  # flat overnight + violent open push: news momentum, don't fade
'''
assert a in s
open("strategy.py","w").write(s.replace(a, ""))
PY
python3 exact4.py 2>/dev/null | head -1 | sed 's/^/NO_NEWS: /'
cp strategy_backup4.py strategy.py
# 3) remove onrng gate
python3 - << 'PY'
s = open("strategy.py").read()
a = '''        if hot and not agree and abs(onv) > 200 and hi[idx][:where[0]].max() - lo[idx][:where[0]].min() < ON_RNG_MIN:
            sz = 0  # big one-way drift in tight overnight range: trend day, don't fade
'''
assert a in s
open("strategy.py","w").write(s.replace(a, ""))
PY
python3 exact4.py 2>/dev/null | head -1 | sed 's/^/NO_ONRNG: /'
cp strategy_backup4.py strategy.py
# 4) remove thursday gate
python3 - << 'PY'
s = open("strategy.py").read()
a = '''        if hot and agree and dow == 3:              # Thursday hot+agree: claims-day, skip
            sz = 0
'''
assert a in s
open("strategy.py","w").write(s.replace(a, ""))
PY
python3 exact4.py 2>/dev/null | head -1 | sed 's/^/NO_THURSDAY: /'
cp strategy_backup4.py strategy.py
# 5) remove Tuesday cold-dis flip veto
python3 - << 'PY'
s = open("strategy.py").read()
a = '''            (dow == 1 and not hot)
            or '''
assert a in s
open("strategy.py","w").write(s.replace(a, ""))
PY
python3 exact4.py 2>/dev/null | head -1 | sed 's/^/NO_TUESVETO: /'
cp strategy_backup4.py strategy.py
# 6) remove Friday small-ON flip veto
python3 - << 'PY'
s = open("strategy.py").read()
a = '''            or (dow == 4 and abs(onv) < 50)
'''
assert a in s
open("strategy.py","w").write(s.replace(a, ""))
PY
python3 exact4.py 2>/dev/null | head -1 | sed 's/^/NO_FRIVETO: /'
cp strategy_backup4.py strategy.py
rm strategy_backup4.py
