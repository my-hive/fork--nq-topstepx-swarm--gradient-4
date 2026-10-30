from gt_pm import run_pm

print("== smaller / tighter fade ==")
for sz, st, tg in ((2,15,30),(2,20,40),(3,20,30),(2,25,50)):
    run_pm(f"pm 12->13 mm30 sz{sz} {st}/{tg}", 12, 13, 30, sz, float(st), float(tg))
print("== momentum (sz<0 = with move) ==")
for sz, st, tg in ((-3,20,40),(-3,25,50),(-4,20,40),(-2,15,30)):
    run_pm(f"pm MOM 12->13 mm30 sz{-sz} {st}/{tg}", 12, 13, 30, sz, float(st), float(tg))
print("== momentum bigger minmove ==")
for sz, st, tg in ((-3,25,60),(-4,30,60),(-3,30,60)):
    run_pm(f"pm MOM 12->13 mm50 sz{-sz} {st}/{tg}", 12, 13, 50, sz, float(st), float(tg))
