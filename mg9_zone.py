from mg8_orth import run
for hu in (225, 250, 275, 300, 325, 350, 400):
    run(f"up {hu} dn 225", hi_up=float(hu), hi_dn=225.0)
for hd in (200, 225, 250, 275):
    run(f"up 300 dn {hd}", hi_up=300.0, hi_dn=float(hd))
for lu in (45, 50, 55, 60):
    run(f"up lo{lu} 300 dn 225", lo_up=float(lu), hi_up=300.0)
