import gt_v1 as g
g.ev("BASE")
print("== Thu gate scope ==")
# need per-bucket Thu knobs: hack via existing params is limited; test cold+dis Thu skip via quick param
g.ev("thu all-buckets? (manual test below)")
