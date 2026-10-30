import gt_v1 as g
g.ev("BASE 0.8904")
print("== their axes on my base ==")
g.ev("ha 10", sz_hot_agree=10)
g.ev("big tgt 70", big_tgt=70.0)
g.ev("ha10 + big70", sz_hot_agree=10, big_tgt=70.0)
g.ev("global flip add 2", flip_sz_add=2)
g.ev("ha10+big70+flip2", sz_hot_agree=10, big_tgt=70.0, flip_sz_add=2)
