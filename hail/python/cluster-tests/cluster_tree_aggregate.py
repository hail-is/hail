import hail as hl
S = 500
V = 2000
mt = hl.balding_nichols_model(1, S, V, 500)
mt = mt.annotate_cols(n_called = hl.agg.filter(hl.is_defined(mt.GT), hl.agg.count()))
mt = mt.filter_cols(mt.n_called > 0).count()
