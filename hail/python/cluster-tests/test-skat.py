import hail as hl

mt = hl.balding_nichols_model(3, 100, 100)
t = hl.skat(mt.locus, mt.ancestral_af, mt.pop, mt.GT.n_alt_alleles(), covariates=[1])
t.show()
