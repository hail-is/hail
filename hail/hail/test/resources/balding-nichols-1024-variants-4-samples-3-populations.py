import hail as hl

hl.set_global_seed(0)
mt = hl.balding_nichols_model(n_populations=3, n_variants=(1 << 10), n_samples=4)
mt = mt.key_cols_by(s='s' + hl.str(mt.sample_idx))
mt = mt.annotate_entries(GT=hl.or_missing(hl.rand_bool(0.99), mt.GT))
hl.export_plink(mt, 'balding-nichols-1024-variants-4-samples-3-populations', fam_id='f' + mt.s)
