import hail as hl

hl.init()

mt = hl.balding_nichols_model(3, 10, 10)

mt._force_count_rows()

rg = mt.locus.dtype.reference_genome
rg.add_liftover('gs://hail-common/references/grch37_to_grch38.over.chain.gz', 'GRCh38')
mt = mt.annotate_rows(locus2=hl.liftover(mt.locus, 'GRCh38'))

mt._force_count_rows()

hl.locus(contig='20', pos=1, reference_genome='GRCh37').show()
