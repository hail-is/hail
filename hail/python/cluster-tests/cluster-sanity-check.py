import hail as hl

mt = hl.import_vcf('gs://hail-1kg/1kg_coreexome.vcf.bgz')
mt = mt.annotate_rows(x = 5)
mt._force_count_rows()

mt = hl.import_bgen('gs://hail-ci/example.8bits.bgen', entry_fields=['GT'])
mt._force_count_rows()
