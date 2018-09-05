import hail as hl

mt = hl.import_vcf('gs://hail-1kg/1kg_coreexome.vcf.bgz')
mt = mt.annotate_rows(x = 5)
mt._force_count_rows()
