import hail as hl

mt = hl.import_vcf('gs://hail-1kg/1kg_coreexome.vcf.bgz')
mt._force_count_rows()
