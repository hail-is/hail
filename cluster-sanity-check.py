import hail as hl

mt = hl.import_vcf('gs://hail-1kg/raw/*.vcf.bgz')
mt._force_count_rows()
