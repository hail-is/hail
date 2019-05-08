from hail import *

hc = HailContext()
vds = hc.import_vcf('gs://hail-1kg/1kg_coreexome.vcf.bgz')
vds.count_variants()
