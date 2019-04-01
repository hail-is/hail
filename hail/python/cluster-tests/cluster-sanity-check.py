import hail as hl
from hail.linalg import BlockMatrix

mt = hl.import_vcf('gs://hail-1kg/1kg_coreexome.vcf.bgz')
mt = mt.annotate_rows(x = 5)
mt._force_count_rows()

mt = hl.import_bgen('gs://hail-ci/example.8bits.bgen', entry_fields=['GT'])
mt._force_count_rows()

bm = BlockMatrix.random(10, 11)
bm.to_numpy(_force_blocking=True)
bm.to_numpy()
