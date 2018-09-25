
import hail as hl

mt = hl.import_vcf('gs://hail-datasets/raw-data/dbsnp/dbsnp_build151.GRCh37.vcf.bgz', reference_genome='GRCh37')

mt_split = hl.split_multi(mt)
mt_split = mt_split.annotate_rows(info=mt_split['info'].annotate(CAF=mt_split['info']['CAF'][mt_split['a_index'] - 1],
                                                                 TOPMED=mt_split['info']['TOPMED'][mt_split['a_index'] - 1]))
mt_split = mt_split.drop('old_locus', 'old_alleles')

ht = mt_split.rows()
ht = ht.repartition(500)

ht.describe()
ht.write('gs://hail-datasets/hail-data/dbsnp_build151.GRCh37.ht', overwrite=True)
