
import hail as hl

contig_recoding = {str(i): 'chr' + str(i) for i in range(1, 23)}
contig_recoding.update({'X': 'chrX', 'Y': 'chrY', 'MT': 'chrM'})

mt = hl.import_vcf('gs://hail-datasets/raw-data/dbsnp/dbsnp_build151.GRCh38.vcf.bgz',
                   reference_genome='GRCh38',
                   contig_recoding=contig_recoding)

mt_split = hl.split_multi(mt)
mt_split = mt_split.annotate_rows(info=mt_split['info'].annotate(CAF=mt_split['info']['CAF'][mt_split['a_index'] - 1],
                                                                 TOPMED=mt_split['info']['TOPMED'][mt_split['a_index'] - 1]))
mt_split = mt_split.drop('old_locus', 'old_alleles')

ht = mt_split.rows()
ht = ht.repartition(500)

ht.describe()
ht.write('gs://hail-datasets/hail-data/dbsnp_build151.GRCh38.ht', overwrite=True)
