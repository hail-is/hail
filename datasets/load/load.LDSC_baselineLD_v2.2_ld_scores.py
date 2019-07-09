
import hail as hl

ht = hl.import_table('gs://hail-datasets-raw-data/LDSC/baselineLD_v2.2_ld_scores.tsv.bgz',
                     impute=True, types={'CHR': hl.tstr})
ht = ht.annotate(locus=hl.locus(ht.CHR, hl.int(ht.BP), 'GRCh37'))
ht = ht.drop(ht.CHR, ht.BP)
ht = ht.key_by(ht.locus)
ht.write('gs://hail-datasets-hail-data/LDSC_baselineLD_v2.2_ld_scores.ht', overwrite=True)
ht.describe()
