
import hail as hl

ht = hl.import_table('gs://hail-datasets-raw-data/LDSC/baselineLD_v2.2/annotations.GRCh37.tsv.bgz',
                     impute=True, types={'CHR': hl.tstr})

ht = ht.annotate_globals(metadata=hl.struct(
    name='LDSC_baselineLD_v2.2_annotations',
    reference_genome='GRCh37',
    n_rows=ht.count(),
    n_partitions=ht.n_partitions()))
ht = ht.annotate(locus=hl.locus(ht.CHR, hl.int(ht.BP), 'GRCh37'))
ht = ht.drop(ht.CHR, ht.BP)
ht = ht.key_by(ht.locus)

ht.describe()
ht.write('gs://hail-datasets-hail-data/LDSC_baselineLD_v2.2_annotations.GRCh37.ht', overwrite=True)
