
import hail as hl

ht = hl.import_table(
    'gs://hail-datasets-raw-data/1000_Genomes/1000_Genomes_phase3_sites_GRCh37.tsv.bgz',
    reference_genome='GRCh37')

ht.describe()
