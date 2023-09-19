
import hail as hl
import argparse

raw_data_root = 'gs://hail-datasets-raw-data/DANN'
hail_data_root = 'gs://hail-datasets-hail-data'

parser = argparse.ArgumentParser()
parser.add_argument('-b', required=True, choices=['GRCh37', 'GRCh38'], help='Ensembl reference genome build.')
args = parser.parse_args()

name = 'DANN'
build = args.b

ht = hl.import_table(f'{raw_data_root}/DANN_GRCh37.tsv.bgz',
                     types={'position': hl.tint,
                            'DANN_score': hl.tfloat})

ht = ht.annotate(locus=hl.locus(ht['chromosome'], ht['position'], 'GRCh37'),
                 alleles=hl.array([ht['ref'], ht['alt']]))

if build == 'GRCh38':
    b37 = hl.get_reference('GRCh37')
    b37.add_liftover('gs://hail-common/references/grch37_to_grch38.over.chain.gz', 'GRCh38')
    ht = ht.annotate(locus=hl.liftover(ht['locus'], 'GRCh38'))
    ht = ht.filter(hl.is_defined(ht['locus']))

n_rows = ht.count()
n_partitions = ht.n_partitions()

ht = ht.key_by('locus', 'alleles')
ht = ht.rename({'DANN_score': 'score'})
ht = ht.select('score')

ht = ht.annotate_globals(metadata=hl.struct(name=name,
                                            version=hl.missing(hl.tstr),
                                            reference_genome=build,
                                            n_rows=n_rows,
                                            n_partitions=n_partitions))

ht.write(f'{hail_data_root}/DANN.{build}.ht', overwrite=True)
ht = hl.read_table(f'{hail_data_root}/DANN.{build}.ht')
ht.describe()
