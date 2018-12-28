
import hail as hl
import argparse

EXTRACT_BUCKET = 'gs://hail-datasets-extracted-data/'
HAIL_BUCKET = 'gs://hail-datasets/'

parser = argparse.ArgumentParser()
parser.add_argument('-b', required=True, choices=['GRCh37', 'GRCh38'], help='Ensembl reference genome build.')
args = parser.parse_args()

name = 'DANN'
reference_genome = args.b

ht = hl.import_table(EXTRACT_BUCKET + 'DANN/DANN.GRCh37.tsv.bgz',
                     types={'position': hl.tint,
                            'DANN_score': hl.tfloat})
ht = ht.annotate(locus=hl.locus(ht['chromosome'], ht['position'], 'GRCh37'),
                 alleles=hl.array([ht['ref'], ht['alt']]))

ht = ht.drop('chromosome', 'position', 'ref', 'alt')
ht = ht.rename({'DANN_score': 'score'})

if reference_genome == 'GRCh38':
    b37 = hl.get_reference('GRCh37')
    b37.add_liftover('gs://hail-common/references/grch37_to_grch38.over.chain.gz', 'GRCh38')
    ht = ht.annotate(locus=hl.liftover(ht['locus'], 'GRCh38'))

ht = ht.select('locus', 'alleles', 'score')
ht = ht.key_by('locus', 'alleles')

n_rows = ht.count()
n_partitions = ht.n_partitions()
ht = ht.annotate_globals(metadata=hl.struct(name=name,
                                            version=hl.null(hl.tstr),
                                            reference_genome=reference_genome,
                                            n_rows=n_rows,
                                            n_partitions=n_partitions))

ht.describe()
ht.write(HAIL_BUCKET + '{n}.{rg}.ht'.format(n=name, rg=args.b), overwrite=True)
