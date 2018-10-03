
import argparse
import hail as hl

parser = argparse.ArgumentParser()
parser.add_argument('-d', required=True, choices=['scores', 'elements'], help='GERP++ dataset to load.')
parser.add_argument('-b', required=True, choices=['GRCh37', 'GRCh38'], help='Reference genome build to load.')
args = parser.parse_args()

hg19 = hl.ReferenceGenome.from_fasta_file('hg19', 
                                          'gs://hail-datasets/raw-data/assemblies/ucsc.hg19.fasta.gz', 
                                          'gs://hail-datasets/raw-data/assemblies/ucsc.hg19.fasta.fai')

if args.d == 'scores':
    name = 'GERP++_scores'
    ht = hl.import_table('gs://hail-datasets/raw-data/GERP++/GERP++_scores.hg19.tsv.bgz',
                         types={'position': hl.tint, 'N': hl.tfloat, 'S': hl.tfloat}, min_partitions=300)
    ht = ht.annotate(locus=hl.locus('chr' + ht['chromosome'].replace('MT', 'M'), ht['position'], 'hg19'))
    if args.b == 'GRCh37':
        hg19.add_liftover('gs://hail-datasets/raw-data/assemblies/hg19tob37.chain.gz', 'GRCh37')
        ht = ht.annotate(locus=hl.liftover(ht['locus'], 'GRCh37'))
    if args.b == 'GRCh38':
        hg19.add_liftover('gs://hail-datasets/raw-data/assemblies/hg19ToHg38.over.chain.gz', 'GRCh38')
        ht = ht.annotate(locus=hl.liftover(ht['locus'], 'GRCh38'))
    ht = ht.filter(hl.is_defined(ht['locus']))
    ht = ht.select('locus', 'N', 'S')
    ht = ht.key_by('locus')

if args.d == 'elements':
    name = 'GERP++_elements'
    ht = hl.import_table('gs://hail-datasets/raw-data/GERP++/GERP++_elements.hg19.tsv.bgz',
                         types={'start': hl.tint, 'end': hl.tint, 'S': hl.tfloat, 'p_value': hl.tfloat})
    ht = ht.annotate(interval=hl.interval(hl.locus(ht['chromosome'], ht['start'], 'hg19'),
                                          hl.locus(ht['chromosome'], ht['end'], 'hg19')))
    if args.b == 'GRCh37':
        hg19.add_liftover('gs://hail-datasets/raw-data/assemblies/hg19tob37.chain.gz', 'GRCh37')
        ht = ht.annotate(interval=hl.liftover(ht['interval'], 'GRCh37'))
    if args.b == 'GRCh38':
        hg19.add_liftover('gs://hail-datasets/raw-data/assemblies/hg19ToHg38.over.chain.gz', 'GRCh38')
        ht = ht.annotate(interval=hl.liftover(ht['interval'], 'GRCh38'))
    ht = ht.filter(hl.is_defined(ht['interval']))
    ht = ht.select('interval', 'S', 'p_value')
    ht = ht.key_by('interval')

n_rows = ht.count()
n_partitions = ht.n_partitions()
ht = ht.annotate_globals(name=name,
                         version=hl.null(hl.tstr),
                         reference_genome=args.b,
                         n_rows=n_rows,
                         n_partitions=n_partitions)
ht.describe()
ht.write('gs://hail-datasets/hail-data/{n}.{rg}.ht'.format(n=name, rg=args.b), overwrite=True)
