
import argparse
import hail as hl

parser = argparse.ArgumentParser()
parser.add_argument('-r', required=True, help='Ensembl release version.')
parser.add_argument('-b', required=True, choices=['GRCh37', 'GRCh38'], help='Ensembl reference genome build.')
parser.add_argument('-d', required=True, choices=['dna', 'lcr', 'cdna', 'cds', 'ncrna', 'pep'], help='Ensembl sequence dataset to load.')
args = parser.parse_args()

version = 'release_{:}'.format(args.r)
reference_genome = args.b

if args.d == 'dna':
    name = 'Ensembl_reference_genome_sequence'
    ht = hl.import_table('gs://hail-datasets-extracted-data/Ensembl/{n}.{v}.{rg}.tsv.bgz'.format(n=name, v=version, rg=reference_genome),
                         types={'position': hl.tint})
    if reference_genome == 'GRCh38':
        ht = ht.annotate(chromosome='chr' + ht['chromosome'].replace('MT', 'M'))
    ht = ht.annotate(locus=hl.locus(ht['chromosome'], ht['position'], reference_genome))
    ht = ht.select('locus', 'reference_allele')
    ht = ht.key_by('locus')

if args.d == 'lcr':
    name = 'Ensembl_low_complexity_regions'
    ht = hl.import_table('gs://hail-datasets-extracted-data/Ensembl/{n}.{v}.{rg}.tsv.bgz'.format(n=name, v=version, rg=reference_genome),
                         types={'start': hl.tint, 'end': hl.tint})
    if reference_genome == 'GRCh38':
        ht = ht.annotate(chromosome='chr' + ht['chromosome'].replace('MT', 'M'))
    ht = ht.annotate(interval=hl.interval(hl.locus(ht['chromosome'], ht['start'], reference_genome),
                                          hl.locus(ht['chromosome'], ht['end'], reference_genome)))
    ht = ht.select('interval')
    ht = ht.key_by('interval')

if args.d in set(['cdna', 'cds', 'ncrna']):
    if args.d == 'cdna':
        name = 'Ensembl_cDNA_regions'
    elif args.d == 'cds':
        name = 'Ensembl_CDS_regions'
    else:
        name = 'Ensembl_ncRNA_regions'
    ht = hl.import_table('gs://hail-datasets-extracted-data/Ensembl/Ensembl_{0}_regions.{1}.{2}.tsv.bgz'.format(args.d, version, reference_genome),
                         types={'start': hl.tint, 'end': hl.tint})
    ht = ht.filter(hl.set([str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']).contains(ht['chromosome']))
    if reference_genome == 'GRCh38':
        ht = ht.annotate(chromosome='chr' + ht['chromosome'].replace('MT', 'M'))
    ht = ht.annotate(interval=hl.interval(hl.locus(ht['chromosome'], ht['start'], reference_genome),
                                          hl.locus(ht['chromosome'], ht['end'], reference_genome)))
    ht = ht.select('interval', 'strand', 'seqtype', 'status', 'transcript_id', 'transcript_biotype', 'gene_id', 'gene_biotype')
    ht = ht.key_by('interval')

if args.d == 'pep':
    name = 'Ensembl_peptide_sequences'
    ht = hl.import_table('gs://hail-datasets-extracted-data/Ensembl/{n}.{v}.{rg}.tsv.bgz'.format(n=name, v=version, rg=reference_genome),
                         types={'start': hl.tint, 'end': hl.tint})
    ht = ht.filter(hl.set([str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']).contains(ht['chromosome']))
    if reference_genome == 'GRCh38':
        ht = ht.annotate(chromosome='chr' + ht['chromosome'].replace('MT', 'M'))
    ht = ht.annotate(interval=hl.interval(hl.locus(ht['chromosome'], ht['start'], reference_genome),
                                          hl.locus(ht['chromosome'], ht['end'], reference_genome)))
    ht = ht.select('interval', 'strand', 'status', 'transcript_id', 'transcript_biotype', 'gene_id', 'gene_biotype', 'peptide_sequence')
    ht = ht.key_by('interval')

n_rows = ht.count()
n_partitions = ht.n_partitions()

ht = ht.annotate_globals(metadata=hl.struct(name=name,
                                            version=version,
                                            reference_genome=reference_genome,
                                            n_rows=n_rows,
                                            n_partitions=n_partitions))
ht.describe()
ht.write('gs://hail-datasets/{n}.{v}.{rg}.ht'.format(n=name, v=version, rg=reference_genome), overwrite=True)
