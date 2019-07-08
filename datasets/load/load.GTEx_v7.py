

import hail as hl
from hail import Table
from hail.utils import wrap_to_list


def import_gtf(path, reference_genome=None, skip_invalid_contigs=False) -> hl.Table:
    """Import a GTF file.

       The GTF file format is identical to the GFF version 2 file format,
       and so this function can be used to import GFF version 2 files as
       well.

       See https://www.ensembl.org/info/website/upload/gff.html for more
       details on the GTF/GFF2 file format.

       The :class:`.Table` returned by this function will be keyed by the
       ``interval`` row field and will include the following row fields:

       .. code-block:: text

           'source': str
           'feature': str
           'score': float64
           'strand': str
           'frame': int32
           'interval': interval<>

       There will also be corresponding fields for every tag found in the
       attribute field of the GTF file.

       Note
       ----

       This function will return an ``interval`` field of type :class:`.tinterval`
       constructed from the ``seqname``, ``start``, and ``end`` fields in the
       GTF file. This interval is inclusive of both the start and end positions
       in the GTF file. 

       If the ``reference_genome`` parameter is specified, the start and end
       points of the ``interval`` field will be of type :class:`.tlocus`.
       Otherwise, the start and end points of the ``interval`` field will be of
       type :class:`.tstruct` with fields ``seqname`` (type :class:`str`) and
       ``position`` (type :class:`.tint32`).

       Furthermore, if the ``reference_genome`` parameter is specified and
       ``skip_invalid_contigs`` is ``True``, this import function will skip
       lines in the GTF where ``seqname`` is not consistent with the reference
       genome specified.

       Example
       -------

       >>> ht = hl.experimental.import_gtf('data/test.gtf', 
                                           reference_genome='GRCh37',
                                           skip_invalid_contigs=True)
       >>> ht.describe()

       .. code-block:: text

           ----------------------------------------
           Global fields:
           None
           ----------------------------------------
           Row fields:
               'source': str
               'feature': str
               'score': float64
               'strand': str
               'frame': int32
               'gene_type': str
               'exon_id': str
               'havana_transcript': str
               'level': str
               'transcript_name': str
               'gene_status': str
               'gene_id': str
               'transcript_type': str
               'tag': str
               'transcript_status': str
               'gene_name': str
               'transcript_id': str
               'exon_number': str
               'havana_gene': str
               'interval': interval<locus<GRCh37>>
           ----------------------------------------
           Key: ['interval']
           ----------------------------------------

       Parameters
       ----------

       path : :obj:`str`
           File to import.
       reference_genome : :obj:`str` or :class:`.ReferenceGenome`, optional
           Reference genome to use.
       skip_invalid_contigs : :obj:`bool`
           If ``True`` and `reference_genome` is not ``None``, skip lines where
           ``seqname`` is not consistent with the reference genome.

       Returns
       -------
       :class:`.Table`
       """

    ht = hl.import_table(path,
                         comment='#',
                         no_header=True,
                         types={'f3': hl.tint,
                                'f4': hl.tint,
                                'f5': hl.tfloat,
                                'f7': hl.tint},
                         missing='.',
                         delimiter='\t')

    ht = ht.rename({'f0': 'seqname',
                    'f1': 'source',
                    'f2': 'feature',
                    'f3': 'start',
                    'f4': 'end',
                    'f5': 'score',
                    'f6': 'strand',
                    'f7': 'frame',
                    'f8': 'attribute'})

    ht = ht.annotate(attribute=hl.dict(
        hl.map(lambda x: (x.split(' ')[0],
                          x.split(' ')[1].replace('"', '').replace(';$', '')),
               ht['attribute'].split('; '))))

    attributes = ht.aggregate(hl.agg.explode(lambda x: hl.agg.collect_as_set(x), ht['attribute'].keys()))

    ht = ht.transmute(**{x: hl.or_missing(ht['attribute'].contains(x),
                                          ht['attribute'][x])
                         for x in attributes if x})

    if reference_genome:
        if reference_genome == 'GRCh37':
            ht = ht.annotate(seqname=ht['seqname'].replace('^chr', ''))
        else:
            ht = ht.annotate(seqname=hl.case()
                                       .when(ht['seqname'].startswith('HLA'), ht['seqname'])
                                       .when(ht['seqname'].startswith('chrHLA'), ht['seqname'].replace('^chr', ''))
                                       .when(ht['seqname'].startswith('chr'), ht['seqname'])
                                       .default('chr' + ht['seqname']))
        if skip_invalid_contigs:
            valid_contigs = hl.literal(set(hl.get_reference(reference_genome).contigs))
            ht = ht.filter(valid_contigs.contains(ht['seqname']))
        ht = ht.transmute(interval=hl.locus_interval(ht['seqname'],
                                                     ht['start'],
                                                     ht['end'],
                                                     includes_start=True,
                                                     includes_end=True,
                                                     reference_genome=reference_genome))
    else:
        ht = ht.transmute(interval=hl.interval(hl.struct(seqname=ht['seqname'], position=ht['start']),
                                               hl.struct(seqname=ht['seqname'], position=ht['end']),
                                               includes_start=True,
                                               includes_end=True))

    ht = ht.key_by('interval')

    return ht


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-b', required=True, choices=['GRCh37', 'GRCh38'], help='Reference genome build.')
parser.add_argument('-d', required=True, choices=['eqtl_associations', 'eqtl_covariates', 'genes', 'transcripts', 'exons', 'junctions'], help='GTEx dataset to load.')
args = parser.parse_args()

version = 'v7'
reference_genome = args.b
dataset = args.d

EXTRACT_BUCKET = 'gs://hail-datasets-extracted-data/'
LOAD_BUCKET = 'gs://hail-datasets/'

float_sample_attributes = [
    'SMRIN',
    'SME2MPRT',
    'SMNTRART',
    'SMMAPRT',
    'SMEXNCRT',
    'SM550NRM',
    'SMUNMPRT',
    'SM350NRM',
    'SMMNCPB',
    'SME1MMRT',
    'SMNTERRT',
    'SMMNCV',
    'SMGAPPCT',
    'SMNTRNRT',
    'SMMPUNRT',
    'SMEXPEFF',
    'SME2MMRT',
    'SMBSMMRT',
    'SME1PCTS',
    'SMRRNART',
    'SME1MPRT',
    'SMDPMPRT',
    'SME2PCTS']

int_sample_attributes = [
    'SMTSISCH',
    'SMATSSCR',
    'SMTSPAX',
    'SMCHMPRS',
    'SMNUMGPS',
    'SMGNSDTC',
    'SMRDLGTH',
    'SMSFLGTH',
    'SMESTLBS',
    'SMMPPD',
    'SMRRNANM',
    'SMVQCFL',
    'SMTRSCPT',
    'SMMPPDPR',
    'SMCGLGTH',
    'SMUNPDRD',
    'SMMPPDUN',
    'SME2ANTI',
    'SMALTALG',
    'SME2SNSE',
    'SMMFLGTH',
    'SMSPLTRD',
    'SME1ANTI',
    'SME1SNSE',
    'SMNUM5CD']

ht_subject_phenotypes = hl.import_table(EXTRACT_BUCKET + 'GTEx/v7/GTEx_subject_phenotypes.v7.tsv.bgz',
                                        key='SUBJID',
                                        missing='',
                                        types={'SEX': hl.tint, 'DTHHRDY': hl.tint})
ht_subject_phenotypes = ht_subject_phenotypes.rename({'SUBJID': 'subject_id',
                                                      'SEX': 'subject_sex',
                                                      'AGE': 'subject_age_group',
                                                      'DTHHRDY': 'subject_death_classification'})

ht_sample_attributes = hl.import_table(EXTRACT_BUCKET + 'GTEx/v7/GTEx_sample_attributes.v7.tsv.bgz', 
                                       key='SAMPID',
                                       missing='')
ht_sample_attributes = ht_sample_attributes.annotate(
    **{x: hl.float(ht_sample_attributes[x]) for x in float_sample_attributes})
ht_sample_attributes = ht_sample_attributes.annotate(
    **{x: hl.int(ht_sample_attributes[x].replace('.0$', '')) for x in int_sample_attributes})
ht_sample_attributes = ht_sample_attributes.rename({'SAMPID': 'sample_id',})
ht_sample_attributes = ht_sample_attributes.annotate(subject_id=hl.delimit(ht_sample_attributes['sample_id'].split('-')[:2], '-'))
ht_sample_attributes = ht_sample_attributes.annotate(**ht_subject_phenotypes[ht_sample_attributes.subject_id])

if dataset == 'genes':

    name = 'GTEx_gene_expression'

    ht_genes = import_gtf(path=EXTRACT_BUCKET + 'GTEx/v7/GTEx_genes.v7.GRCh37.gtf.bgz',
                          reference_genome='GRCh37')
    ht_genes = ht_genes.filter(ht_genes['feature'] == 'gene')
    ht_genes = ht_genes.key_by(ht_genes['gene_id'])
    ht_genes = ht_genes.select('interval',
                               'strand',
                               'gene_name',
                               'havana_gene',
                               'gene_type',
                               'gene_status',
                               'level',
                               'tag')
    ht_genes = ht_genes.rename({'interval': 'gene_interval'})
    ht_genes = ht_genes.distinct()

    mt_counts = hl.import_matrix_table(
        EXTRACT_BUCKET + 'GTEx/v7/GTEx_gene_read_counts.v7.GRCh37.tsv.bgz',
        row_fields={'Name': hl.tstr, 'Description': hl.tstr}, row_key='Name', missing=' ', entry_type=hl.tfloat)
    mt_counts = mt_counts.drop('Description')
    mt_counts = mt_counts.transmute_entries(read_count=hl.int(mt_counts['x']))
    mt_counts = mt_counts.rename({'col_id': 'sample_id', 'Name': 'gene_id'})

    mt_tpm = hl.import_matrix_table(
        EXTRACT_BUCKET + 'GTEx/v7/GTEx_gene_tpm.v7.GRCh37.tsv.bgz',
        row_fields={'Name': hl.tstr, 'Description': hl.tstr}, row_key='Name', missing=' ', entry_type=hl.tfloat)
    mt_tpm = mt_tpm.drop('Description')
    mt_tpm = mt_tpm.rename({'col_id': 'sample_id', 'Name': 'gene_id', 'x': 'TPM'})

    mt = mt_counts.annotate_entries(TPM=mt_tpm[mt_counts.gene_id, mt_counts.sample_id]['TPM'])
    mt = mt.annotate_rows(**ht_genes[mt.gene_id])
    mt = mt.annotate_cols(**ht_sample_attributes[mt.sample_id])

    if reference_genome == 'GRCh38':
        b37 = hl.get_reference('GRCh37')
        b37.add_liftover('gs://hail-common/references/grch37_to_grch38.over.chain.gz', 'GRCh38')
        mt = mt.annotate_rows(gene_interval=hl.liftover(mt['gene_interval'], 'GRCh38'))
        mt = mt.filter_rows(hl.is_defined(mt['gene_interval']))

    mt = mt.repartition(20)

elif dataset == 'transcripts':

    name = 'GTEx_transcript_expression'

    ht_transcripts = import_gtf(path=EXTRACT_BUCKET + 'GTEx/v7/GTEx_transcripts.v7.GRCh37.gtf.bgz',
                                reference_genome='GRCh37')
    ht_transcripts = ht_transcripts.filter(ht_transcripts['feature'] == 'transcript')
    ht_transcripts = ht_transcripts.select('transcript_id', 
                                           'strand',
                                           'transcript_name',
                                           'transcript_type',
                                           'transcript_status',
                                           'havana_transcript',
                                           'tag',
                                           'level',
                                           'ont',
                                           'source',
                                           'ccdsid',
                                           'gene_id',
                                           'gene_name',
                                           'gene_type',
                                           'gene_status',
                                           'havana_gene')
    ht_transcripts = ht_transcripts.key_by(ht_transcripts['transcript_id'])
    ht_transcripts = ht_transcripts.rename({'interval': 'transcript_interval'})
    ht_transcripts = ht_transcripts.distinct()

    mt_counts = hl.import_matrix_table(
        EXTRACT_BUCKET + 'GTEx/v7/GTEx_transcript_read_counts.v7.GRCh37.tsv.bgz',
        row_fields={'transcript_id': hl.tstr, 'gene_id': hl.tstr}, row_key='transcript_id', missing=' ', entry_type=hl.tfloat)
    mt_counts = mt_counts.drop('gene_id')
    mt_counts = mt_counts.transmute_entries(read_count=hl.int(mt_counts['x']))
    mt_counts = mt_counts.rename({'col_id': 'sample_id'})

    mt_tpm = hl.import_matrix_table(
        EXTRACT_BUCKET + 'GTEx/v7/GTEx_transcript_tpm.v7.GRCh37.tsv.bgz',
        row_fields={'transcript_id': hl.tstr, 'gene_id': hl.tstr}, row_key='transcript_id', missing=' ', entry_type=hl.tfloat)
    mt_tpm = mt_tpm.drop('gene_id')
    mt_tpm = mt_tpm.rename({'col_id': 'sample_id', 'x': 'TPM'})

    mt = mt_counts.annotate_entries(TPM=mt_tpm[mt_counts.transcript_id, mt_counts.sample_id]['TPM'])
    mt = mt.annotate_rows(**ht_transcripts[mt.transcript_id])
    mt = mt.annotate_cols(**ht_sample_attributes[mt.sample_id])

    if reference_genome == 'GRCh38':
        b37 = hl.get_reference('GRCh37')
        b37.add_liftover('gs://hail-common/references/grch37_to_grch38.over.chain.gz', 'GRCh38')
        mt = mt.annotate_rows(transcript_interval=hl.liftover(mt['transcript_interval'], 'GRCh38'))
        mt = mt.filter_rows(hl.is_defined(mt['transcript_interval']))

    mt = mt.repartition(80)


elif dataset == 'exons':

    name = 'GTEx_exon_expression'

    ht_exons = import_gtf(path=EXTRACT_BUCKET + 'GTEx/v7/GTEx_genes.v7.GRCh37.gtf.bgz',
                          reference_genome='GRCh37')
    ht_exons = ht_exons.filter(ht_exons['feature'] == 'exon')
    ht_exons = ht_exons.transmute(exon_number=hl.int(ht_exons['exon_number']))
    ht_exons = ht_exons.key_by(ht_exons['gene_id'], ht_exons['exon_number'])
    ht_exons = ht_exons.distinct()
    ht_exons.show()

    """
    ht_genes = ht_genes.select('interval',
                               'strand',
                               'gene_name',
                               'havana_gene',
                               'gene_type',
                               'gene_status',
                               'level',
                               'tag')
    ht_genes = ht_genes.rename({'interval': 'gene_interval'})
    ht_genes.describe()
    """
    
    mt = hl.import_matrix_table(
        EXTRACT_BUCKET + 'GTEx/v7/GTEx_exon_read_counts.v7.GRCh37.tsv.bgz',
        row_fields={'exon_id': hl.tstr}, row_key='exon_id', missing=' ', entry_type=hl.tfloat)
    mt = mt.transmute_entries(read_count=hl.int(mt['x']))
    mt = mt.rename({'col_id': 'sample_id'})
    mt.describe()
    """
    mt = mt.annotate_rows(gene_id=mt['exon_id'].split('_')[0])
    mt = mt.annotate_rows(**ht_genes[mt.gene_id])
    mt = mt.annotate_rows(**ht_exons[mt.exon_id])
    mt = mt.annotate_cols(**ht_sample_attributes[mt.sample_id])

    if reference_genome == 'GRCh38':
        b37 = hl.get_reference('GRCh37')
        b37.add_liftover('gs://hail-common/references/grch37_to_grch38.over.chain.gz', 'GRCh38')
        mt = mt.annotate_rows(interval=hl.liftover(mt['interval'], 'GRCh38'))
        mt = mt.filter_rows(hl.is_defined(mt['interval']))
    """

elif dataset == 'junctions':

    name = 'GTEx_junction_expression'

    ht_genes = import_gtf(path=EXTRACT_BUCKET + 'GTEx/v7/GTEx_genes.v7.GRCh37.gtf.bgz',
                          reference_genome='GRCh37')
    ht_genes = ht_genes.filter(ht_genes['feature'] == 'gene')
    ht_genes = ht_genes.key_by(ht_genes['gene_id'])
    ht_genes = ht_genes.select('interval',
                               'strand',
                               'gene_name',
                               'havana_gene',
                               'gene_type',
                               'gene_status',
                               'level',
                               'tag')
    ht_genes = ht_genes.rename({'interval': 'gene_interval'})
    ht_genes = ht_genes.distinct()

    mt = hl.import_matrix_table(
        EXTRACT_BUCKET + 'GTEx/v7/GTEx_junction_read_counts.v7.GRCh37.tsv.bgz',
        row_fields={'junction_id': hl.tstr, 'Description': hl.tstr}, missing=' ', entry_type=hl.tfloat)
    mt = mt.transmute_rows(chr_start_end=mt['junction_id'].split('_'))
    mt = mt.transmute_rows(junction_interval=hl.locus_interval(mt['chr_start_end'][0],
                                                               hl.int(mt['chr_start_end'][1]),
                                                               hl.int(mt['chr_start_end'][2]),
                                                               includes_start=True,
                                                               includes_end=True,
                                                               reference_genome='GRCh37'))
    mt = mt.key_rows_by(mt['junction_interval'])
    mt = mt.transmute_entries(read_count=hl.int(mt['x']))
    mt = mt.rename({'Description': 'gene_id', 'col_id': 'sample_id'})
    mt = mt.annotate_cols(**ht_sample_attributes[mt.sample_id])

    if reference_genome == 'GRCh38':
        b37 = hl.get_reference('GRCh37')
        b37.add_liftover('gs://hail-common/references/grch37_to_grch38.over.chain.gz', 'GRCh38')
        mt = mt.key_rows_by()
        mt = mt.annotate_rows(junction_interval=hl.liftover(mt['junction_interval'], 'GRCh38'))
        mt = mt.filter_rows(hl.is_defined(mt['junction_interval']))
        mt = mt.key_rows_by(mt['junction_interval'])

"""
n_rows, n_cols = mt.count()
n_partitions = mt.n_partitions()
mt = mt.annotate_globals(metadata=hl.struct(
    name=name,
    version=version,
    reference_genome=reference_genome,
    n_rows=n_rows,
    n_cols=n_cols,
    n_partitions=n_partitions))

mt.describe()
mt.write(LOAD_BUCKET + '{n}.{v}.{rg}.mt'.format(n=name, v=version, rg=reference_genome), overwrite=True)
"""