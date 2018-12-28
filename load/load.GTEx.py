

import hail as hl
from hail.utils import wrap_to_list


def import_gtf(path, key=None):
    """Import a GTF file.

       The GTF file format is identical to the GFF version 2 file format,
       and so this function can be used to import GFF version 2 files as
       well.

       See https://www.ensembl.org/info/website/upload/gff.html for more
       details on the GTF/GFF2 file format.

       The :class:`.Table` returned by this function will include the following
       row fields:

       .. code-block:: text

           'seqname': str
           'source': str
           'feature': str
           'start': int32
           'end': int32
           'score': float64
           'strand': str
           'frame': int32

       There will also be corresponding fields for every tag found in the
       attribute field of the GTF file.

       .. note::

           The "end" field in the table will be incremented by 1 in
           comparison to the value found in the GTF file, as the end
           coordinate in a GTF file is inclusive while the end
           coordinate in Hail is exclusive.

       Example
       -------

       >>> ht = hl.experimental.import_gtf('data/test.gtf', key='gene_id')
       >>> ht.describe()

       .. code-block:: text

           ----------------------------------------
           Global fields:
           None
           ----------------------------------------
           Row fields:
               'seqname': str
               'source': str
               'feature': str
               'start': int32
               'end': int32
               'score': float64
               'strand': str
               'frame': int32
               'havana_gene': str
               'exon_id': str
               'havana_transcript': str
               'transcript_name': str
               'gene_type': str
               'tag': str
               'transcript_status': str
               'exon_number': str
               'level': str
               'transcript_id': str
               'transcript_type': str
               'gene_id': str
               'gene_name': str
               'gene_status': str
           ----------------------------------------
           Key: ['gene_id']
           ----------------------------------------

       Parameters
       ----------

       path : :obj:`str`
           File to import.
       key : :obj:`str` or :obj:`list` of :obj:`str`
           Key field(s). Can be tag name(s) found in the attribute field
           of the GTF file.

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

    ht = ht.annotate(end=ht['end'] + 1)
    ht = ht.annotate(attribute=hl.dict(
        hl.map(lambda x: (x.split(' ')[0],
                          x.split(' ')[1].replace('"', '').replace(';$', '')),
               ht['attribute'].split('; '))))

    attributes = list(ht.aggregate(
        hl.set(hl.flatten(hl.agg.collect(ht['attribute'].keys())))))

    ht = ht.annotate(**{x: hl.or_missing(ht['attribute'].contains(x),
                                         ht['attribute'][x])
                        for x in attributes})

    ht = ht.drop(ht['attribute'])

    if key:
        key = wrap_to_list(key)
        ht = ht.key_by(*key)

    return ht


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-v', required=True, help='GTEx version.')
parser.add_argument('-b', required=True, choices=['GRCh37', 'GRCh38'], help='Ensembl reference genome build.')
parser.add_argument('-d', required=True, choices=['eqtl_associations', 'gene', 'transcript', 'exon', 'junction', ], help='Ensembl sequence dataset to load.')
args = parser.parse_args()

version = 'v{:}'.format(args.v)
reference_genome = args.b
dataset = args.d

if dataset == 'gene':

    ht_samples = hl.import_table('gs://hail-datasets/raw-data/gtex/v7/annotations/GTEx_v7_Annotations.SampleAttributesDS.txt', 
                                 key='SAMPID',
                                 missing='')

    float_cols = ['SMRIN',
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

    int_cols = ['SMTSISCH',
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

    ht_samples = ht_samples.annotate(**{x: hl.float(ht_samples[x]) for x in float_cols})
    ht_samples = ht_samples.annotate(**{x: hl.int(ht_samples[x].replace('.0$', '')) for x in int_cols})

    ht = ht.filter(ht.feature_type == 'gene')
    ht = ht.annotate(interval=hl.interval(hl.locus(ht['contig'], ht['start'], 'GRCh37'), hl.locus(ht['contig'], ht['end'] + 1, 'GRCh37')))
    ht = ht.annotate(attributes=hl.dict(hl.map(lambda x: (x.split(' ')[0], x.split(' ')[1].replace('"', '').replace(';$', '')), ht['attributes'].split('; '))))
    attribute_cols = list(ht.aggregate(hl.set(hl.flatten(hl.agg.collect(ht.attributes.keys())))))
    ht = ht.annotate(**{x: hl.or_missing(ht_genes.attributes.contains(x), ht_genes.attributes[x]) for x in attribute_cols})
    ht = ht.select(*(['gene_id', 'interval', 'gene_type', 'strand', 'annotation_source', 'havana_gene', 'gene_status', 'tag']))
    ht = ht.rename({'havana_gene': 'havana_gene_id'})
    ht = ht.key_by(ht_genes.gene_id)

"""

