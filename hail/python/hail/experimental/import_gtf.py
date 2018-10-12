
import hail as hl
from hail.utils import wrap_to_list

def import_gtf(path, reference_genome=None, skip_invalid_intervals=False) -> Table:
    """Import a GTF file.

       The GTF file format is identical to the GFF version 2 file format,
       and so this function can be used to import GFF version 2 files as
       well.

       See https://www.ensembl.org/info/website/upload/gff.html for more
       details on the GTF/GFF2 file format.

       The :class:`.Table` returned by this function will be keyed by the
       ``interval`` row field and will include the following row fields:

       .. code-block:: text

           'interval': str
           'source': str
           'feature': str
           'score': float64
           'strand': str
           'frame': int32

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
       type :class:`.tstruct` with fields ``contig`` (type :class:`str`) and
       ``position`` (type :class:`.tint32`).

       Furthermore, if the ``reference_genome`` parameter is specified and
       ``skip_invalid_intervals`` is ``True``, this import function will skip
       lines in the GTF that are not consistent with the reference genome
       specified.

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
       reference_genome : :obj:`str` or :class:`.ReferenceGenome`, optional
           Reference genome to use.
       skip_invalid_intervals : :obj:`bool`
           If ``True`` and `reference_genome` is not ``None``, skip lines with
           intervals that are not consistent with the reference genome.

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

    attributes = ht.aggregate(hl.agg.collect_as_set(hl.agg.explode(ht['attribute'].keys())))
    #attributes = list(ht.aggregate(
    #    hl.set(hl.flatten(hl.agg.collect(ht['attribute'].keys())))))

    ht = ht.transmute(**{x: hl.or_missing(ht['attribute'].contains(x),
                                          ht['attribute'][x])
                         for x in attributes})

    if reference_genome:
        ht = ht.transmute(interval=hl.locus_interval(ht['seqname'],
                                                     ht['start'],
                                                     ht['end'],
                                                     includes_start=True,
                                                     includes_end=True))
    else:
        ht = ht.transmute(interval=hl.interval(hl.struct(seqname=ht['seqname'], position=ht['start']),
                                               hl.struct(seqname=ht['seqname'], position=ht['end'])),
                                               includes_start=True,
                                               includes_end=True)

    ht = ht.key_by('interval')

    return ht
