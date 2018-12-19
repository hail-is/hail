
import hail as hl


def import_gtf(path, reference_genome=None, skip_invalid_contigs=False, min_partitions=None) -> hl.Table:
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
       ...                                 reference_genome='GRCh37',
       ...                                 skip_invalid_contigs=True)

       >>> ht.describe()  # doctest: +NOTEST
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
       min_partitions : :obj:`int` or :obj:`None`
           Minimum number of partitions (passed to import_table).

       Returns
       -------
       :class:`.Table`
       """

    ht = hl.import_table(path,
                         min_partitions=min_partitions,
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
