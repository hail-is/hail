from hail.typecheck import *
from hail.utils.java import Env, handle_py4j, joption, FatalError, jindexed_seq_args, jset_args
from hail.utils import wrap_to_list
from hail.api2 import Table, MatrixTable
from hail.expr.types import *
from hail.expr.expression import analyze, expr_any
from hail.genetics import GenomeReference
from hail.methods.misc import require_biallelic


@handle_py4j
@typecheck(table=Table,
           address=strlike,
           keyspace=strlike,
           table_name=strlike,
           block_size=integral,
           rate=integral)
def export_cassandra(table, address, keyspace, table_name, block_size=100, rate=1000):
    """Export to Cassandra.

    Warning
    -------
    :func:`export_cassandra` is EXPERIMENTAL.
    """

    table._jkt.exportCassandra(address, keyspace, table_name, block_size, rate)


@handle_py4j
@typecheck(dataset=MatrixTable,
           output=strlike,
           precision=integral)
def export_gen(dataset, output, precision=4):
    """Export variant dataset as GEN and SAMPLE files.

    .. include:: ../_templates/req_tvariant.rst

    .. include:: ../_templates/req_biallelic.rst

    Examples
    --------
    Import genotype probability data, filter variants based on INFO score, and
    export data to a GEN and SAMPLE file:
    
    >>> ds = methods.import_gen('data/example.gen', sample_file='data/example.sample')
    >>> ds = ds.filter_rows(agg.info_score(ds.GP).score >= 0.9) # doctest: +SKIP
    >>> methods.export_gen(ds, 'output/infoscore_filtered')

    Notes
    -----
    Writes out the dataset to a GEN and SAMPLE fileset in the
    `Oxford spec <http://www.stats.ox.ac.uk/%7Emarchini/software/gwas/file_format.html>`__.

    This method requires a `GP` (genotype probabilities) entry field of type
    ``Array[Float64]``. The values at indices 0, 1, and 2 are exported as the
    probabilities of homozygous reference, heterozygous, and homozygous variant,
    respectively. Missing `GP` values are exported as ``0 0 0``.

    The first six columns of the GEN file are as follows:

    - chromosome (`v.contig`)
    - variant ID (`varid` if defined, else Contig:Position:Ref:Alt)
    - rsID (`rsid` if defined, else ``.``)
    - position (`v.start`)
    - reference allele (`v.ref`)
    - alternate allele (`v.alt`)

    The SAMPLE file has three columns:

    - ID_1 and ID_2 are identical and set to the sample ID (`s`).
    - The third column (``missing``) is set to 0 for all samples.

    Parameters
    ----------
    dataset : :class:`.MatrixTable`
        Dataset with entry field `GP` of type Array[TFloat64].
    output : :obj:`str`
        Filename root for output GEN and SAMPLE files.
    precision : :obj:`int`
        Number of digits to write after the decimal point.
    """

    dataset = require_biallelic(dataset, 'export_gen')
    try:
        gp = dataset['GP']
        if gp.dtype != TArray(TFloat64()) or gp._indices != dataset._entry_indices:
            raise KeyError
    except KeyError:
        raise FatalError("export_gen: no entry field 'GP' of type Array[Float64]")

    dataset = require_biallelic(dataset, 'export_plink')

    Env.hail().io.gen.ExportGen.apply(dataset._jvds, output, precision)


@handle_py4j
@typecheck(dataset=MatrixTable,
           output=strlike,
           fam_args=expr_any)
def export_plink(dataset, output, **fam_args):
    """Export variant dataset as
    `PLINK2 <https://www.cog-genomics.org/plink2/formats>`__
    BED, BIM and FAM files.

    .. include:: ../_templates/req_tvariant.rst
    
    .. include:: ../_templates/req_tstring.rst

    .. include:: ../_templates/req_biallelic.rst

    Examples
    --------
    Import data from a VCF file, split multi-allelic variants, and export to
    PLINK files with the FAM file individual ID set to the sample ID:

    >>> ds = methods.split_multi_hts(dataset)
    >>> methods.export_plink(ds, 'output/example', id = ds.s)

    Notes
    -----
    `fam_args` may be used to set the fields in the output
    `FAM file <https://www.cog-genomics.org/plink2/formats#fam>`__
    via expressions with column and global fields in scope:

    - ``fam_id``: :class:`.TString` for the family ID
    - ``id``: :class:`.TString` for the individual (proband) ID
    - ``mat_id``: :class:`.TString` for the maternal ID
    - ``pat_id``: :class:`.TString` for the paternal ID
    - ``is_female``: :class:`.TBoolean` for the proband sex
    - ``is_case``: :class:`.TBoolean` or `quant_pheno`: :class:`.TFloat64` for the
       phenotype

    If no assignment is given, the corresponding PLINK missing value is written:
    ``0`` for IDs and sex, ``NA`` for phenotype. Only one of ``is_case`` or
    ``quant_pheno`` can be assigned. For Boolean expressions, true and false are
    output as ``2`` and ``1``, respectively (i.e., female and case are ``2``).

    The BIM file ID field has the form ``chr:pos:ref:alt`` with values given by
    `v.contig`, `v.start`, `v.ref`, and `v.alt`.

    On an imported VCF, the example above will behave similarly to the PLINK
    conversion command

    .. code-block:: text

        plink --vcf /path/to/file.vcf --make-bed --out sample --const-fid --keep-allele-order

    except that:

    - Variants that result from splitting a multi-allelic variant may be
      re-ordered relative to the BIM and BED files.
    - PLINK uses the rsID for the BIM file ID.

    Parameters
    ----------
    dataset : :class:`.MatrixTable`
        Dataset.
    output : :obj:`str`
        Filename root for output BED, BIM, and FAM files.
    fam_args : varargs of :class:`hail.expr.expression.Expression`
        Named expressions defining FAM field values.
    """

    fam_dict = {'fam_id': TString(), 'id': TString(), 'mat_id': TString(), 'pat_id': TString(),
                'is_female': TBoolean(), 'is_case': TBoolean(), 'quant_pheno': TFloat64()}

    exprs = []
    named_exprs = {k: v for k, v in fam_args.items()}
    if ('is_case' in named_exprs) and ('quant_pheno' in named_exprs):
        raise ValueError("At most one of 'is_case' and 'quant_pheno' may be given as fam_args. Found both.")
    for k, v in named_exprs.items():
        if k not in fam_dict:
            raise ValueError("fam_arg '{}' not recognized. Valid names: {}".format(k, ', '.join(fam_dict)))
        elif (v.dtype != fam_dict[k]):
            raise TypeError("fam_arg '{}' expression has type {}, expected type {}".format(k, v.dtype, fam_dict[k]))

        analyze('export_plink/{}'.format(k), v, dataset._col_indices)
        exprs.append('`{k}` = {v}'.format(k=k, v=v._ast.to_hql()))
    base, _ = dataset._process_joins(*named_exprs.values())
    base = require_biallelic(base, 'export_plink')

    Env.hail().io.plink.ExportPlink.apply(base._jvds, output, ','.join(exprs))


@handle_py4j
@typecheck(table=Table,
           zk_host=strlike,
           collection=strlike,
           block_size=integral)
def export_solr(table, zk_host, collection, block_size=100):
    """Export to Solr.
    
    Warning
    -------
    :func:`export_solr` is EXPERIMENTAL.
    """

    table._jkt.exportSolr(zk_host, collection, block_size)


@handle_py4j
@typecheck(dataset=MatrixTable,
           output=strlike,
           append_to_header=nullable(strlike),
           parallel=nullable(enumeration('separate_header', 'header_per_shard')),
           metadata=nullable(dictof(strlike, dictof(strlike, dictof(strlike, strlike)))))
def export_vcf(dataset, output, append_to_header=None, parallel=None, metadata=None):
    """Export variant dataset as a VCF file in ``.vcf`` or ``.vcf.bgz`` format.

    .. include:: ../_templates/req_tvariant.rst

    Examples
    --------
    Export to VCF as a block-compressed file:

    >>> methods.export_vcf(dataset, 'output/example.vcf.bgz')

    Notes
    -----
    :func:`export_vcf` writes the dataset to disk in VCF format as described in the
    `VCF 4.2 spec <https://samtools.github.io/hts-specs/VCFv4.2.pdf>`__.

    Use the ``.vcf.bgz`` extension rather than ``.vcf`` in the output file name
    for `blocked GZIP <http://www.htslib.org/doc/tabix.html>`__ compression.

    Note
    ----
        We strongly recommended compressed (``.bgz`` extension) and parallel
        output (`parallel` set to ``'separate_header'`` or
        ``'header_per_shard'``) when exporting large VCFs.

    Hail exports the fields of Struct `info` as INFO fields,
    the elements of Set[String] `filters` as FILTERS, and the
    value of Float64 `qual` as QUAL. No other row fields are exported.

    The FORMAT field is generated from the entry schema, which
    must be a :class:`~hail.expr.TStruct`.  There is a FORMAT
    field for each field of the Struct.

    INFO and FORMAT fields may be generated from Struct fields of type Call,
    Int32, Float32, Float64, or String. If a field has type Int64, every value
    must be a valid Int32. Arrays and Sets containing these types are also
    allowed but cannot be nested; for example, Array[Array[Int32]] is invalid.
    Sets and Arrays are written with the same comma-separated format. Boolean
    fields are also permitted in `info` and will generate INFO fields of
    VCF type Flag.

    Hail also exports the name, length, and assembly of each contig as a VCF
    header line, where the assembly is set to the :class:`.GenomeReference`
    name.

    Consider the workflow of importing a VCF and immediately exporting the
    dataset back to VCF. The output VCF header will contain FORMAT lines for
    each entry field and INFO lines for all fields in `info`, but these lines
    will have empty Description fields and the Number and Type fields will be
    determined from their corresponding Hail types. To output a desired
    Description, Number, and/or Type value in a FORMAT or INFO field or to
    specify FILTER lines, use the `metadata` parameter to supply a dictionary
    with the relevant information. See
    :func:`get_vcf_metadata` for how to obtain the
    dictionary corresponding to the original VCF, and for info on how this
    dictionary should be structured. 
    
    The output VCF header will also contain CONTIG lines
    with ID, length, and assembly fields derived from the reference genome of
    the dataset.
    
    The output VCF header will `not` contain lines added by external tools
    (such as bcftools and GATK) unless they are explicitly inserted using the
    `append_to_header` parameter.

    Warning
    -------
        INFO fields stored at VCF import are `not` automatically modified to
        reflect filtering of samples or genotypes, which can affect the value of
        AC (allele count), AF (allele frequency), AN (allele number), etc. If a
        filtered dataset is exported to VCF without updating `info`, downstream
        tools which may produce erroneous results. The solution is to create new
        annotations in `info` or overwrite existing annotations. For example, in
        order to produce an accurate `AC` field, one can run :func:`variant_qc` and
        copy the `variant_qc.AC` field to `info.AC` as shown below.
    
        >>> ds = dataset.filter_entries(dataset.GQ >= 20)
        >>> ds = methods.variant_qc(ds)
        >>> ds = ds.annotate_rows(info = ds.info.annotate(AC=ds.variant_qc.AC)) # doctest: +SKIP
        >>> methods.export_vcf(ds, 'output/example.vcf.bgz')
    
    Parameters
    ----------
    dataset : :class:`.MatrixTable`
        Dataset.
    output : :obj:`str`
        Path of .vcf or .vcf.bgz file to write.
    append_to_header : :obj:`str`, optional
        Path of file to append to VCF header.
    parallel : :obj:`str`, optional
        If ``'header_per_shard'``, return a set of VCF files (one per
        partition) rather than serially concatenating these files. If
        ``'separate_header'``, return a separate VCF header file and a set of
        VCF files (one per partition) without the header. If ``None``,
        concatenate the header and all partitions into one VCF file.
    metadata : :obj:`dict[str]` or :obj:`dict[str, dict[str, str]`, optional
        Dictionary with information to fill in the VCF header. See
        :func:`get_vcf_metadata` for how this
        dictionary should be structured.
    """

    typ = TDict(TString(), TDict(TString(), TDict(TString(), TString())))
    Env.hail().io.vcf.ExportVCF.apply(dataset._jvds, output, joption(append_to_header),
                                      Env.hail().utils.ExportType.getExportType(parallel),
                                      joption(typ._convert_to_j(metadata)))


@handle_py4j
@typecheck(path=strlike,
           reference_genome=nullable(GenomeReference))
def import_interval_list(path, reference_genome=None):
    """Import an interval list file in the GATK standard format.

    Examples
    --------

    >>> intervals = methods.import_interval_list('data/capture_intervals.txt')

    Notes
    -----

    Hail expects an interval file to contain either three or five fields per
    line in the following formats:

    - ``contig:start-end``
    - ``contig  start  end`` (tab-separated)
    - ``contig  start  end  direction  target`` (tab-separated)

    A file in either of the first two formats produces a table with one
    field:

     - **interval** (*Interval*), key field

    A file in the third format (with a "target" column) produces a table with two
    fields:

     - **interval** (*Interval*), key field
     - **target** (*String*)

    Note
    ----
    ``start`` and ``end`` match positions inclusively, e.g.
    ``start <= position <= end``. :meth:`.Interval.parse`
    is exclusive of the end position.

    Refer to :class:`.GenomeReference` for contig ordering and behavior.

    Warning
    -------
    The interval parser for these files does not support the full range of
    formats supported by the python parser
    :meth:`representation.Interval.parse`. 'k', 'm', 'start', and 'end' are all
    invalid motifs in the ``contig:start-end`` format here.

    Parameters
    ----------
    path : :obj:`str`
        Path to file.

    reference_genome : :class:`.GenomeReference`
        Reference genome to use. Default is
        :meth:`.HailContext.default_reference`.

    Returns
    -------
    :class:`.Table`
        Interval-keyed table.
    """
    rg = reference_genome if reference_genome else Env.hc().default_reference
    t = Env.hail().table.Table.importIntervalList(Env.hc()._jhc, path, rg._jrep)
    return Table(t)


@handle_py4j
@typecheck(path=strlike,
           reference_genome=nullable(GenomeReference))
def import_bed(path, reference_genome=None):
    """Import a UCSC .bed file as a :class:`.Table`.

    Examples
    --------

    >>> bed = methods.import_bed('data/file1.bed')

    >>> bed = methods.import_bed('data/file2.bed')

    The file formats are

    .. code-block:: text

        $ cat data/file1.bed
        track name="BedTest"
        20    1          14000000
        20    17000000   18000000
        ...

        $ cat file2.bed
        track name="BedTest"
        20    1          14000000  cnv1
        20    17000000   18000000  cnv2
        ...


    Notes
    -----

    The table produced by this method has one of two possible structures. If
    the .bed file has only three fields (`chrom`, `chromStart`, and
    `chromEnd`), then the produced table has only one column:

        - **interval** (*Interval*) - Genomic interval.

    If the .bed file has four or more columns, then Hail will store the fourth
    column as a field in the table:

        - **interval** (*Interval*) - Genomic interval.
        - **target** (*String*) - Fourth column of .bed file.

    `UCSC bed files <https://genome.ucsc.edu/FAQ/FAQformat.html#format1>`__ can
    have up to 12 fields, but Hail will only ever look at the first four. Hail
    ignores header lines in BED files.

    Warning
    -------
        UCSC BED files are 0-indexed and end-exclusive. The line "5  100  105"
        will contain locus ``5:105`` but not ``5:100``. Details
        `here <http://genome.ucsc.edu/blog/the-ucsc-genome-browser-coordinate-counting-systems/>`__.

    Parameters
    ----------
    path : :obj:`str`
        Path to .bed file.

    reference_genome : :class:`.GenomeReference`
        Reference genome to use. Default is
        :meth:`.HailContext.default_reference`.

    Returns
    -------
    :class:`.Table`
        Interval-indexed table containing information from file.
    """
    # FIXME: once interval join support is added, add the following examples:
    # Add the variant annotation ``va.cnvRegion: Boolean`` indicating inclusion in
    # at least one interval of the three-column BED file `file1.bed`:

    # >>> bed = methods.import_bed('data/file1.bed')
    # >>> vds_result = vds.annotate_rows(cnvRegion = bed[vds.v])

    # Add a variant annotation **va.cnvRegion** (*String*) with value given by the
    # fourth column of ``file2.bed``:

    # >>> bed = methods.import_bed('data/file2.bed')
    # >>> vds_result = vds.annotate_rows(cnvID = bed[vds.v])

    rg = reference_genome if reference_genome else Env.hc().default_reference
    jt = Env.hail().table.Table.importBED(Env.hc()._jhc, path, rg._jrep)
    return Table(jt)


@handle_py4j
@typecheck(path=strlike,
           quant_pheno=bool,
           delimiter=strlike,
           missing=strlike)
def import_fam(path, quant_pheno=False, delimiter=r'\\s+', missing='NA'):
    """Import PLINK .fam file into a key table.

    Examples
    --------

    Import a tab-separated
    `FAM file <https://www.cog-genomics.org/plink2/formats#fam>`__
    with a case-control phenotype:

    >>> fam_kt = methods.import_fam('data/case_control_study.fam')

    Import a FAM file with a quantitative phenotype:

    >>> fam_kt = methods.import_fam('data/quantitative_study.fam', quant_pheno=True)

    Notes
    -----

    In Hail, unlike PLINK, the user must *explicitly* distinguish between
    case-control and quantitative phenotypes. Importing a quantitative
    phenotype without ``quant_pheno=True`` will return an error
    (unless all values happen to be `0`, `1`, `2`, or `-9`):

    The resulting :class:`.Table` will have fields, types, and values that are interpreted as missing.

     - **fam_id** (*String*) -- Family ID (missing = "0")
     - **id** (*String*) -- Sample ID (key column)
     - **pat_id** (*String*) -- Paternal ID (missing = "0")
     - **mat_id** (*String*) -- Maternal ID (missing = "0")
     - **is_female** (*Boolean*) -- Sex (missing = "NA", "-9", "0")

    One of:

     - **is_case** (*Boolean*) -- Case-control phenotype (missing = "0", "-9",
        non-numeric or the ``missing`` argument, if given.
     - **quant_pheno** (*Float64*) -- Quantitative phenotype (missing = "NA" or the
        ``missing`` argument, if given.

    Parameters
    ----------
    path : :obj:`str`
        Path to FAM file.
    quant_pheno : :obj:`bool`
        If ``True``, phenotype is interpreted as quantitative.
    delimiter : :obj:`str`
        Field delimiter regex.
    missing : :obj:`str`
        The string used to denote missing values. For case-control, 0, -9, and
        non-numeric are also treated as missing.

    Returns
    -------
    :class:`.Table`
        Table representing the data of a FAM file.
    """

    jkt = Env.hail().table.Table.importFam(Env.hc()._jhc, path,
                                           quant_pheno, delimiter, missing)
    return Table(jkt)


@handle_py4j
@typecheck(regex=strlike,
           path=oneof(strlike, listof(strlike)),
           max_count=integral)
def grep(regex, path, max_count=100):
    Env.hc()._jhc.grep(regex, jindexed_seq_args(path), max_count)


@handle_py4j
@typecheck(path=oneof(strlike, listof(strlike)),
           tolerance=numeric,
           sample_file=nullable(strlike),
           min_partitions=nullable(integral),
           reference_genome=nullable(GenomeReference),
           contig_recoding=nullable(dictof(strlike, strlike)))
def import_bgen(path, tolerance=0.2, sample_file=None, min_partitions=None, reference_genome=None,
                contig_recoding=None):
    rg = reference_genome if reference_genome else Env.hc().default_reference
    
    if contig_recoding:
        contig_recoding = TDict(TString(), TString())._convert_to_j(contig_recoding)
        
    jmt = Env.hc()._jhc.importBgens(jindexed_seq_args(path), joption(sample_file),
                                    tolerance, joption(min_partitions), rg._jrep,
                                    joption(contig_recoding))
    return MatrixTable(jmt)


@handle_py4j
@typecheck(path=oneof(strlike, listof(strlike)),
           sample_file=nullable(strlike),
           tolerance=numeric,
           min_partitions=nullable(integral),
           chromosome=nullable(strlike),
           reference_genome=nullable(GenomeReference),
           contig_recoding=nullable(dictof(strlike, strlike)))
def import_gen(path, sample_file=None, tolerance=0.2, min_partitions=None, chromosome=None, reference_genome=None,
               contig_recoding=None):
    rg = reference_genome if reference_genome else Env.hc().default_reference

    if contig_recoding:
        contig_recoding = TDict(TString(), TString())._convert_to_j(contig_recoding)

    jmt = Env.hc()._jhc.importGens(jindexed_seq_args(path), sample_file, joption(chromosome), joption(min_partitions),
                                   tolerance, rg._jrep, joption(contig_recoding))
    return MatrixTable(jmt)


@handle_py4j
@typecheck(paths=oneof(strlike, listof(strlike)),
           key=oneof(strlike, listof(strlike)),
           min_partitions=nullable(int),
           impute=bool,
           no_header=bool,
           comment=nullable(strlike),
           delimiter=strlike,
           missing=strlike,
           types=dictof(strlike, Type),
           quote=nullable(char),
           reference_genome=nullable(GenomeReference))
def import_table(paths, key=[], min_partitions=None, impute=False, no_header=False,
                 comment=None, delimiter="\t", missing="NA", types={}, quote=None, reference_genome=None):
    key = wrap_to_list(key)
    paths = wrap_to_list(paths)
    jtypes = {k: v._jtype for k, v in types.items()}
    rg = reference_genome if reference_genome else Env.hc().default_reference
    
    jt = Env.hc()._jhc.importTable(paths, key, min_partitions, jtypes, comment, delimiter, missing,
                                   no_header, impute, quote, rg._jrep)
    return Table(jt)


@handle_py4j
@typecheck(bed=strlike,
           bim=strlike,
           fam=strlike,
           min_partitions=nullable(integral),
           delimiter=strlike,
           missing=strlike,
           quant_pheno=bool,
           a2_reference=bool,
           reference_genome=nullable(GenomeReference),
           contig_recoding=nullable(dictof(strlike, strlike)),
           drop_chr0=bool)
def import_plink(bed, bim, fam, min_partitions=None, delimiter='\\\\s+',
                 missing='NA', quant_pheno=False, a2_reference=True, reference_genome=None,
                 contig_recoding={'23': 'X', '24': 'Y', '25': 'X', '26': 'MT'}, drop_chr0=False):
    """Import PLINK binary file (BED, BIM, FAM) as variant dataset.

    Examples
    --------

    Import data from a PLINK binary file:

    >>> vds = methods.import_plink(bed="data/test.bed",
    ...                            bim="data/test.bim",
    ...                            fam="data/test.fam")

    Notes
    -----

    Only binary SNP-major mode files can be read into Hail. To convert your
    file from individual-major mode to SNP-major mode, use PLINK to read in
    your fileset and use the ``--make-bed`` option.

    The centiMorgan position is not currently used in Hail (Column 3 in BIM
    file).

    The ID (``s``) used by Hail is the individual ID (column 2 in FAM file).

    .. warning::

        No duplicate individual IDs are allowed.

    Annotations
    -----------

    :meth:`~hail.HailContext.import_plink` adds the following annotations:

     - **va.rsid** (*String*) -- Column 2 in the BIM file.
     - **sa.famID** (*String*) -- Column 1 in the FAM file. Set to missing if
       ID equals "0".
     - **sa.patID** (*String*) -- Column 3 in the FAM file. Set to missing if
       ID equals "0".
     - **sa.matID** (*String*) -- Column 4 in the FAM file. Set to missing if
       ID equals "0".
     - **sa.isFemale** (*String*) -- Column 5 in the FAM file. Set to missing
       if value equals "-9", "0", or "N/A". Set to true if value equals "2".
       Set to false if value equals "1".
     - **sa.isCase** (*String*) -- Column 6 in the FAM file. Only present if
       ``quantpheno`` equals False. Set to missing if value equals "-9", "0",
       "N/A", or the value specified by ``missing``. Set to true if value
       equals "2". Set to false if value equals "1".
     - **sa.qPheno** (*String*) -- Column 6 in the FAM file. Only present if
       ``quantpheno`` equals True. Set to missing if value equals ``missing``.

    Parameters
    ----------
    bed : :obj:`str`
        PLINK BED file.

    bim : :obj:`str`
        PLINK BIM file.

    fam : :obj:`str`
        PLINK FAM file.

    min_partitions : :obj:`int` or :obj:`None`
        Number of partitions.

    missing : :obj:`str`
        The string used to denote missing values **only** for the phenotype
        field. This is in addition to "-9", "0", and "N/A" for case-control
        phenotypes.

    delimiter : :obj:`str`
        FAM file field delimiter regex.

    quant_pheno : :obj:`bool`
    If True, FAM phenotype is interpreted as quantitative.

    a2_reference : :obj:`bool`
        If True, A2 is treated as the reference allele. If False, A1 is treated
        as the reference allele.

    reference_genome : :class:`.GenomeReference`
        Reference genome to use. Default is
        :class:`~.HailContext.default_reference`.

    contig_recoding : :obj:`dict` of :obj:`str` to :obj:`str` (or :obj:`None`).
        Dict of old contig name to new contig name. The new contig name must be
        in the reference genome given by ``reference_genome``.

    drop_chr0 : :obj:`bool`
        If True, do not include variants with contig == "0".

    Returns
    -------
    :class:`.VariantDataset`
        Variant dataset imported from PLINK binary file.
    """
    rg = reference_genome if reference_genome else Env.hc().default_reference

    if contig_recoding:
        contig_recoding = TDict(TString(), TString())._convert_to_j(contig_recoding)

    jmt = Env.hc()._jhc.importPlink(bed, bim, fam, joption(min_partitions), delimiter,
                                    missing, quant_pheno, a2_reference, rg._jrep,
                                    joption(contig_recoding), drop_chr0)

    return MatrixTable(jmt)


@handle_py4j
@typecheck(path=oneof(strlike, listof(strlike)),
           drop_cols=bool,
           drop_rows=bool)
def read_matrix(path, drop_cols=False, drop_rows=False):
    return MatrixTable(Env.hc()._jhc.read(path, drop_cols, drop_rows))


@handle_py4j
@typecheck(path=strlike)
def get_vcf_metadata(path):
    """Extract metadata from VCF header.

    Examples
    --------

    >>> metadata = methods.get_vcf_metadata('data/example2.vcf.bgz')
    {'filter': {'LowQual': {'Description': ''}, ...},
     'format': {'AD': {'Description': 'Allelic depths for the ref and alt alleles in the order listed',
                       'Number': 'R',
                       'Type': 'Integer'}, ...},
     'info': {'AC': {'Description': 'Allele count in genotypes, for each ALT allele, in the same order as listed',
                     'Number': 'A',
                     'Type': 'Integer'}, ...}}

    Notes
    -----

    This method parses the VCF header to extract the `ID`, `Number`,
    `Type`, and `Description` fields from FORMAT and INFO lines as
    well as `ID` and `Description` for FILTER lines. For example,
    given the following header lines:

    .. code-block:: text

        ##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">
        ##FILTER=<ID=LowQual,Description="Low quality">
        ##INFO=<ID=MQ,Number=1,Type=Float,Description="RMS Mapping Quality">

    The resulting Python dictionary returned would be

    .. code-block:: python

        metadata = {'filter': {'LowQual': {'Description': 'Low quality'}},
                    'format': {'DP': {'Description': 'Read Depth',
                                      'Number': '1',
                                      'Type': 'Integer'}},
                    'info': {'MQ': {'Description': 'RMS Mapping Quality',
                                    'Number': '1',
                                    'Type': 'Float'}}}

    which can be used with :meth:`.export_vcf` to fill in the relevant fields in the header.
    
    Parameters
    ----------
    path : `obj`:str
        VCF file(s) to read. If more than one file is given, the first
        file is used.

    Returns
    -------
    `obj`:dict: of `obj`:str: to (`obj`:dict: of `obj`:str: to (`obj`:dict: of `obj`:str: to `obj`:str:))
    """
    typ = TDict(TString(), TDict(TString(), TDict(TString(), TString())))
    return typ._convert_to_py(Env.hc()._jhc.parseVCFMetadata(path))


@handle_py4j
@typecheck(path=oneof(strlike, listof(strlike)),
           force=bool,
           force_bgz=bool,
           header_file=nullable(strlike),
           min_partitions=nullable(integral),
           drop_samples=bool,
           call_fields=oneof(strlike, listof(strlike)),
           reference_genome=nullable(GenomeReference),
           contig_recoding=nullable(dictof(strlike, strlike)))
def import_vcf(path, force=False, force_bgz=False, header_file=None, min_partitions=None,
               drop_samples=False, call_fields=[], reference_genome=None, contig_recoding=None):
    """Import VCF file(s) as a matrix table.

    Examples
    --------

    >>> ds = methods.import_vcf('data/example2.vcf.bgz')

    Notes
    -----

    Hail is designed to be maximally compatible with files in the `VCF v4.2
    spec <https://samtools.github.io/hts-specs/VCFv4.2.pdf>`__.

    :func:`.import_vcf` takes a list of VCF files to load. All files must have
    the same header and the same set of samples in the same order (e.g., a
    dataset split by chromosome). Files can be specified as :ref:`Hadoop glob
    patterns <sec-hadoop-glob>`.

    Ensure that the VCF file is correctly prepared for import: VCFs should
    either be uncompressed (**.vcf**) or block compressed (**.vcf.bgz**). If you
    have a large compressed VCF that ends in **.vcf.gz**, it is likely that the
    file is actually block-compressed, and you should rename the file to
    **.vcf.bgz** accordingly. If you actually have a standard gzipped file, it
    is possible to import it to Hail using the `force` parameter. However, this
    is not recommended -- all parsing will have to take place on one node
    because gzip decompression is not parallelizable. In this case, import will
    take significantly longer.

    :func:`.import_vcf` does not perform deduplication - if the provided VCF(s)
    contain multiple records with the same chrom, pos, ref, alt, all these
    records will be imported as-is (in multiple rows) and will not be collapsed
    into a single variant.

    .. note::

        Using the **FILTER** field:

        The information in the FILTER field of a VCF is contained in the
        ``filters`` row field. This annotation is a ``Set[String]`` and can be
        queried for filter membership with expressions like
        ``ds.filters.contains("VQSRTranche99.5...")``. Variants that are flagged
        as "PASS" will have no filters applied; for these variants,
        ``ds.filters.is_empty()`` is ``True``. Thus, filtering to PASS variants
        can be done with :meth:`.MatrixTable.filter_rows` as follows:

        >>> pass_ds = dataset.filter_rows(ds.filters.is_empty())

    **Column Fields**

    - `s` (:class:`.TString`) -- Column key. This is the sample ID.

    **Row Fields**

    - `v` (:class:`.TVariant`) -- Row key. This is the variant created from the CHROM,
      POS, REF, and ALT fields.
    - `filters` (:class:`.TSet` of :class:`.TString`) -- Set containing all filters applied to a
      variant.
    - `rsid` (:class:`.TString`) -- rsID of the variant.
    - `qual` (:class:`.TFloat64`) -- Floating-point number in the QUAL field.
    - `info` (:class:`.TStruct`) -- All INFO fields defined in the VCF header
      can be found in the struct `info`. Data types match the type specified
      in the VCF header, and if the declared ``Number`` is not 1, the result
      will be stored as an array.

    **Entry Fields**

    :func:`.import_vcf` generates an entry field for each FORMAT field declared
    in the VCF header. The types of these fields are generated according to the
    same rules as INFO fields, with one difference -- "GT" and other fields
    specified in `call_fields` will be read as :class:`.TCall`.

    Parameters
    ----------
    path : :obj:`str` or :obj:`list` of :obj:`str`
        VCF file(s) to read.
    force : :obj:`bool`
        If ``True``, load **.vcf.gz** files serially. No downstream operations
        can be parallelized, so this mode is strongly discouraged.
    force_bgz : :obj:`bool`
        If ``True``, load **.vcf.gz** files as blocked gzip files, assuming
        that they were actually compressed using the BGZ codec.
    header_file : :obj:`str`, optional
        Optional header override file. If not specified, the first file in
        `path` is used.
    min_partitions : :obj:`int`, optional
        Minimum partitions to load per file.
    drop_samples : :obj:`bool`
        If ``True``, create sites-only dataset. Don't load sample IDs or
        entries.
    call_fields : :obj:`list` of :obj:`str`
        List of FORMAT fields to load as :class:`.TCall`. "GT" is loaded as
        a call automatically.
    reference_genome: :class:`.GenomeReference`, optional
        Reference genome to use. If ``None``, then the
        :meth:`.HailContext.default_reference` is used.
    contig_recoding: :obj:`dict` of (:obj:`str`, :obj:`str`)
        Mapping from contig name in VCF to contig name in loaded dataset.
        All contigs must be present in the `reference_genome`, so this is
        useful for mapping differently-formatted data onto known references.

    Returns
    -------
    :class:`.MatrixTable`
    """
    from hail2 import default_reference
    rg = reference_genome if reference_genome else default_reference()

    if contig_recoding:
        contig_recoding = TDict(TString(), TString())._convert_to_j(contig_recoding)

    jmt = Env.hc()._jhc.importVCFs(jindexed_seq_args(path), force, force_bgz, joption(header_file),
                                   joption(min_partitions), drop_samples, jset_args(call_fields), rg._jrep,
                                   joption(contig_recoding))

    return MatrixTable(jmt)


@handle_py4j
@typecheck(path=oneof(strlike, listof(strlike)))
def index_bgen(path):
    Env.hc()._jhc.indexBgen(jindexed_seq_args(path))


@handle_py4j
@typecheck(path=strlike)
def read_table(path):
    Table(Env.hc()._jhc.readTable(path))
