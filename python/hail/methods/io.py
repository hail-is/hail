from hail.api2 import MatrixTable, Table
from hail.expr.types import *
from hail.genetics import GenomeReference
from hail.history import *
from hail.typecheck import *
from hail.utils.java import Env, handle_py4j, joption

@handle_py4j
@typecheck(dataset=MatrixTable,
           output=strlike,
           append_to_header=nullable(strlike),
           parallel=nullable(enumeration('separate_header', 'header_per_shard')),
           metadata=nullable(dictof(strlike, dictof(strlike, dictof(strlike, strlike)))))
def export_vcf(dataset, output, append_to_header=None, parallel=None, metadata=None):
    """Export variant dataset as a ``.vcf`` or ``.vcf.bgz`` file.

    .. include:: ../_templates/req_tvariant.rst

    Examples
    --------
    Export to VCF as a block-compressed file:

    >>> methods.export_vcf(dataset, 'output/example.vcf.bgz')

    Notes
    -----
    :meth:`export_vcf` writes the dataset to disk in VCF format as described in the
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
    must be a :py:class:`~hail.expr.TStruct`.  There is a FORMAT
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
    :py:class:`~hail.api2.HailContext.get_vcf_metadata` for how to obtain the
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
    order to produce an accurate `AC` field, one can run :meth:`variant_qc` and
    copy the `variant_qc.AC` field to `info.AC` as shown below.

    >>> dataset = dataset.filter_entries(dataset.GQ >= 20)
    >>> dataset = methods.variant_qc(dataset)
    >>> dataset = dataset.annotate_rows(info = dataset.info.annotate(AC=dataset.variant_qc.AC)) # doctest: +SKIP
    >>> methods.export_vcf(dataset, 'output/example.vcf.bgz')
    
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
        :py:class:`~hail.api2.HailContext.get_vcf_metadata` for how this
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

    >>> intervals = KeyTable.import_interval_list('data/capture_intervals.txt')

    The File Format
    ---------------

    Hail expects an interval file to contain either three or five fields per
    line in the following formats:

    - ``contig:start-end``
    - ``contig  start  end`` (tab-separated)
    - ``contig  start  end  direction  target`` (tab-separated)

    A file in either of the first two formats produces a key table with one
    column:

     - **interval** (*Interval*), key column

    A file in the third format (with a "target" column) produces a key with two
    columns:

     - **interval** (*Interval*), key column
     - **target** (*String*)

    Note
    ----
    ``start`` and ``end`` match positions inclusively, e.g.
    ``start <= position <= end``. :meth:`representation.Interval.parse`
    is exclusive of the end position.

    Note
    ----
    Hail uses the following ordering for contigs: 1-22 sorted numerically, then
    X, Y, MT, then alphabetically for any contig not matching the standard human
    chromosomes.

    Warning
    -------
    The interval parser for these files does not support the full range of
    formats supported by the python parser
    :meth:`representation.Interval.parse`. 'k', 'm', 'start', and 'end' are all
    invalid motifs in the ``contig:start-end`` format here.

    Parameters
    ----------
    filename : :obj:`str`
        Path to file.

    reference_genome : :class:`.GenomeReference`
        Reference genome to use. Default is
        :class:`.HailContext.default_reference`.

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
    """Import a UCSC .bed file as a key table.

    Examples
    --------

    Add the variant annotation ``va.cnvRegion: Boolean`` indicating inclusion in
    at least one interval of the three-column BED file `file1.bed`:

    >>> bed = methods.import_bed('data/file1.bed')
    >>> vds_result = vds.annotate_rows(cnvRegion = bed[vds.v])

    Add a variant annotation **va.cnvRegion** (*String*) with value given by the
    fourth column of ``file2.bed``:

    >>> bed = methods.import_bed('data/file2.bed')
    >>> vds_result = vds.annotate_rows(cnvID = bed[vds.v])

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

    The key table produced by this method has one of two possible structures. If
    the .bed file has only three fields (``chrom``, ``chromStart``, and
    ``chromEnd``), then the produced key table has only one column:

        - **interval** (*Interval*) - Genomic interval.

    If the .bed file has four or more columns, then Hail will store the fourth
    column in the table:

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
        """

    rg = reference_genome if reference_genome else Env.hc().default_reference
    jt = Env.hail().table.Table.importBED(Env.hc()._jhc, path, rg._jrep)
    return Table(jt)

@handle_py4j
@typecheck(path=strlike,
           quantitative=bool,
           delimiter=strlike,
           missing=strlike)
def import_fam(path, quantitative=False, delimiter='\\\\s+', missing='NA'):
    """Import PLINK .fam file into a key table.

    Examples
    --------

    Import case-control phenotype data from a tab-separated `PLINK .fam
    <https://www.cog-genomics.org/plink2/formats#fam>`_ file into sample
    annotations:

    >>> fam_kt = KeyTable.import_fam('data/myStudy.fam')

    In Hail, unlike PLINK, the user must *explicitly* distinguish between
    case-control and quantitative phenotypes. Importing a quantitative
    phenotype without ``quantitative=True`` will return an error
    (unless all values happen to be ``0``, ``1``, ``2``, and ``-9``):

    >>> fam_kt = KeyTable.import_fam('data/myStudy.fam', quantitative=True)

    Columns
    -------

    The column, types, and missing values are shown below.

        - **ID** (*String*) -- Sample ID (key column)
        - **famID** (*String*) -- Family ID (missing = "0")
        - **patID** (*String*) -- Paternal ID (missing = "0")
        - **matID** (*String*) -- Maternal ID (missing = "0")
        - **isFemale** (*Boolean*) -- Sex (missing = "NA", "-9", "0")

    One of:

        - **isCase** (*Boolean*) -- Case-control phenotype (missing = "0", "-9",
        non-numeric or the ``missing`` argument, if given.
        - **qPheno** (*Double*) -- Quantitative phenotype (missing = "NA" or the
        ``missing`` argument, if given.

    Parameters
    ----------
    path : :obj:`str`
        Path to .fam file.

    quantitative : :obj:`bool`
        If True, .fam phenotype is interpreted as quantitative.

    delimiter : :obj:`str`
        .fam file field delimiter regex.

    missing : :obj:`str`
        The string used to denote missing values. For case-control, 0, -9, and
        non-numeric are also treated as missing.

    Returns
    -------
    :class:`.Table`
        Table with information from .fam file.
    """

    jkt = Env.hail().table.Table.importFam(Env.hc()._jhc, path,
                                           quantitative, delimiter, missing)
    return Table(jkt)
