import json

from hail.typecheck import *
from hail.utils.java import Env, joption, FatalError, jindexed_seq_args, jset_args
from hail.utils import wrap_to_list
from hail.utils.misc import plural
from hail.matrixtable import MatrixTable
from hail.table import Table
from hail.expr.types import *
from hail.expr.expressions import *
from hail.ir import *
from hail.genetics.reference_genome import reference_genome_type
from hail.methods.misc import require_biallelic, require_row_key_variant, require_row_key_variant_w_struct_locus, require_col_key_str
import hail as hl

from typing import List

_cached_importvcfs = None


def locus_interval_expr(contig, start, end, includes_start, includes_end,
                        reference_genome, skip_invalid_intervals):
    if reference_genome:
        if skip_invalid_intervals:
            is_valid_locus_interval = (
                (hl.is_valid_contig(contig, reference_genome) &
                 (hl.is_valid_locus(contig, start, reference_genome) |
                  (~hl.bool(includes_start) & (start == 0))) &
                 (hl.is_valid_locus(contig, end, reference_genome) |
                  (~hl.bool(includes_end) & hl.is_valid_locus(contig, end - 1, reference_genome)))))

            return hl.or_missing(is_valid_locus_interval,
                                 hl.locus_interval(contig, start, end,
                                                   includes_start, includes_end,
                                                   reference_genome))
        else:
            return hl.locus_interval(contig, start, end, includes_start,
                                     includes_end, reference_genome)
    else:
        return hl.interval(hl.struct(contig=contig, position=start),
                           hl.struct(contig=contig, position=end),
                           includes_start,
                           includes_end)


def expr_or_else(expr, default, f=lambda x: x):
    if expr is not None:
        return hl.or_else(f(expr), default)
    else:
        return to_expr(default)


@typecheck(dataset=MatrixTable,
           output=str,
           precision=int,
           gp=nullable(expr_array(expr_float64)),
           id1=nullable(expr_str),
           id2=nullable(expr_str),
           missing=nullable(expr_numeric),
           varid=nullable(expr_str),
           rsid=nullable(expr_str))
def export_gen(dataset, output, precision=4, gp=None, id1=None, id2=None,
               missing=None, varid=None, rsid=None):
    """Export a :class:`.MatrixTable` as GEN and SAMPLE files.

    .. include:: ../_templates/req_tvariant.rst

    .. include:: ../_templates/req_biallelic.rst

    Examples
    --------
    Import genotype probability data, filter variants based on INFO score, and
    export data to a GEN and SAMPLE file:

    >>> example_ds = hl.import_gen('data/example.gen', sample_file='data/example.sample')
    >>> example_ds = example_ds.filter_rows(agg.info_score(example_ds.GP).score >= 0.9) # doctest: +SKIP
    >>> hl.export_gen(example_ds, 'output/infoscore_filtered')

    Notes
    -----
    Writes out the dataset to a GEN and SAMPLE fileset in the
    `Oxford spec <http://www.stats.ox.ac.uk/%7Emarchini/software/gwas/file_format.html>`__.

    Parameters
    ----------
    dataset : :class:`.MatrixTable`
        Dataset.
    output : :obj:`str`
        Filename root for output GEN and SAMPLE files.
    precision : :obj:`int`
        Number of digits to write after the decimal point.
    gp : :class:`.ArrayExpression` of type :py:data:`.tfloat64`, optional
        Expression for the genotype probabilities to output. If ``None``, the
        entry field `GP` is used if defined and is of type :py:data:`.tarray`
        with element type :py:data:`.tfloat64`. The array length must be 3.
        The values at indices 0, 1, and 2 are exported as the probabilities of
        homozygous reference, heterozygous, and homozygous variant,
        respectively. The default and missing value is ``[0, 0, 0]``.
    id1 : :class:`.StringExpression`, optional
        Expression for the first column of the SAMPLE file. If ``None``, the
        column key of the dataset is used and must be one field of type
        :py:data:`.tstr`.
    id2 : :class:`.StringExpression`, optional
        Expression for the second column of the SAMPLE file. If ``None``, the
        column key of the dataset is used and must be one field of type
        :py:data:`.tstr`.
    missing : :class:`.NumericExpression`, optional
        Expression for the third column of the SAMPLE file, which is the sample
        missing rate. Values must be between 0 and 1.
    varid : :class:`.StringExpression`, optional
        Expression for the variant ID (2nd column of the GEN file). If ``None``,
        the row field `varid` is used if defined and is of type :py:data:`.tstr`.
        The default and missing value is
        ``hl.delimit([dataset.locus.contig, hl.str(dataset.locus.position), dataset.alleles[0], dataset.alleles[1]], ':')``
    rsid : :class:`.StringExpression`, optional
        Expression for the rsID (3rd column of the GEN file). If ``None``,
        the row field `rsid` is used if defined and is of type :py:data:`.tstr`.
        The default and missing value is ``"."``.
    """

    require_biallelic(dataset, 'export_gen')

    if gp is None:
        if 'GP' in dataset.entry and dataset.GP.dtype == tarray(tfloat64):
            entry_exprs = {'GP': dataset.GP}
        else:
            entry_exprs = {}
    else:
        entry_exprs = {'GP': gp}

    if id1 is None:
        require_col_key_str(dataset, "export_gen")
        id1 = dataset.col_key[0]

    if id2 is None:
        require_col_key_str(dataset, "export_gen")
        id2 = dataset.col_key[0]

    if missing is None:
        missing = hl.float64(0.0)

    if varid is None:
        if 'varid' in dataset.row and dataset.varid.dtype == tstr:
            varid = dataset.varid

    if rsid is None:
        if 'rsid' in dataset.row and dataset.rsid.dtype == tstr:
            rsid = dataset.rsid

    sample_exprs = {'id1': id1, 'id2': id2, 'missing': missing}

    l = dataset.locus
    a = dataset.alleles

    gen_exprs = {'varid': expr_or_else(varid, hl.delimit([l.contig, hl.str(l.position), a[0], a[1]], ':')),
                 'rsid': expr_or_else(rsid, ".")}

    for exprs, axis in [(sample_exprs, dataset._col_indices),
                        (gen_exprs, dataset._row_indices),
                        (entry_exprs, dataset._entry_indices)]:
        for name, expr in exprs.items():
            analyze('export_gen/{}'.format(name), expr, axis)

    dataset = dataset._select_all(col_exprs=sample_exprs,
                                  col_key=[],
                                  row_exprs=gen_exprs,
                                  entry_exprs=entry_exprs)

    writer = MatrixGENWriter(output, precision)
    Env.backend().execute(MatrixWrite(dataset._mir, writer))


@typecheck(dataset=MatrixTable,
           output=str,
           call=nullable(expr_call),
           fam_id=nullable(expr_str),
           ind_id=nullable(expr_str),
           pat_id=nullable(expr_str),
           mat_id=nullable(expr_str),
           is_female=nullable(expr_bool),
           pheno=oneof(nullable(expr_bool), nullable(expr_numeric)),
           varid=nullable(expr_str),
           cm_position=nullable(expr_float64))
def export_plink(dataset, output, call=None, fam_id=None, ind_id=None, pat_id=None,
                 mat_id=None, is_female=None, pheno=None, varid=None,
                 cm_position=None):
    """Export a :class:`.MatrixTable` as
    `PLINK2 <https://www.cog-genomics.org/plink2/formats>`__
    BED, BIM and FAM files.

    .. include:: ../_templates/req_tvariant_w_struct_locus.rst
    .. include:: ../_templates/req_tstring.rst
    .. include:: ../_templates/req_biallelic.rst
    .. include:: ../_templates/req_unphased_diploid_gt.rst

    Examples
    --------
    Import data from a VCF file, split multi-allelic variants, and export to
    PLINK files with the FAM file individual ID set to the sample ID:

    >>> ds = hl.split_multi_hts(dataset)
    >>> hl.export_plink(ds, 'output/example', ind_id = ds.s)

    Notes
    -----
    On an imported VCF, the example above will behave similarly to the PLINK
    conversion command

    .. code-block:: text

        plink --vcf /path/to/file.vcf --make-bed --out sample --const-fid --keep-allele-order

    except that:

    - Variants that result from splitting a multi-allelic variant may be
      re-ordered relative to the BIM and BED files.

    Parameters
    ----------
    dataset : :class:`.MatrixTable`
        Dataset.
    output : :obj:`str`
        Filename root for output BED, BIM, and FAM files.
    call : :class:`.CallExpression`, optional
        Expression for the genotype call to output. If ``None``, the entry field
        `GT` is used if defined and is of type :py:data:`.tcall`.
    fam_id : :class:`.StringExpression`, optional
        Expression for the family ID. The default and missing values are
        ``'0'``.
    ind_id : :class:`.StringExpression`, optional
        Expression for the individual (proband) ID. If ``None``, the column key
        of the dataset is used and must be one field of type :py:data:`.tstr`.
    pat_id : :class:`.StringExpression`, optional
        Expression for the paternal ID. The default and missing values are
        ``'0'``.
    mat_id : :class:`.StringExpression`, optional
        Expression for the maternal ID. The default and missing values are
        ``'0'``.
    is_female : :class:`.BooleanExpression`, optional
        Expression for the proband sex. ``True`` is output as ``'2'`` and
        ``False`` is output as ``'1'``. The default and missing values are
        ``'0'``.
    pheno : :class:`.BooleanExpression` or :class:`.NumericExpression`, optional
        Expression for the phenotype. If `pheno` is a boolean expression,
        ``True`` is output as ``'2'`` and ``False`` is output as ``'1'``. The
        default and missing values are ``'NA'``.
    varid : :class:`.StringExpression`, optional
        Expression for the variant ID (2nd column of the BIM file). The default
        value is ``hl.delimit([dataset.locus.contig, hl.str(dataset.locus.position), dataset.alleles[0], dataset.alleles[1]], ':')``
    cm_position : :class:`.Float64Expression`, optional
        Expression for the 3rd column of the BIM file (position in centimorgans).
        The default value is ``0.0``. The missing value is ``0.0``.
    """

    require_biallelic(dataset, 'export_plink')
    require_row_key_variant_w_struct_locus(dataset, 'export_plink')

    if ind_id is None:
        require_col_key_str(dataset, "export_plink")
        ind_id = dataset.col_key[0]

    if call is None:
        if 'GT' in dataset.entry and dataset.GT.dtype == tcall:
            entry_exprs = {'GT': dataset.GT}
        else:
            entry_exprs = {}
    else:
        entry_exprs = {'GT': call}

    fam_exprs = {'fam_id': expr_or_else(fam_id, '0'),
                 'ind_id': hl.or_else(ind_id, '0'),
                 'pat_id': expr_or_else(pat_id, '0'),
                 'mat_id': expr_or_else(mat_id, '0'),
                 'is_female': expr_or_else(is_female, '0',
                                           lambda x: hl.cond(x, '2', '1')),
                 'pheno': expr_or_else(pheno, 'NA',
                                       lambda x: hl.cond(x, '2', '1') if x.dtype == tbool else hl.str(x))}

    l = dataset.locus
    a = dataset.alleles

    bim_exprs = {'varid': expr_or_else(varid, hl.delimit([l.contig, hl.str(l.position), a[0], a[1]], ':')),
                 'cm_position': expr_or_else(cm_position, 0.0)}

    for exprs, axis in [(fam_exprs, dataset._col_indices),
                        (bim_exprs, dataset._row_indices),
                        (entry_exprs, dataset._entry_indices)]:
        for name, expr in exprs.items():
            analyze('export_plink/{}'.format(name), expr, axis)

    dataset = dataset._select_all(col_exprs=fam_exprs,
                                  col_key=[],
                                  row_exprs=bim_exprs,
                                  entry_exprs=entry_exprs)

    # check FAM ids for white space
    t_cols = dataset.cols()
    errors = []
    for name in ['ind_id', 'fam_id', 'pat_id', 'mat_id']:
        ids = t_cols.filter(t_cols[name].matches(r"\s+"))[name].collect()

        if ids:
            errors.append(f"""expr '{name}' has spaces in the following values:\n""")
            for row in ids:
                errors.append(f"""  {row}\n""")

    if errors:
        raise TypeError("\n".join(errors))

    writer = MatrixPLINKWriter(output)
    Env.backend().execute(MatrixWrite(dataset._mir, writer))


@typecheck(dataset=MatrixTable,
           output=str,
           append_to_header=nullable(str),
           parallel=nullable(enumeration('separate_header', 'header_per_shard')),
           metadata=nullable(dictof(str, dictof(str, dictof(str, str)))))
def export_vcf(dataset, output, append_to_header=None, parallel=None, metadata=None):
    """Export a :class:`.MatrixTable` as a VCF file.

    .. include:: ../_templates/req_tvariant.rst

    Examples
    --------
    Export to VCF as a block-compressed file:

    >>> hl.export_vcf(dataset, 'output/example.vcf.bgz')

    Notes
    -----
    :func:`.export_vcf` writes the dataset to disk in VCF format as described in the
    `VCF 4.2 spec <https://samtools.github.io/hts-specs/VCFv4.2.pdf>`__.

    Use the ``.vcf.bgz`` extension rather than ``.vcf`` in the output file name
    for `blocked GZIP <http://www.htslib.org/doc/tabix.html>`__ compression.

    Note
    ----
        We strongly recommended compressed (``.bgz`` extension) and parallel
        output (`parallel` set to ``'separate_header'`` or
        ``'header_per_shard'``) when exporting large VCFs.

    Hail exports the fields of struct `info` as INFO fields, the elements of
    ``set<str>`` `filters` as FILTERS, the value of str `rsid` as ID, and the
    value of float64 `qual` as QUAL. No other row fields are exported.

    The FORMAT field is generated from the entry schema, which
    must be a :class:`.tstruct`.  There is a FORMAT
    field for each field of the Struct.

    INFO and FORMAT fields may be generated from Struct fields of type
    :py:data:`.tcall`, :py:data:`.tint32`, :py:data:`.tfloat32`,
    :py:data:`.tfloat64`, or :py:data:`.tstr`. If a field has type
    :py:data:`.tint64`, every value must be a valid ``int32``. Arrays and sets
    containing these types are also allowed but cannot be nested; for example,
    ``array<array<int32>>`` is invalid. Arrays and sets are written with the
    same comma-separated format. Fields of type :py:data:`.tbool` are also
    permitted in `info` and will generate INFO fields of VCF type Flag.

    Hail also exports the name, length, and assembly of each contig as a VCF
    header line, where the assembly is set to the :class:`.ReferenceGenome`
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
    fields in `info` or overwrite existing fields. For example, in order to
    produce an accurate `AC` field, one can run :func:`.variant_qc` and copy
    the `variant_qc.AC` field to `info.AC` as shown below.

    >>> ds = dataset.filter_entries(dataset.GQ >= 20)
    >>> ds = hl.variant_qc(ds)
    >>> ds = ds.annotate_rows(info = ds.info.annotate(AC=ds.variant_qc.AC)) # doctest: +SKIP
    >>> hl.export_vcf(ds, 'output/example.vcf.bgz')

    Warning
    -------
    Do not export to a path that is being read from in the same pipeline.

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
    metadata : :obj:`dict[str, dict[str, dict[str, str]]]`, optional
        Dictionary with information to fill in the VCF header. See
        :func:`get_vcf_metadata` for how this
        dictionary should be structured.

    """

    require_row_key_variant(dataset, 'export_vcf')
    writer = MatrixVCFWriter(output,
                             append_to_header,
                             Env.hail().utils.ExportType.getExportType(parallel),
                             metadata)
    Env.backend().execute(MatrixWrite(dataset._mir, writer))


@typecheck(path=str,
           reference_genome=nullable(reference_genome_type),
           skip_invalid_intervals=bool)
def import_locus_intervals(path, reference_genome='default', skip_invalid_intervals=False) -> Table:
    """Import a locus interval list as a :class:`.Table`.

    Examples
    --------

    Add the row field `capture_region` indicating inclusion in
    at least one locus interval from `capture_intervals.txt`:

    >>> intervals = hl.import_locus_intervals('data/capture_intervals.txt', reference_genome='GRCh37')
    >>> result = dataset.annotate_rows(capture_region = hl.is_defined(intervals[dataset.locus]))

    Notes
    -----

    Hail expects an interval file to contain either one, three or five fields
    per line in the following formats:

    - ``contig:start-end``
    - ``contig  start  end`` (tab-separated)
    - ``contig  start  end  direction  target`` (tab-separated)

    A file in either of the first two formats produces a table with one
    field:

    - **interval** (:class:`.tinterval`) - Row key. Genomic interval. If
      `reference_genome` is defined, the point type of the interval will be
      :class:`.tlocus` parameterized by the `reference_genome`. Otherwise,
      the point type is a :class:`.tstruct` with two fields: `contig` with
      type :py:data:`.tstr` and `position` with type :py:data:`.tint32`.

    A file in the third format (with a "target" column) produces a table with two
    fields:

     - **interval** (:class:`.tinterval`) - Row key. Same schema as above.
     - **target** (:py:data:`.tstr`)

    If `reference_genome` is defined **AND** the file has one field, intervals
    are parsed with :func:`.parse_locus_interval`. See the documentation for
    valid inputs.

    If `reference_genome` is **NOT** defined and the file has one field,
    intervals are parsed with the regex ```"([^:]*):(\\d+)\\-(\\d+)"``
    where contig, start, and end match each of the three capture groups.
    ``start`` and ``end`` match positions inclusively, e.g.
    ``start <= position <= end``.

    For files with three or five fields, ``start`` and ``end`` match positions
    inclusively, e.g. ``start <= position <= end``.

    Parameters
    ----------
    path : :obj:`str`
        Path to file.
    reference_genome : :obj:`str` or :class:`.ReferenceGenome`, optional
        Reference genome to use.
    skip_invalid_intervals : :obj:`bool`
        If ``True`` and `reference_genome` is not ``None``, skip lines with
        intervals that are not consistent with the reference genome.

    Returns
    -------
    :class:`.Table`
        Interval-keyed table.
    """

    t = import_table(path, comment="@", impute=False, no_header=True,
                     types={'f0': tstr, 'f1': tint32, 'f2': tint32,
                            'f3': tstr, 'f4': tstr})

    if t.row.dtype == tstruct(f0=tstr):
        if reference_genome:
            t = t.select(interval=hl.parse_locus_interval(t['f0'],
                                                          reference_genome))
        else:
            interval_regex = r"([^:]*):(\d+)\-(\d+)"

            def checked_match_interval_expr(match):
                return hl.or_missing(hl.len(match) == 3,
                                     locus_interval_expr(match[0],
                                                         hl.int32(match[1]),
                                                         hl.int32(match[2]),
                                                         True,
                                                         True,
                                                         reference_genome,
                                                         skip_invalid_intervals))

            expr = (
                hl.bind(t['f0'].first_match_in(interval_regex),
                        lambda match: hl.cond(hl.bool(skip_invalid_intervals),
                                              checked_match_interval_expr(match),
                                              locus_interval_expr(match[0],
                                                                  hl.int32(match[1]),
                                                                  hl.int32(match[2]),
                                                                  True,
                                                                  True,
                                                                  reference_genome,
                                                                  skip_invalid_intervals))))

            t = t.select(interval=expr)

    elif t.row.dtype == tstruct(f0=tstr, f1=tint32, f2=tint32):
        t = t.select(interval=locus_interval_expr(t['f0'],
                                                  t['f1'],
                                                  t['f2'],
                                                  True,
                                                  True,
                                                  reference_genome,
                                                  skip_invalid_intervals))

    elif t.row.dtype == tstruct(f0=tstr, f1=tint32, f2=tint32, f3=tstr, f4=tstr):
        t = t.select(interval=locus_interval_expr(t['f0'],
                                                  t['f1'],
                                                  t['f2'],
                                                  True,
                                                  True,
                                                  reference_genome,
                                                  skip_invalid_intervals),
                     target=t['f4'])

    else:
        raise FatalError("""invalid interval format.  Acceptable formats:
              'chr:start-end'
              'chr  start  end' (tab-separated)
              'chr  start  end  strand  target' (tab-separated, strand is '+' or '-')""")

    if skip_invalid_intervals and reference_genome:
        t = t.filter(hl.is_defined(t.interval))

    return t.key_by('interval')


@typecheck(path=str,
           reference_genome=nullable(reference_genome_type),
           skip_invalid_intervals=bool)
def import_bed(path, reference_genome='default', skip_invalid_intervals=False) -> Table:
    """Import a UCSC BED file as a :class:`.Table`.

    Examples
    --------

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

    Add the row field `cnv_region` indicating inclusion in
    at least one interval of the three-column BED file:

    >>> bed = hl.import_bed('data/file1.bed', reference_genome='GRCh37')
    >>> result = dataset.annotate_rows(cnv_region = hl.is_defined(bed[dataset.locus]))

    Add a row field `cnv_id` with the value given by the
    fourth column of a BED file:

    >>> bed = hl.import_bed('data/file2.bed')
    >>> result = dataset.annotate_rows(cnv_id = bed[dataset.locus].target)

    Notes
    -----

    The table produced by this method has one of two possible structures. If
    the .bed file has only three fields (`chrom`, `chromStart`, and
    `chromEnd`), then the produced table has only one column:

        - **interval** (:class:`.tinterval`) - Row key. Genomic interval. If
          `reference_genome` is defined, the point type of the interval will be
          :class:`.tlocus` parameterized by the `reference_genome`. Otherwise,
          the point type is a :class:`.tstruct` with two fields: `contig` with
          type :py:data:`.tstr` and `position` with type :py:data:`.tint32`.

    If the .bed file has four or more columns, then Hail will store the fourth
    column as a row field in the table:

        - *interval* (:class:`.tinterval`) - Row key. Genomic interval. Same schema as above.
        - *target* (:py:data:`.tstr`) - Fourth column of .bed file.

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
    reference_genome : :obj:`str` or :class:`.ReferenceGenome`, optional
        Reference genome to use.
    skip_invalid_intervals : :obj:`bool`
        If ``True`` and `reference_genome` is not ``None``, skip lines with
        intervals that are not consistent with the reference genome.

    Returns
    -------
    :class:`.Table`
        Interval-keyed table.
    """

    # UCSC BED spec defined here: https://genome.ucsc.edu/FAQ/FAQformat.html#format1

    t = import_table(path, no_header=True, delimiter=r"\s+", impute=False,
                     skip_blank_lines=True, types={'f0': tstr, 'f1': tint32,
                                                   'f2': tint32, 'f3': tstr,
                                                   'f4': tstr},
                     comment=["""^browser.*""", """^track.*""",
                              r"""^\w+=("[\w\d ]+"|\d+).*"""])

    if t.row.dtype == tstruct(f0=tstr, f1=tint32, f2=tint32):
        t = t.select(interval=locus_interval_expr(t['f0'],
                                                  t['f1'] + 1,
                                                  t['f2'],
                                                  True,
                                                  True,
                                                  reference_genome,
                                                  skip_invalid_intervals))

    elif len(t.row) >= 4 and tstruct(**dict([(n, typ) for n, typ in t.row.dtype._field_types.items()][:4])) == tstruct(f0=tstr, f1=tint32, f2=tint32, f3=tstr):
        t = t.select(interval=locus_interval_expr(t['f0'],
                                                  t['f1'] + 1,
                                                  t['f2'],
                                                  True,
                                                  True,
                                                  reference_genome,
                                                  skip_invalid_intervals),
                     target=t['f3'])

    else:
        raise FatalError("too few fields for BED file: expected 3 or more, but found {}".format(len(t.row)))

    if skip_invalid_intervals and reference_genome:
        t = t.filter(hl.is_defined(t.interval))

    return t.key_by('interval')


@typecheck(path=str,
           quant_pheno=bool,
           delimiter=str,
           missing=str)
def import_fam(path, quant_pheno=False, delimiter=r'\\s+', missing='NA') -> Table:
    """Import a PLINK FAM file into a :class:`.Table`.

    Examples
    --------

    Import a tab-separated
    `FAM file <https://www.cog-genomics.org/plink2/formats#fam>`__
    with a case-control phenotype:

    >>> fam_kt = hl.import_fam('data/case_control_study.fam')

    Import a FAM file with a quantitative phenotype:

    >>> fam_kt = hl.import_fam('data/quantitative_study.fam', quant_pheno=True)

    Notes
    -----

    In Hail, unlike PLINK, the user must *explicitly* distinguish between
    case-control and quantitative phenotypes. Importing a quantitative
    phenotype with ``quant_pheno=False`` will return an error
    (unless all values happen to be `0`, `1`, `2`, or `-9`):

    The resulting :class:`.Table` will have fields, types, and values that are interpreted as missing.

     - *fam_id* (:py:data:`.tstr`) -- Family ID (missing = "0")
     - *id* (:py:data:`.tstr`) -- Sample ID (key column)
     - *pat_id* (:py:data:`.tstr`) -- Paternal ID (missing = "0")
     - *mat_id* (:py:data:`.tstr`) -- Maternal ID (missing = "0")
     - *is_female* (:py:data:`.tstr`) -- Sex (missing = "NA", "-9", "0")

    One of:

     - *is_case* (:py:data:`.tbool`) -- Case-control phenotype (missing = "0", "-9",
       non-numeric or the ``missing`` argument, if given.
     - *quant_pheno* (:py:data:`.tfloat64`) -- Quantitative phenotype (missing = "NA" or
       the ``missing`` argument, if given.

    Warning
    -------
    Hail will interpret the value "-9" as a valid quantitative phenotype, which
    differs from default PLINK behavior. Use ``missing='-9'`` to interpret this
    value as missing.

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
    """
    type_and_data = json.loads(Env.hail().table.Table.importFamJSON(path, quant_pheno, delimiter, missing))
    typ = hl.dtype(type_and_data['type'])
    return hl.Table.parallelize(
        hl.tarray(typ)._convert_from_json_na(type_and_data['data']), typ, key=['id'])


@typecheck(regex=str,
           path=oneof(str, sequenceof(str)),
           max_count=int)
def grep(regex, path, max_count=100):
    r"""Searches given paths for all lines containing regex matches.

        Examples
        --------

        Print all lines containing the string ``hello`` in *file.txt*:

        >>> hl.grep('hello','data/file.txt')

        Print all lines containing digits in *file1.txt* and *file2.txt*:

        >>> hl.grep('\d', ['data/file1.txt','data/file2.txt'])

        Notes
        -----
        :func:`.grep` mimics the basic functionality of Unix ``grep`` in
        parallel, printing results to the screen. This command is provided as a
        convenience to those in the statistical genetics community who often
        search enormous text files like VCFs. Hail uses `Java regular expression
        patterns
        <https://docs.oracle.com/javase/8/docs/api/java/util/regex/Pattern.html>`__.
        The `RegExr sandbox <http://regexr.com/>`__ may be helpful.

        Parameters
        ----------
        regex : :obj:`str`
            The regular expression to match.
        path : :obj:`str` or :obj:`list` of :obj:`str`
            The files to search.
        max_count : :obj:`int`
            The maximum number of matches to return
        """
    Env.hc()._jhc.grep(regex, jindexed_seq_args(path), max_count)


@typecheck(path=oneof(str, sequenceof(str)),
           sample_file=nullable(str),
           entry_fields=sequenceof(enumeration('GT', 'GP', 'dosage')),
           n_partitions=nullable(int),
           block_size=nullable(int),
           index_file_map=nullable(dictof(str, str)),
           variants=nullable(oneof(sequenceof(hl.utils.Struct), sequenceof(hl.genetics.Locus),
                                   StructExpression, LocusExpression, Table)),
           _row_fields=sequenceof(enumeration('varid', 'rsid')))
def import_bgen(path,
                entry_fields,
                sample_file=None,
                n_partitions=None,
                block_size=None,
                index_file_map=None,
                variants=None,
                _row_fields=['varid', 'rsid']) -> MatrixTable:
    """Import BGEN file(s) as a :class:`.MatrixTable`.

    Examples
    --------

    Import a BGEN file as a matrix table with GT and GP entry fields:

    >>> ds_result = hl.import_bgen("data/example.8bits.bgen",
    ...                            entry_fields=['GT', 'GP'],
    ...                            sample_file="data/example.8bits.sample")

    Import a BGEN file as a matrix table with genotype dosage entry field:

    >>> ds_result = hl.import_bgen("data/example.8bits.bgen",
    ...                             entry_fields=['dosage'],
    ...                             sample_file="data/example.8bits.sample")

    Load a single variant from a BGEN file:

    >>> ds_result = hl.import_bgen("data/example.8bits.bgen",
    ...                            entry_fields=['dosage'],
    ...                            sample_file="data/example.8bits.sample",
    ...                            variants=[hl.eval(hl.parse_variant('1:2000:A:G'))])

    Load a set of variants specified by a table expression from a BGEN file:

    >>> variants = hl.import_table("data/bgen-variants.txt")
    >>> variants = variants.annotate(v=hl.parse_variant(variants.v)).key_by('v')
    >>> ds_result = hl.import_bgen("data/example.8bits.bgen",
    ...                            entry_fields=['dosage'],
    ...                            sample_file="data/example.8bits.sample",
    ...                            variants=variants.v)

    Load a set of variants specified by a table keyed by 'locus' and 'alleles' from a BGEN file:

    >>> ds_result = hl.import_bgen("data/example.8bits.bgen",
    ...                            entry_fields=['dosage'],
    ...                            sample_file="data/example.8bits.sample",
    ...                            variants=variants_table)

    Notes
    -----

    Hail supports importing data from v1.2 of the `BGEN file format
    <http://www.well.ox.ac.uk/~gav/bgen_format/bgen_format.html>`__.
    Genotypes must be **unphased** and **diploid**, genotype
    probabilities must be stored with 8 bits, and genotype probability
    blocks must be compressed with zlib or uncompressed. All variants
    must be bi-allelic.

    Each BGEN file must have a corresponding index file, which can be generated
    with :func:`.index_bgen`. All files must have been indexed with the same
    reference genome. To load multiple files at the same time,
    use :ref:`Hadoop Glob Patterns <sec-hadoop-glob>`.

    If n_partitions and block_size are both specified, block_size is
    used. If neither are specified, the default is a 128MB block
    size.

    **Column Fields**

    - `s` (:py:data:`.tstr`) -- Column key. This is the sample ID imported
      from the first column of the sample file if given. Otherwise, the sample
      ID is taken from the sample identifying block in the first BGEN file if it
      exists; else IDs are assigned from `_0`, `_1`, to `_N`.

    **Row Fields**

    Between two and four row fields are created. The `locus` and `alleles` are
    always included. `_row_fields` determines if `varid` and `rsid` are also
    included. For best performance, only include fields necessary for your
    analysis. NOTE: the `_row_fields` parameter is considered an experimental
    feature and may be removed without warning.

    - `locus` (:class:`.tlocus` or :class:`.tstruct`) -- Row key. The chromosome
      and position. If `reference_genome` is defined, the type will be
      :class:`.tlocus` parameterized by `reference_genome`. Otherwise, the type
      will be a :class:`.tstruct` with two fields: `contig` with type
      :py:data:`.tstr` and `position` with type :py:data:`.tint32`.
    - `alleles` (:class:`.tarray` of :py:data:`.tstr`) -- Row key. An
      array containing the alleles of the variant. The reference
      allele is the first element in the array.
    - `varid` (:py:data:`.tstr`) -- The variant identifier. The third field in
      each variant identifying block.
    - `rsid` (:py:data:`.tstr`) -- The rsID for the variant. The fifth field in
      each variant identifying block.

    **Entry Fields**

    Up to three entry fields are created, as determined by
    `entry_fields`.  For best performance, include precisely those
    fields required for your analysis. It is also possible to pass an
    empty tuple or list for `entry_fields`, which can greatly
    accelerate processing speed if your workflow does not use the
    genotype data.

    - `GT` (:py:data:`.tcall`) -- The hard call corresponding to the genotype with
      the greatest probability. If there is not a unique maximum probability, the
      hard call is set to missing.
    - `GP` (:class:`.tarray` of :py:data:`.tfloat64`) -- Genotype probabilities
      as defined by the BGEN file spec. For bi-allelic variants, the array has
      three elements giving the probabilities of homozygous reference,
      heterozygous, and homozygous alternate genotype, in that order.
    - `dosage` (:py:data:`.tfloat64`) -- The expected value of the number of
      alternate alleles, given by the probability of heterozygous genotype plus
      twice the probability of homozygous alternate genotype. All variants must
      be bi-allelic.

    See Also
    --------
    :func:`.index_bgen`

    Parameters
    ----------
    path : :obj:`str` or :obj:`list` of :obj:`str`
        BGEN file(s) to read.
    entry_fields : :obj:`list` of :obj:`str`
        List of entry fields to create.
        Options: ``'GT'``, ``'GP'``, ``'dosage'``.
    sample_file : :obj:`str`, optional
        Sample file to read the sample ids from. If specified, the number of
        samples in the file must match the number in the BGEN file(s).
    n_partitions : :obj:`int`, optional
        Number of partitions.
    block_size : :obj:`int`, optional
        Block size, in MB.
    index_file_map : :obj:`dict` of :obj:`str` to :obj:`str`, optional
        Dict of BGEN file to index file location. Cannot use Hadoop glob
        patterns in file names.
    variants : :class:`.StructExpression` or :class:`.LocusExpression` or :obj:`list` of :class:`.Struct` or :obj:`list` of :class:`.Locus` or :class:`.Table`
        Variants to filter to. The underlying type of the input (row key in the case of a :class:`.Table`)
        must be either a :class:`.tlocus`, a struct with one field `locus`, or a struct with two fields:
        `locus` and `alleles`. The type of `locus` can either be a :class:`.tlocus` or a :class:`.tstruct`
        with two fields: `contig` of type :class:`.tstr` and `position` of type :class:`.tint`. If the
        type of `locus` is :class:`.tlocus`, the reference genome must match that used to index the BGEN
        file(s). The type of `alleles` is a :class:`.tarray` of :class:`.tstr`.
    _row_fields : :obj:`list` of :obj:`str`
        List of non-key row fields to create.
        Options: ``'varid'``, ``'rsid'``

    Returns
    -------
    :class:`.MatrixTable`

    """

    if n_partitions is None and block_size is None:
        block_size = 128

    if index_file_map is None:
        index_file_map = {}

    entry_set = set(entry_fields)
    row_set = set(_row_fields)

    if variants is not None:
        mt_type = Env.backend().matrix_type(
            MatrixRead(MatrixBGENReader(path, sample_file, index_file_map, n_partitions, block_size, None)))
        lt = mt_type.row_type['locus']

        expected_vtype = tstruct(locus=lt, alleles=tarray(tstr))

        if isinstance(variants, StructExpression) or isinstance(variants, LocusExpression):
            if isinstance(variants, LocusExpression):
                variants = hl.struct(locus=variants)

            if len(variants.dtype) == 0 or not variants.dtype._is_prefix_of(expected_vtype):
                raise TypeError("'import_bgen' requires the expression type for 'variants' is a non-empty prefix of the BGEN key type: \n" +
                                 f"\tFound: {repr(variants.dtype)}\n" +
                                  f"\tExpected: {repr(expected_vtype)}\n")

            uid = Env.get_uid()
            fnames = list(variants.dtype)
            variants = variants._to_table(uid) # This will add back the other key fields of the source, which we don't want
            variants = variants.key_by(**{fname: variants[uid][fname] for fname in fnames})
            variants = variants.select()
        elif isinstance(variants, Table):
            if len(variants.key) == 0 or not variants.key.dtype._is_prefix_of(expected_vtype):
                raise TypeError("'import_bgen' requires the row key type for 'variants' is a non-empty prefix of the BGEN key type: \n" +
                                 f"\tFound: {repr(variants.key.dtype)}\n" +
                                  f"\tExpected: {repr(expected_vtype)}\n")
            variants = variants.select()
        else:
            assert isinstance(variants, list)
            try:
                if len(variants) == 0:
                    variants = hl.Table.parallelize(variants,
                                                    schema=expected_vtype,
                                                    key=['locus', 'alleles'])
                else:
                    first_v = variants[0]
                    if isinstance(first_v, hl.Locus):
                        variants = hl.Table.parallelize([hl.Struct(locus=v) for v in variants],
                                                        schema=hl.tstruct(locus=lt),
                                                        key='locus')
                    else:
                        assert isinstance(first_v, hl.utils.Struct)
                        if len(first_v) == 1:
                            variants = hl.Table.parallelize(variants,
                                                            schema=hl.tstruct(locus=lt),
                                                            key='locus')
                        else:
                            variants = hl.Table.parallelize(variants,
                                                            schema=expected_vtype,
                                                            key=['locus', 'alleles'])
            except:
                raise TypeError(f"'import_bgen' requires all elements in 'variants' are a non-empty prefix of the BGEN key type: {repr(expected_vtype)}")

    reader = MatrixBGENReader(path, sample_file, index_file_map, n_partitions, block_size, variants)

    mt = (MatrixTable(MatrixRead(reader))
          .drop(*[fd for fd in ['GT', 'GP', 'dosage'] if fd not in entry_set],
                *[fd for fd in ['rsid', 'varid', 'offset', 'file_idx'] if fd not in row_set]))

    return mt


@typecheck(path=oneof(str, sequenceof(str)),
           sample_file=nullable(str),
           tolerance=numeric,
           min_partitions=nullable(int),
           chromosome=nullable(str),
           reference_genome=nullable(reference_genome_type),
           contig_recoding=nullable(dictof(str, str)),
           skip_invalid_loci=bool)
def import_gen(path,
               sample_file=None,
               tolerance=0.2,
               min_partitions=None,
               chromosome=None,
               reference_genome='default',
               contig_recoding=None,
               skip_invalid_loci=False) -> MatrixTable:
    """
    Import GEN file(s) as a :class:`.MatrixTable`.

    Examples
    --------

    >>> ds = hl.import_gen('data/example.gen',
    ...                    sample_file='data/example.sample',
    ...                    reference_genome='GRCh37')

    Notes
    -----

    For more information on the GEN file format, see `here
    <http://www.stats.ox.ac.uk/%7Emarchini/software/gwas/file_format.html#mozTocId40300>`__.

    If the GEN file has only 5 columns before the start of the genotype
    probability data (chromosome field is missing), you must specify the
    chromosome using the `chromosome` parameter.

    To load multiple files at the same time, use :ref:`Hadoop Glob Patterns
    <sec-hadoop-glob>`.

    **Column Fields**

    - `s` (:py:data:`.tstr`) -- Column key. This is the sample ID imported
      from the first column of the sample file.

    **Row Fields**

    - `locus` (:class:`.tlocus` or :class:`.tstruct`) -- Row key. The genomic
      location consisting of the chromosome (1st column if present, otherwise
      given by `chromosome`) and position (3rd column if `chromosome` is not
      defined). If `reference_genome` is defined, the type will be
      :class:`.tlocus` parameterized by `reference_genome`. Otherwise, the type
      will be a :class:`.tstruct` with two fields: `contig` with type
      :py:data:`.tstr` and `position` with type :py:data:`.tint32`.
    - `alleles` (:class:`.tarray` of :py:data:`.tstr`) -- Row key. An array
      containing the alleles of the variant. The reference allele (4th column if
      `chromosome` is not defined) is the first element of the array and the
      alternate allele (5th column if `chromosome` is not defined) is the second
      element.
    - `varid` (:py:data:`.tstr`) -- The variant identifier. 2nd column of GEN
      file if chromosome present, otherwise 1st column.
    - `rsid` (:py:data:`.tstr`) -- The rsID. 3rd column of GEN file if
      chromosome present, otherwise 2nd column.

    **Entry Fields**

    - `GT` (:py:data:`.tcall`) -- The hard call corresponding to the genotype with
      the highest probability.
    - `GP` (:class:`.tarray` of :py:data:`.tfloat64`) -- Genotype probabilities
      as defined by the GEN file spec. The array is set to missing if the
      sum of the probabilities is a distance greater than the `tolerance`
      parameter from 1.0. Otherwise, the probabilities are normalized to sum to
      1.0. For example, the input ``[0.98, 0.0, 0.0]`` will be normalized to
      ``[1.0, 0.0, 0.0]``.

    Parameters
    ----------
    path : :obj:`str` or :obj:`list` of :obj:`str`
        GEN files to import.
    sample_file : :obj:`str`
        Sample file to import.
    tolerance : :obj:`float`
        If the sum of the genotype probabilities for a genotype differ from 1.0
        by more than the tolerance, set the genotype to missing.
    min_partitions : :obj:`int`, optional
        Number of partitions.
    chromosome : :obj:`str`, optional
        Chromosome if not included in the GEN file
    reference_genome : :obj:`str` or :class:`.ReferenceGenome`, optional
        Reference genome to use.
    contig_recoding : :obj:`dict` of :obj:`str` to :obj:`str`, optional
        Dict of old contig name to new contig name. The new contig name must be
        in the reference genome given by `reference_genome`.
    skip_invalid_loci : :obj:`bool`
        If ``True``, skip loci that are not consistent with `reference_genome`.

    Returns
    -------
    :class:`.MatrixTable`
    """
    path = wrap_to_list(path)
    rg = reference_genome.name if reference_genome else None
    if contig_recoding is None:
        contig_recoding = {}
    return MatrixTable(MatrixRead(MatrixGENReader(
        path, sample_file, chromosome, min_partitions, tolerance, rg, contig_recoding, skip_invalid_loci)))


@typecheck(paths=oneof(str, sequenceof(str)),
           key=table_key_type,
           min_partitions=nullable(int),
           impute=bool,
           no_header=bool,
           comment=oneof(str, sequenceof(str)),
           delimiter=str,
           missing=oneof(str, sequenceof(str)),
           types=dictof(str, hail_type),
           quote=nullable(char),
           skip_blank_lines=bool,
           force_bgz=bool,
           filter=nullable(str),
           find_replace=nullable(sized_tupleof(str, str)),
           force=bool)
def import_table(paths,
                 key=None,
                 min_partitions=None,
                 impute=False,
                 no_header=False,
                 comment=(),
                 delimiter="\t",
                 missing="NA",
                 types={},
                 quote=None,
                 skip_blank_lines=False,
                 force_bgz=False,
                 filter=None,
                 find_replace=None,
                 force=False) -> Table:
    """Import delimited text file (text table) as :class:`.Table`.

    The resulting :class:`.Table` will have no key fields. Use
    :meth:`.Table.key_by` to specify keys.

    Examples
    --------

    Consider this file:

    .. code-block:: text

        $ cat data/samples1.tsv
        Sample     Height  Status  Age
        PT-1234    154.1   ADHD    24
        PT-1236    160.9   Control 19
        PT-1238    NA      ADHD    89
        PT-1239    170.3   Control 55

    The field ``Height`` contains floating-point numbers and the field ``Age``
    contains integers.

    To import this table using field types:

    >>> table = hl.import_table('data/samples1.tsv',
    ...                              types={'Height': hl.tfloat64, 'Age': hl.tint32})

    Note ``Sample`` and ``Status`` need no type, because :py:data:`.tstr` is
    the default type.

    To import a table using type imputation (which causes the file to be parsed
    twice):

    >>> table = hl.import_table('data/samples1.tsv', impute=True)

    **Detailed examples**

    Let's import fields from a CSV file with missing data and special characters:

    .. code-block:: text

        $ cat data/samples2.tsv
        Batch,PT-ID
        1kg,PT-0001
        1kg,PT-0002
        study1,PT-0003
        study3,PT-0003
        .,PT-0004
        1kg,PT-0005
        .,PT-0006
        1kg,PT-0007

    In this case, we should:

    - Pass the non-default delimiter ``,``

    - Pass the non-default missing value ``.``

    >>> table = hl.import_table('data/samples2.tsv', delimiter=',', missing='.')

    Let's import a table from a file with no header and sample IDs that need to
    be transformed.  Suppose the sample IDs are of the form ``NA#####``. This
    file has no header line, and the sample ID is hidden in a field with other
    information.

    .. code-block: text

        $ cat data/samples3.tsv
        1kg_NA12345   female
        1kg_NA12346   male
        1kg_NA12348   female
        pgc_NA23415   male
        pgc_NA23418   male

    To import:

    >>> t = hl.import_table('data/samples3.tsv', no_header=True)
    >>> t = t.annotate(sample = t.f0.split("_")[1]).key_by('sample')

    Notes
    -----

    The `impute` parameter tells Hail to scan the file an extra time to gather
    information about possible field types. While this is a bit slower for large
    files because the file is parsed twice, the convenience is often worth this
    cost.

    The `delimiter` parameter is either a delimiter character (if a single
    character) or a field separator regex (2 or more characters). This regex
    follows the `Java regex standard
    <http://docs.oracle.com/javase/7/docs/api/java/util/regex/Pattern.html>`_.

    .. note::

        Use ``delimiter='\\s+'`` to specify whitespace delimited files.

    If set, the `comment` parameter causes Hail to skip any line that starts
    with the given string(s). For example, passing ``comment='#'`` will skip any
    line beginning in a pound sign. If the string given is a single character,
    Hail will skip any line beginning with the character. Otherwise if the
    length of the string is greater than 1, Hail will interpret the string as a
    regex and will filter out lines matching the regex. For example, passing
    ``comment=['#', '^track.*']`` will filter out lines beginning in a pound sign
    and any lines that match the regex ``'^track.*'``.

    The `missing` parameter defines the representation of missing data in the table.

    .. note::

        The `missing` parameter is **NOT** a regex. The `comment` parameter is
        treated as a regex **ONLY** if the length of the string is greater than
        1 (not a single character).

    The `no_header` parameter indicates that the file has no header line. If
    this option is passed, then the field names will be `f0`, `f1`,
    ... `fN` (0-indexed).

    The `types` parameter allows the user to pass the types of fields in the
    table. It is an :obj:`dict` keyed by :obj:`str`, with :class:`.HailType` values.
    See the examples above for a standard usage. Additionally, this option can
    be used to override type imputation. For example, if the field
    ``Chromosome`` only contains the values ``1`` through ``22``, it will be
    imputed to have type :py:data:`.tint32`, whereas most Hail methods expect
    that a chromosome field will be of type :py:data:`.tstr`. Setting
    ``impute=True`` and ``types={'Chromosome': hl.tstr}`` solves this problem.

    Parameters
    ----------

    paths : :obj:`str` or :obj:`list` of :obj:`str`
        Files to import.
    key : :obj:`str` or :obj:`list` of :obj:`str`
        Key fields(s).
    min_partitions : :obj:`int` or :obj:`None`
        Minimum number of partitions.
    no_header : :obj:`bool`
        If ``True```, assume the file has no header and name the N fields `f0`,
        `f1`, ... `fN` (0-indexed).
    impute : :obj:`bool`
        If ``True``, Impute field types from the file.
    comment : :obj:`str` or :obj:`list` of :obj:`str`
        Skip lines beginning with the given string if the string is a single
        character. Otherwise, skip lines that match the regex specified.
    delimiter : :obj:`str`
        Field delimiter regex.
    missing : :obj:`str` or :obj:`List[str]`
        Identifier(s) to be treated as missing.
    types : :obj:`dict` mapping :obj:`str` to :class:`.HailType`
        Dictionary defining field types.
    quote : :obj:`str` or :obj:`None`
        Quote character.
    skip_blank_lines : :obj:`bool`
        If ``True``, ignore empty lines. Otherwise, throw an error if an empty
        line is found.
    force_bgz : :obj:`bool`
        If ``True``, load files as blocked gzip files, assuming
        that they were actually compressed using the BGZ codec. This option is
        useful when the file extension is not ``'.bgz'``, but the file is
        blocked gzip, so that the file can be read in parallel and not on a
        single node.
    filter : :obj:`str`, optional
        Line filter regex. A partial match results in the line being removed
        from the file. Applies before `find_replace`, if both are defined.
    find_replace : (:obj:`str`, :obj:`str`)
        Line substitution regex. Functions like ``re.sub``, but obeys the exact
        semantics of Java's
        `String.replaceAll <https://docs.oracle.com/javase/8/docs/api/java/lang/String.html#replaceAll-java.lang.String-java.lang.String->`__.
    force : :obj:`bool`
        If ``True``, load gzipped files serially on one core. This should
        be used only when absolutely necessary, as processing time will be
        increased due to lack of parallelism.

    Returns
    -------
    :class:`.Table`
    """
    paths = wrap_to_list(paths)
    comment = wrap_to_list(comment)
    missing = wrap_to_list(missing)

    tr = TextTableReader(paths, min_partitions, types, comment,
                         delimiter, missing, no_header, impute, quote,
                         skip_blank_lines, force_bgz, filter, find_replace,
                         force)
    t = Table(TableRead(tr))
    if key:
        key = wrap_to_list(key)
        t = t.key_by(*key)
    return t


@typecheck(paths=oneof(str, sequenceof(str)),
           row_fields=dictof(str, hail_type),
           row_key=oneof(str, sequenceof(str)),
           entry_type=enumeration(tint32, tint64, tfloat32, tfloat64, tstr),
           missing=str,
           min_partitions=nullable(int),
           no_header=bool,
           force_bgz=bool,
           sep=str)
def import_matrix_table(paths,
                        row_fields={},
                        row_key=[],
                        entry_type=tint32,
                        missing="NA",
                        min_partitions=None,
                        no_header=False,
                        force_bgz=False,
                        sep='\t') -> MatrixTable:
    """
    Import tab-delimited file(s) as a :class:`.MatrixTable`.

    Examples
    --------

        Consider the following file containing counts from a RNA sequencing
        dataset:

    .. code-block:: text

        $ cat data/matrix1.tsv
        Barcode Tissue  Days    GENE1   GENE2   GENE3   GENE4
        TTAGCCA brain   1.0     0       0       1       0
        ATCACTT kidney  5.5     3       0       2       0
        CTCTTCT kidney  2.5     0       0       0       1
        CTATATA brain   7.0     0       0       3       0

    The field ``Height`` contains floating-point numbers and the field ``Age``
    contains integers.

    To import this matrix:

    >>> matrix1 = hl.import_matrix_table('data/matrix1.tsv',
    ...                                  row_fields={'Barcode': hl.tstr, 'Tissue': hl.tstr, 'Days':hl.tfloat32},
    ...                                  row_key='Barcode')
    >>> matrix1.describe()  # doctest: +NOTEST
    ----------------------------------------
    Global fields:
        None
    ----------------------------------------
    Column fields:
        'col_id': str
    ----------------------------------------
    Row fields:
        'Barcode': str
        'Tissue': str
        'Days': float32
    ----------------------------------------
    Entry fields:
        'x': int32
    ----------------------------------------
    Column key:
        'col_id': str
    Row key:
        'Barcode': str
    ----------------------------------------

    In this example, the header information is missing for the row fields, but
    the column IDs are still present:

    .. code-block:: text

        $ cat data/matrix2.tsv
        GENE1   GENE2   GENE3   GENE4
        TTAGCCA brain   1.0     0       0       1       0
        ATCACTT kidney  5.5     3       0       2       0
        CTCTTCT kidney  2.5     0       0       0       1
        CTATATA brain   7.0     0       0       3       0

    The row fields get imported as `f0`, `f1`, and `f2`, so we need to do:

    >>> matrix2 = hl.import_matrix_table('data/matrix2.tsv',
    ...                                  row_fields={'f0': hl.tstr, 'f1': hl.tstr, 'f2':hl.tfloat32},
    ...                                  row_key='f0')
    >>> matrix2.rename({'f0': 'Barcode', 'f1': 'Tissue', 'f2': 'Days'})

    Sometimes, the header and row information is missing completely:

    .. code-block:: text

        $ cat data/matrix3.tsv
        0       0       1       0
        3       0       2       0
        0       0       0       1
        0       0       3       0

    >>> matrix3 = hl.import_matrix_table('data/matrix3.tsv', no_header=True)

    In this case, the file has no row fields, so we use the default
    row index as a key for the imported matrix table.

    Notes
    -----

    The resulting matrix table has the following structure:

        * The row fields are named as specified in the column header. If they
          are missing from the header or ``no_header=True``, row field names are
          set to the strings `f0`, `f1`, ... (0-indexed) in column order. The types
          of all row fields must be specified in the `row_fields` argument.
        * The row key is taken from the `row_key` argument, and must be a
          subset of row fields. If left empty, the row key will be a new row field
          `row_idx` of type :obj:`int`, whose values 0, 1, ... index the original
          rows of the matrix.
        * There is one column field, **col_id**, which is a key field of type
          :obj:str or :obj:int. By default, its values are the strings given by
          the corresponding column names in the header line. If ``no_header=True``,
          column IDs are set to integers 0, 1, ... (also 0-indexed) in column
          order.
        * There is one entry field, **x**, that contains the data from the imported
          matrix.


    All columns to be imported as row fields must be at the start of the row.

    Unlike :func:`import_table`, no type imputation is done so types must be specified
    for all columns that should be imported as row fields. (The other columns are
    imported as entries in the matrix.)

    The header information for row fields is allowed to be missing, if the
    column IDs are present, but the header must then consist only of tab-delimited
    column IDs (no row field names).

    Parameters
    ----------
    paths: :obj:`str` or :obj:`list` of :obj:`str`
        Files to import.
    row_fields: :obj:`dict` of :obj:`str` to :class:`.HailType`
        Columns to take as row fields in the MatrixTable. They must be located
        before all entry columns.
    row_key: :obj:`str` or :obj:`list` of :obj:`str`
        Key fields(s). If empty, creates an index `row_id` to use as key.
    entry_type: :class:`.HailType`
        Type of entries in matrix table. Must be one of: :py:data:`.tint32`,
        :py:data:`.tint64`, :py:data:`.tfloat32`, :py:data:`.tfloat64`, or
        :py:data:`.tstr`. Default: :py:data:`.tint32`.
    missing: :obj:`str`
        Identifier to be treated as missing. Default: NA
    min_partitions: :obj:`int` or :obj:`None`
        Minimum number of partitions.
    no_header: :obj:`bool`
        If ``True``, assume the file has no header and name the row fields `f0`,
        `f1`, ... `fK` (0-indexed) and the column keys 0, 1, ... N.
    force_bgz : :obj:`bool`
        If ``True``, load **.gz** files as blocked gzip files, assuming
        that they were actually compressed using the BGZ codec.

    Returns
    -------
    :class:`.MatrixTable`
        MatrixTable constructed from imported data
    """

    paths = wrap_to_list(paths)
    jrow_fields = {k: v._parsable_string() for k, v in row_fields.items()}
    for k, v in row_fields.items():
        if v not in {tint32, tint64, tfloat32, tfloat64, tstr}:
            raise FatalError("""import_matrix_table expects field types to be one of:
            'int32', 'int64', 'float32', 'float64', 'str': field {} had type '{}'""".format(repr(k), v))
    row_key = wrap_to_list(row_key)
    if entry_type not in {tint32, tint64, tfloat32, tfloat64, tstr}:
        raise FatalError("""import_matrix_table expects entry types to be one of:
        'int32', 'int64', 'float32', 'float64', 'str': found '{}'""".format(entry_type))

    if len(sep) != 1:
        raise FatalError('sep must be a single character')

    return MatrixTable._from_java(
        Env.hc()._jhc.importMatrix(paths, jrow_fields, row_key, entry_type._parsable_string(), missing, joption(min_partitions),
                                   no_header, force_bgz, sep))


@typecheck(bed=str,
           bim=str,
           fam=str,
           min_partitions=nullable(int),
           delimiter=str,
           missing=str,
           quant_pheno=bool,
           a2_reference=bool,
           reference_genome=nullable(reference_genome_type),
           contig_recoding=nullable(dictof(str, str)),
           skip_invalid_loci=bool)
def import_plink(bed, bim, fam,
                 min_partitions=None,
                 delimiter='\\\\s+',
                 missing='NA',
                 quant_pheno=False,
                 a2_reference=True,
                 reference_genome='default',
                 contig_recoding=None,
                 skip_invalid_loci=False) -> MatrixTable:
    """Import a PLINK dataset (BED, BIM, FAM) as a :class:`.MatrixTable`.

    Examples
    --------

    >>> ds = hl.import_plink(bed='data/test.bed',
    ...                      bim='data/test.bim',
    ...                      fam='data/test.fam',
    ...                      reference_genome='GRCh37')

    Notes
    -----

    Only binary SNP-major mode files can be read into Hail. To convert your
    file from individual-major mode to SNP-major mode, use PLINK to read in
    your fileset and use the ``--make-bed`` option.

    Hail uses the individual ID (column 2 in FAM file) as the sample id (`s`).
    The individual IDs must be unique.

    The resulting :class:`.MatrixTable` has the following fields:

    * Row fields:

        * `locus` (:class:`.tlocus` or :class:`.tstruct`) -- Row key. The
          chromosome and position. If `reference_genome` is defined, the type
          will be :class:`.tlocus` parameterized by `reference_genome`.
          Otherwise, the type will be a :class:`.tstruct` with two fields:
          `contig` with type :py:data:`.tstr` and `position` with type
          :py:data:`.tint32`.
        * `alleles` (:class:`.tarray` of :py:data:`.tstr`) -- Row key. An
          array containing the alleles of the variant. The reference allele (A2
          if `a2_reference` is ``True``) is the first element in the array.
        * `rsid` (:py:data:`.tstr`) -- Column 2 in the BIM file.
        * `cm_position` (:py:data:`.tfloat64`) -- Column 3 in the BIM file,
          the position in centimorgans.

    * Column fields:

        * `s` (:py:data:`.tstr`) -- Column 2 in the Fam file (key field).
        * `fam_id` (:py:data:`.tstr`) -- Column 1 in the FAM file. Set to
          missing if ID equals "0".
        * `pat_id` (:py:data:`.tstr`) -- Column 3 in the FAM file. Set to
          missing if ID equals "0".
        * `mat_id` (:py:data:`.tstr`) -- Column 4 in the FAM file. Set to
          missing if ID equals "0".
        * `is_female` (:py:data:`.tstr`) -- Column 5 in the FAM file. Set to
          missing if value equals "-9", "0", or "N/A". Set to true if value
          equals "2". Set to false if value equals "1".
        * `is_case` (:py:data:`.tstr`) -- Column 6 in the FAM file. Only
          present if `quant_pheno` equals False. Set to missing if value equals
          "-9", "0", "N/A", or the value specified by `missing`. Set to true if
          value equals "2". Set to false if value equals "1".
        * `quant_pheno` (:py:data:`.tstr`) -- Column 6 in the FAM file. Only
          present if `quant_pheno` equals True. Set to missing if value equals
          `missing`.

    * Entry fields:

        * `GT` (:py:data:`.tcall`) -- Genotype call (diploid, unphased).

    Warning
    -------
    Hail will interpret the value "-9" as a valid quantitative phenotype, which
    differs from default PLINK behavior. Use ``missing='-9'`` to interpret this
    value as missing.

    Parameters
    ----------
    bed : :obj:`str`
        PLINK BED file.

    bim : :obj:`str`
        PLINK BIM file.

    fam : :obj:`str`
        PLINK FAM file.

    min_partitions : :obj:`int`, optional
        Number of partitions.

    missing : :obj:`str`
        String used to denote missing values **only** for the phenotype field.
        This is in addition to "-9", "0", and "N/A" for case-control
        phenotypes.

    delimiter : :obj:`str`
        FAM file field delimiter regex.

    quant_pheno : :obj:`bool`
        If ``True``, FAM phenotype is interpreted as quantitative.

    a2_reference : :obj:`bool`
        If ``True``, A2 is treated as the reference allele. If False, A1 is treated
        as the reference allele.

    reference_genome : :obj:`str` or :class:`.ReferenceGenome`, optional
        Reference genome to use.

    contig_recoding : :obj:`dict` of :obj:`str` to :obj:`str`, optional
        Dict of old contig name to new contig name. The new contig name must be
        in the reference genome given by ``reference_genome``. If ``None``, the
        default is dependent on the ``reference_genome``. For "GRCh37", the default
        is ``{'23': 'X', '24': 'Y', '25': 'X', '26': 'MT'}``. For "GRCh38", the
        default is ``{'1': 'chr1', ..., '22': 'chr22', '23': 'chrX', '24': 'chrY', '25': 'chrX', '26': 'chrM'}``.

    skip_invalid_loci : :obj:`bool`
        If ``True``, skip loci that are not consistent with `reference_genome`.

    Returns
    -------
    :class:`.MatrixTable`
    """

    if contig_recoding is None:
        if reference_genome is None:
            contig_recoding = {}
        elif reference_genome.name == "GRCh37":
            contig_recoding = {'23': 'X', '24': 'Y', '25': 'X', '26': 'MT'}
        elif reference_genome.name == "GRCh38":
            contig_recoding = {**{str(i):f'chr{i}' for i in range(1, 23)},
                               **{'23': 'chrX', '24': 'chrY', '25': 'chrX', '26': 'chrM'}}
        else:
            contig_recoding = {}

    reader = MatrixPLINKReader(bed, bim, fam, min_partitions, missing, delimiter,
                               quant_pheno, a2_reference, reference_genome, contig_recoding,
                               skip_invalid_loci)
    return MatrixTable(MatrixRead(reader, drop_cols=False, drop_rows=False))


@typecheck(path=oneof(str, sequenceof(str)),
           _drop_cols=bool,
           _drop_rows=bool)
def read_matrix_table(path, _drop_cols=False, _drop_rows=False) -> MatrixTable:
    """Read in a :class:`.MatrixTable` written with :meth:`.MatrixTable.write`.

    Parameters
    ----------
    path : :obj:`str`
        File to read.

    Returns
    -------
    :class:`.MatrixTable`
    """
    return MatrixTable(MatrixRead(MatrixNativeReader(path), _drop_cols, _drop_rows))


@typecheck(path=str)
def get_vcf_metadata(path):
    """Extract metadata from VCF header.

    Examples
    --------

    >>> hl.get_vcf_metadata('data/example2.vcf.bgz')  # doctest: +NOTEST
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

    which can be used with :func:`.export_vcf` to fill in the relevant fields in the header.

    Parameters
    ----------
    path : :obj:`str`
        VCF file(s) to read. If more than one file is given, the first
        file is used.

    Returns
    -------
    :obj:`dict` of :obj:`str` to (:obj:`dict` of :obj:`str` to (:obj:`dict` of :obj:`str` to :obj:`str`))
    """

    return Env.backend().parse_vcf_metadata(path)


@typecheck(path=oneof(str, sequenceof(str)),
           force=bool,
           force_bgz=bool,
           header_file=nullable(str),
           min_partitions=nullable(int),
           drop_samples=bool,
           call_fields=oneof(str, sequenceof(str)),
           reference_genome=nullable(reference_genome_type),
           contig_recoding=nullable(dictof(str, str)),
           array_elements_required=bool,
           skip_invalid_loci=bool,
           entry_float_type=enumeration(tfloat32, tfloat64),
           filter=nullable(str),
           find_replace=nullable(sized_tupleof(str, str)),
            # json
           _partitions=nullable(str))
def import_vcf(path,
               force=False,
               force_bgz=False,
               header_file=None,
               min_partitions=None,
               drop_samples=False,
               call_fields=['PGT'],
               reference_genome='default',
               contig_recoding=None,
               array_elements_required=True,
               skip_invalid_loci=False,
               entry_float_type=tfloat64,
               filter=None,
               find_replace=None,
               _partitions=None) -> MatrixTable:
    """Import VCF file(s) as a :class:`.MatrixTable`.

    Examples
    --------

    >>> ds = hl.import_vcf('data/example2.vcf.bgz', reference_genome='GRCh37')

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
        ``filters`` row field. This annotation is a ``set<str>`` and can be
        queried for filter membership with expressions like
        ``ds.filters.contains("VQSRTranche99.5...")``. Variants that are flagged
        as "PASS" will have no filters applied; for these variants,
        ``hl.len(ds.filters)`` is ``0``. Thus, filtering to PASS variants
        can be done with :meth:`.MatrixTable.filter_rows` as follows:

        >>> pass_ds = dataset.filter_rows(hl.len(dataset.filters) == 0)

    **Column Fields**

    - `s` (:py:data:`.tstr`) -- Column key. This is the sample ID.

    **Row Fields**

    - `locus` (:class:`.tlocus` or :class:`.tstruct`) -- Row key. The
      chromosome (CHROM field) and position (POS field). If `reference_genome`
      is defined, the type will be :class:`.tlocus` parameterized by
      `reference_genome`. Otherwise, the type will be a :class:`.tstruct` with
      two fields: `contig` with type :py:data:`.tstr` and `position` with type
      :py:data:`.tint32`.
    - `alleles` (:class:`.tarray` of :py:data:`.tstr`) -- Row key. An array
      containing the alleles of the variant. The reference allele (REF field) is
      the first element in the array and the alternate alleles (ALT field) are
      the subsequent elements.
    - `filters` (:class:`.tset` of :py:data:`.tstr`) -- Set containing all filters applied to a
      variant.
    - `rsid` (:py:data:`.tstr`) -- rsID of the variant.
    - `qual` (:py:data:`.tfloat64`) -- Floating-point number in the QUAL field.
    - `info` (:class:`.tstruct`) -- All INFO fields defined in the VCF header
      can be found in the struct `info`. Data types match the type specified
      in the VCF header, and if the declared ``Number`` is not 1, the result
      will be stored as an array.

    **Entry Fields**

    :func:`.import_vcf` generates an entry field for each FORMAT field declared
    in the VCF header. The types of these fields are generated according to the
    same rules as INFO fields, with one difference -- "GT" and other fields
    specified in `call_fields` will be read as :py:data:`.tcall`.

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
        List of FORMAT fields to load as :py:data:`.tcall`. "GT" is
        loaded as a call automatically.
    reference_genome: :obj:`str` or :class:`.ReferenceGenome`, optional
        Reference genome to use.
    contig_recoding: :obj:`dict` of (:obj:`str`, :obj:`str`)
        Mapping from contig name in VCF to contig name in loaded dataset.
        All contigs must be present in the `reference_genome`, so this is
        useful for mapping differently-formatted data onto known references.
    array_elements_required : :obj:`bool`
        If ``True``, all elements in an array field must be present. Set this
        parameter to ``False`` for Hail to allow array fields with missing
        values such as ``1,.,5``. In this case, the second element will be
        missing. However, in the case of a single missing element ``.``, the
        entire field will be missing and **not** an array with one missing
        element.
    skip_invalid_loci : :obj:`bool`
        If ``True``, skip loci that are not consistent with `reference_genome`.
    entry_float_type: :class:`.HailType`
        Type of floating point entries in matrix table. Must be one of:
        :py:data:`.tfloat32` or :py:data:`.tfloat64`. Default:
        :py:data:`.tfloat64`.
    filter : :obj:`str`, optional
        Line filter regex. A partial match results in the line being removed
        from the file. Applies before `find_replace`, if both are defined.
    find_replace : (:obj:`str`, :obj:`str`)
        Line substitution regex. Functions like ``re.sub``, but obeys the exact
        semantics of Java's
        `String.replaceAll <https://docs.oracle.com/javase/8/docs/api/java/lang/String.html#replaceAll-java.lang.String-java.lang.String->`__.

    Returns
    -------
    :class:`.MatrixTable`
    """

    reader = MatrixVCFReader(path, call_fields, entry_float_type, header_file, min_partitions,
                             reference_genome, contig_recoding, array_elements_required,
                             skip_invalid_loci, force_bgz, force, filter, find_replace, _partitions)
    return MatrixTable(MatrixRead(reader, drop_cols=drop_samples))


@typecheck(path=sequenceof(str),
           partitions=str,
           force=bool,
           force_bgz=bool,
           call_fields=oneof(str, sequenceof(str)),
           entry_float_type=enumeration(tfloat32, tfloat64),
           reference_genome=nullable(reference_genome_type),
           contig_recoding=nullable(dictof(str, str)),
           array_elements_required=bool,
           skip_invalid_loci=bool,
           filter=nullable(str),
           find_replace=nullable(sized_tupleof(str, str)))
def import_vcfs(path,
                partitions,
                force=False,
                force_bgz=False,
                call_fields=['PGT'],
                entry_float_type=tfloat64,
                reference_genome='default',
                contig_recoding=None,
                array_elements_required=True,
                skip_invalid_loci=False,
                filter=None,
                find_replace=None) -> List[MatrixTable]:
    """Experimental. Import multiple vcfs as :class:`.MatrixTable`s

    The arguments to this function are almost identical to :func:`.import_vcf`,
    the only difference is the `partitions` argument, which is used to divide
    and filter the vcfs. It must be a JSON string that will deserialize to an
    Array of Intervals of Locus structs. A partition will be created for every
    element of the array. Loci that fall outside of any interval will not be
    imported. For example:

    .. code-block:: text
        [
          {
            "start": {
              "locus": {
                "contig": "chr22",
                "position": 1
              }
            },
            "end": {
              "locus": {
                "contig": "chr22",
                "position": 5332423
              }
            },
            "includeStart": true,
            "includeEnd": true
          }
        ]

    The `includeStart` and `includeEnd` keys must be `true`. The `contig` fields must
    be the same.

    One difference between :func:`.import_vcfs` and :func:`.import_vcf` is that
    :func:`.import_vcfs` only keys the resulting matrix tables by `locus`
    rather than `locus, alleles`.
    """

    rg = reference_genome.name if reference_genome else None

    global _cached_importvcfs
    if _cached_importvcfs is None:
        _cached_importvcfs = Env.hail().io.vcf.ImportVCFs

    vector_ref_s = _cached_importvcfs.pyApply(
        wrap_to_list(path),
        wrap_to_list(call_fields),
        entry_float_type._parsable_string(),
        rg,
        contig_recoding,
        array_elements_required,
        skip_invalid_loci,
        force_bgz,
        force,
        partitions,
        filter,
        find_replace[0] if find_replace is not None else None,
        find_replace[1] if find_replace is not None else None)
    tmp = json.loads(vector_ref_s)
    jir_vref = JIRVectorReference(tmp['vector_ir_id'],
                                  tmp['length'],
                                  hl.tmatrix._from_json(tmp['type']))

    return [MatrixTable(JavaMatrixVectorRef(jir_vref, idx)) for idx in range(len(jir_vref))]


@typecheck(path=oneof(str, sequenceof(str)),
           index_file_map=nullable(dictof(str, str)),
           reference_genome=nullable(reference_genome_type),
           contig_recoding=nullable(dictof(str, str)),
           skip_invalid_loci=bool)
def index_bgen(path,
               index_file_map=None,
               reference_genome='default',
               contig_recoding=None,
               skip_invalid_loci=False):
    """Index BGEN files as required by :func:`.import_bgen`.

    The index file is generated in the same directory as `path` with the
    filename of `path` appended by `.idx2` unless `directory` is specified.

    Example
    -------
    Index a BGEN file, renaming contig name "01" to "1":

    >>> hl.index_bgen("data/example.8bits.bgen",
    ...               contig_recoding={"01": "1"},
    ...               reference_genome='GRCh37')

    Warning
    -------
    While this method parallelizes over a list of BGEN files, each file is
    indexed serially by one core. Indexing several BGEN files on a large cluster
    is a waste of resources, so indexing should generally be done once,
    separately from large analyses.

    See Also
    --------
    :func:`.import_bgen`

    Parameters
    ----------
    path : :obj:`str` or :obj:`list` of :obj:`str`
        .bgen files to index.
    index_file_map : :obj:`dict` of :obj:`str` to :obj:`str`, optional
        Dict of BGEN file to index file location. Index file location must have
        a `.idx2` file extension. Cannot use Hadoop glob patterns in file names.
    reference_genome : :obj:`str` or :class:`.ReferenceGenome`, optional
        Reference genome to use.
    contig_recoding : :obj:`dict` of :obj:`str` to :obj:`str`, optional
        Dict of old contig name to new contig name. The new contig name must be
        in the reference genome given by `reference_genome`.
    skip_invalid_loci : :obj:`bool`
        If ``True``, skip loci that are not consistent with `reference_genome`.

    """
    rg = reference_genome.name if reference_genome else None
    if index_file_map is None:
        index_file_map = {}
    if contig_recoding is None:
        contig_recoding = {}
    Env.hc()._jhc.indexBgen(wrap_to_list(path), index_file_map, joption(rg), contig_recoding, skip_invalid_loci)


@typecheck(path=str)
def read_table(path) -> Table:
    """Read in a :class:`.Table` written with :meth:`.Table.write`.

    Parameters
    ----------
    path : :obj:`str`
        File to read.

    Returns
    -------
    :class:`.Table`
    """
    tr = TableNativeReader(path)
    return Table(TableRead(tr, False))


@typecheck(t=Table,
           host=str,
           port=int,
           index=str,
           index_type=str,
           block_size=int,
           config=nullable(dictof(str, str)),
           verbose=bool)
def export_elasticsearch(t, host, port, index, index_type, block_size, config=None, verbose=True):
    """Export a :class:`.Table` to Elasticsearch.

    .. warning::
        :func:`.export_elasticsearch` is EXPERIMENTAL.
    """

    jdf = t.expand_types().to_spark(flatten=False)._jdf
    Env.hail().io.ElasticsearchConnector.export(jdf, host, port, index, index_type, block_size, config, verbose)
