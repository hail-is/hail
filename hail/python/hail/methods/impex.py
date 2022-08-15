import json
import os
import re
from collections import defaultdict
from typing import List

import avro.schema
from avro.datafile import DataFileReader
from avro.io import DatumReader

import hail as hl
from hail import ir
from hail.expr import StructExpression, LocusExpression, \
    expr_array, expr_float64, expr_str, expr_numeric, expr_call, expr_bool, \
    expr_any, \
    to_expr, analyze
from hail.expr.types import hail_type, tarray, tfloat64, tstr, tint32, tstruct, \
    tcall, tbool, tint64, tfloat32
from hail.genetics.reference_genome import reference_genome_type
from hail.ir.utils import parse_type
from hail.matrixtable import MatrixTable
from hail.methods.misc import require_biallelic, require_row_key_variant, require_col_key_str
from hail.table import Table
from hail.typecheck import typecheck, nullable, oneof, dictof, anytype, \
    sequenceof, enumeration, sized_tupleof, numeric, table_key_type, char
from hail.utils import new_temp_file
from hail.utils.deduplicate import deduplicate
from hail.utils.java import Env, FatalError, jindexed_seq_args, warning
from hail.utils.java import info
from hail.utils.misc import wrap_to_list, plural
from .import_lines_helpers import split_lines, should_remove_line


def locus_interval_expr(contig, start, end, includes_start, includes_end,
                        reference_genome, skip_invalid_intervals):
    includes_start = hl.bool(includes_start)
    includes_end = hl.bool(includes_end)

    if reference_genome:
        return hl.locus_interval(contig, start, end, includes_start,
                                 includes_end, reference_genome,
                                 skip_invalid_intervals)
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
    output : :class:`str`
        Filename root for output GEN and SAMPLE files.
    precision : :obj:`int`
        Number of digits to write after the decimal point.
    gp : :class:`.ArrayExpression` of type :py:data:`.tfloat64`, optional
        Expression for the genotype probabilities to output. If ``None``, the
        entry field `GP` is used if defined and is of type :class:`.tarray`
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

    hl.current_backend().validate_file_scheme(output)

    if gp is None:
        if 'GP' in dataset.entry and dataset.GP.dtype == tarray(tfloat64):
            entry_exprs = {'GP': dataset.GP}
        else:
            raise ValueError('exporting to GEN requires a GP (genotype probability) array<float64> field in the entry'
                             '\n  of the matrix table. If you only have hard calls (GT), BGEN is probably not the'
                             '\n  right format.')
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

    locus = dataset.locus
    a = dataset.alleles

    gen_exprs = {'varid': expr_or_else(varid, hl.delimit([locus.contig, hl.str(locus.position), a[0], a[1]], ':')),
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

    writer = ir.MatrixGENWriter(output, precision)
    Env.backend().execute(ir.MatrixWrite(dataset._mir, writer))


@typecheck(mt=MatrixTable,
           output=str,
           gp=nullable(expr_array(expr_float64)),
           varid=nullable(expr_str),
           rsid=nullable(expr_str),
           parallel=nullable(ir.ExportType.checker),
           compression_codec=enumeration('zlib', 'zstd'))
def export_bgen(mt, output, gp=None, varid=None, rsid=None, parallel=None, compression_codec='zlib'):
    """Export MatrixTable as :class:`.MatrixTable` as BGEN 1.2 file with 8
    bits of per probability.  Also writes SAMPLE file.

    If `parallel` is ``None``, the BGEN file is written to ``output + '.bgen'``. Otherwise, ``output
    + '.bgen'`` will be a directory containing many BGEN files. In either case, the SAMPLE file is
    written to ``output + '.sample'``. For example,

    >>> hl.export_bgen(mt, '/path/to/dataset')  # doctest: +SKIP

    Will write two files: `/path/to/dataset.bgen` and `/path/to/dataset.sample`. In contrast,

    >>> hl.export_bgen(mt, '/path/to/dataset', parallel='header_per_shard')  # doctest: +SKIP

    Will create `/path/to/dataset.sample` and will create ``mt.n_partitions()`` files into the
    directory `/path/to/dataset.bgen/`.


    Notes
    -----
    The :func:`export_bgen` function requires genotype probabilities, either as an entry
    field of `mt` (of type ``array<float64>``), or an entry expression passed in the `gp`
    argument.

    Parameters
    ----------
    mt : :class:`.MatrixTable`
        Input matrix table.
    output : :class:`str`
        Root for output BGEN and SAMPLE files.
    gp : :class:`.ArrayExpression` of type :py:data:`.tfloat64`, optional
        Expression for genotype probabilities.  If ``None``, entry
        field `GP` is used if it exists and is of type
        :class:`.tarray` with element type :py:data:`.tfloat64`.
    varid : :class:`.StringExpression`, optional
        Expression for the variant ID. If ``None``, the row field
        `varid` is used if defined and is of type :py:data:`.tstr`.
        The default and missing value is
        ``hl.delimit([mt.locus.contig, hl.str(mt.locus.position), mt.alleles[0], mt.alleles[1]], ':')``
    rsid : :class:`.StringExpression`, optional
        Expression for the rsID. If ``None``, the row field `rsid` is
        used if defined and is of type :py:data:`.tstr`.  The default
        and missing value is ``"."``.
    parallel : :class:`str`, optional
        If ``None``, write a single BGEN file.  If ``'header_per_shard'``, write a collection of
        BGEN files (one per partition), each with its own header.  If ``'separate_header'``, write a
        file for each partition, without header, and a header file for the combined dataset. Note
        that the files produced by ``'separate_header'`` are each individually invalid BGEN files,
        they can only be read if they are concatenated together with the header file.
    compresssion_codec : str, optional
        Compression codec. One of 'zlib', 'zstd'.

    """
    require_row_key_variant(mt, 'export_bgen')
    require_col_key_str(mt, 'export_bgen')

    hl.current_backend().validate_file_scheme(output)

    if gp is None:
        if 'GP' in mt.entry and mt.GP.dtype == tarray(tfloat64):
            entry_exprs = {'GP': mt.GP}
        else:
            raise ValueError('exporting to BGEN requires a GP (genotype probability) array<float64> field in the entry'
                             '\n  of the matrix table. If you only have hard calls (GT), BGEN is probably not the'
                             '\n  right format.')
    else:
        entry_exprs = {'GP': gp}

    if varid is None:
        if 'varid' in mt.row and mt.varid.dtype == tstr:
            varid = mt.varid

    if rsid is None:
        if 'rsid' in mt.row and mt.rsid.dtype == tstr:
            rsid = mt.rsid

    parallel = ir.ExportType.default(parallel)

    locus = mt.locus
    a = mt.alleles
    gen_exprs = {'varid': expr_or_else(varid, hl.delimit([locus.contig, hl.str(locus.position), a[0], a[1]], ':')),
                 'rsid': expr_or_else(rsid, ".")}

    for exprs, axis in [(gen_exprs, mt._row_indices),
                        (entry_exprs, mt._entry_indices)]:
        for name, expr in exprs.items():
            analyze('export_bgen/{}'.format(name), expr, axis)

    mt = mt._select_all(col_exprs={},
                        row_exprs=gen_exprs,
                        entry_exprs=entry_exprs)

    Env.backend().execute(ir.MatrixWrite(mt._mir, ir.MatrixBGENWriter(
        output,
        parallel,
        compression_codec)))


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
    output : :class:`str`
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

    require_biallelic(dataset, 'export_plink', tolerate_generic_locus=True)

    hl.current_backend().validate_file_scheme(output)

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
                                           lambda x: hl.if_else(x, '2', '1')),
                 'pheno': expr_or_else(pheno, 'NA',
                                       lambda x: hl.if_else(x, '2', '1') if x.dtype == tbool else hl.str(x))}

    locus = dataset.locus
    a = dataset.alleles

    bim_exprs = {'varid': expr_or_else(varid, hl.delimit([locus.contig, hl.str(locus.position), a[0], a[1]], ':')),
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

    writer = ir.MatrixPLINKWriter(output)
    Env.backend().execute(ir.MatrixWrite(dataset._mir, writer))


@typecheck(dataset=oneof(MatrixTable, Table),
           output=str,
           append_to_header=nullable(str),
           parallel=nullable(ir.ExportType.checker),
           metadata=nullable(dictof(str, dictof(str, dictof(str, str)))),
           tabix=bool)
def export_vcf(dataset, output, append_to_header=None, parallel=None, metadata=None, *, tabix=False):
    """Export a :class:`.MatrixTable` or :class:`.Table` as a VCF file.

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
    field for each field of the struct. If `dataset` is a :class:`.Table`,
    then there will be no FORMAT field and the output will be a sites-only VCF.

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
    output : :class:`str`
        Path of .vcf or .vcf.bgz file to write.
    append_to_header : :class:`str`, optional
        Path of file to append to VCF header.
    parallel : :class:`str`, optional
        If ``'header_per_shard'``, return a set of VCF files (one per
        partition) rather than serially concatenating these files. If
        ``'separate_header'``, return a separate VCF header file and a set of
        VCF files (one per partition) without the header. If ``None``,
        concatenate the header and all partitions into one VCF file.
    metadata : :obj:`dict` [:obj:`str`, :obj:`dict` [:obj:`str`, :obj:`dict` [:obj:`str`, :obj:`str`]]], optional
        Dictionary with information to fill in the VCF header. See
        :func:`get_vcf_metadata` for how this
        dictionary should be structured.
    tabix : :obj:`bool`, optional
        If true, writes a tabix index for the output VCF.
        **Note**: This feature is experimental, and the interface and defaults
        may change in future versions.
    """
    hl.current_backend().validate_file_scheme(output)

    _, ext = os.path.splitext(output)
    if ext == '.gz':
        warning('VCF export with standard gzip compression requested. This is almost *never* desired and will '
                'cause issues with other tools that consume VCF files. The compression format used for VCF '
                'files is traditionally *block* gzip compression. To use block gzip compression with hail VCF '
                'export, use a path ending in `.bgz`.')

    if isinstance(dataset, Table):
        mt = MatrixTable.from_rows_table(dataset)
        dataset = mt.key_cols_by(sample="")

    require_col_key_str(dataset, 'export_vcf')
    require_row_key_variant(dataset, 'export_vcf')

    if 'filters' in dataset.row and dataset.filters.dtype != hl.tset(hl.tstr):
        raise ValueError(f"'export_vcf': expect the 'filters' field to be set<str>, found {dataset.filters.dtype}"
                         f"\n  Either transform this field to set<str> to export as VCF FILTERS field, or drop it from the dataset.")

    info_fields = list(dataset.info) if "info" in dataset.row else []
    invalid_info_fields = [f for f in info_fields if not re.fullmatch(r"^([A-Za-z_][0-9A-Za-z_.]*|1000G)", f)]
    if invalid_info_fields:
        invalid_info_str = ''.join(f'\n    {f!r}' for f in invalid_info_fields)
        warning(
            'export_vcf: the following info field names are invalid in VCF 4.3 and may not work with some tools: ' + invalid_info_str)

    row_fields_used = {'rsid', 'info', 'filters', 'qual'}

    fields_dropped = []
    for f in dataset.globals:
        fields_dropped.append((f, 'global'))
    for f in dataset.col_value:
        fields_dropped.append((f, 'column'))
    for f in dataset.row_value:
        if f not in row_fields_used:
            fields_dropped.append((f, 'row'))

    if fields_dropped:
        ignored_str = ''.join(f'\n    {f!r} ({axis})' for f, axis in fields_dropped)
        warning('export_vcf: ignored the following fields:' + ignored_str)
        dataset = dataset.drop(*(f for f, _ in fields_dropped))

    parallel = ir.ExportType.default(parallel)

    writer = ir.MatrixVCFWriter(output,
                                append_to_header,
                                parallel,
                                metadata,
                                tabix)
    Env.backend().execute(ir.MatrixWrite(dataset._mir, writer))


@typecheck(path=str,
           reference_genome=nullable(reference_genome_type),
           skip_invalid_intervals=bool,
           contig_recoding=nullable(dictof(str, str)),
           kwargs=anytype)
def import_locus_intervals(path,
                           reference_genome='default',
                           skip_invalid_intervals=False,
                           contig_recoding=None,
                           **kwargs) -> Table:
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
      type :obj:`.tstr` and `position` with type :py:data:`.tint32`.

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
    path : :class:`str`
        Path to file.
    reference_genome : :class:`str` or :class:`.ReferenceGenome`, optional
        Reference genome to use.
    skip_invalid_intervals : :obj:`bool`
        If ``True`` and `reference_genome` is not ``None``, skip lines with
        intervals that are not consistent with the reference genome.
    contig_recoding: :obj:`dict` of (:class:`str`, :obj:`str`)
        Mapping from contig name in file to contig name in loaded dataset.
        All contigs must be present in the `reference_genome`, so this is
        useful for mapping differently-formatted data onto known references.
    **kwargs
        Additional optional arguments to :func:`import_table` are valid
        arguments here except: `no_header`, `comment`, `impute`, and
        `types`, as these are used by :func:`import_locus_intervals`.

    Returns
    -------
    :class:`.Table`
        Interval-keyed table.
    """

    if contig_recoding is not None:
        contig_recoding = hl.literal(contig_recoding)

    def recode_contig(x):
        if contig_recoding is None:
            return x
        return contig_recoding.get(x, x)

    t = import_table(path, comment="@", impute=False, no_header=True,
                     types={'f0': tstr, 'f1': tint32, 'f2': tint32,
                            'f3': tstr, 'f4': tstr},
                     **kwargs)

    if t.row.dtype == tstruct(f0=tstr):
        if reference_genome:
            t = t.select(interval=hl.parse_locus_interval(t['f0'],
                                                          reference_genome))
        else:
            interval_regex = r"([^:]*):(\d+)\-(\d+)"

            def checked_match_interval_expr(match):
                return hl.or_missing(hl.len(match) == 3,
                                     locus_interval_expr(recode_contig(match[0]),
                                                         hl.int32(match[1]),
                                                         hl.int32(match[2]),
                                                         True,
                                                         True,
                                                         reference_genome,
                                                         skip_invalid_intervals))

            expr = (
                hl.bind(t['f0'].first_match_in(interval_regex),
                        lambda match: hl.if_else(hl.bool(skip_invalid_intervals),
                                                 checked_match_interval_expr(match),
                                                 locus_interval_expr(recode_contig(match[0]),
                                                                     hl.int32(match[1]),
                                                                     hl.int32(match[2]),
                                                                     True,
                                                                     True,
                                                                     reference_genome,
                                                                     skip_invalid_intervals))))

            t = t.select(interval=expr)

    elif t.row.dtype == tstruct(f0=tstr, f1=tint32, f2=tint32):
        t = t.select(interval=locus_interval_expr(recode_contig(t['f0']),
                                                  t['f1'],
                                                  t['f2'],
                                                  True,
                                                  True,
                                                  reference_genome,
                                                  skip_invalid_intervals))

    elif t.row.dtype == tstruct(f0=tstr, f1=tint32, f2=tint32, f3=tstr, f4=tstr):
        t = t.select(interval=locus_interval_expr(recode_contig(t['f0']),
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
           skip_invalid_intervals=bool,
           contig_recoding=nullable(dictof(str, str)),
           kwargs=anytype)
def import_bed(path,
               reference_genome='default',
               skip_invalid_intervals=False,
               contig_recoding=None,
               **kwargs) -> Table:
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
    Intervals in UCSC BED files are 0-indexed and half open.
    The line "5  100  105" correpsonds to the interval ``[5:101-5:106)`` in Hail's
    1-indexed notation. Details
    `here <http://genome.ucsc.edu/blog/the-ucsc-genome-browser-coordinate-counting-systems/>`__.

    Parameters
    ----------
    path : :class:`str`
        Path to .bed file.
    reference_genome : :class:`str` or :class:`.ReferenceGenome`, optional
        Reference genome to use.
    skip_invalid_intervals : :obj:`bool`
        If ``True`` and `reference_genome` is not ``None``, skip lines with
        intervals that are not consistent with the reference genome.
    contig_recoding: :obj:`dict` of (:class:`str`, :obj:`str`)
        Mapping from contig name in BED to contig name in loaded dataset.
        All contigs must be present in the `reference_genome`, so this is
        useful for mapping differently-formatted data onto known references.
    **kwargs
        Additional optional arguments to :func:`import_table` are valid arguments here except:
        `no_header`, `delimiter`, `impute`, `skip_blank_lines`, `types`, and `comment` as these
        are used by import_bed.

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
                              r"""^\w+=("[\w\d ]+"|\d+).*"""],
                     **kwargs)

    if contig_recoding is not None:
        contig_recoding = hl.literal(contig_recoding)

    def recode_contig(x):
        if contig_recoding is None:
            return x
        return contig_recoding.get(x, x)

    if t.row.dtype == tstruct(f0=tstr, f1=tint32, f2=tint32):
        t = t.select(interval=locus_interval_expr(recode_contig(t['f0']),
                                                  t['f1'] + 1,
                                                  t['f2'] + 1,
                                                  True,
                                                  False,
                                                  reference_genome,
                                                  skip_invalid_intervals))

    elif len(t.row) >= 4 and tstruct(**dict([(n, typ) for n, typ in t.row.dtype._field_types.items()][:4])) == tstruct(
            f0=tstr, f1=tint32, f2=tint32, f3=tstr):
        t = t.select(interval=locus_interval_expr(recode_contig(t['f0']),
                                                  t['f1'] + 1,
                                                  t['f2'] + 1,
                                                  True,
                                                  False,
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
    path : :class:`str`
        Path to FAM file.
    quant_pheno : :obj:`bool`
        If ``True``, phenotype is interpreted as quantitative.
    delimiter : :class:`str`
        Field delimiter regex.
    missing : :class:`str`
        The string used to denote missing values. For case-control, 0, -9, and
        non-numeric are also treated as missing.

    Returns
    -------
    :class:`.Table`
    """
    type_and_data = Env.backend().import_fam(path, quant_pheno, delimiter, missing)
    typ = hl.dtype(type_and_data['type'])
    return hl.Table.parallelize(
        hl.tarray(typ)._convert_from_json_na(type_and_data['data']), typ, key=['id'])


@typecheck(regex=str,
           path=oneof(str, sequenceof(str)),
           max_count=int,
           show=bool,
           force=bool,
           force_bgz=bool)
def grep(regex, path, max_count=100, *, show: bool = True, force: bool = False, force_bgz: bool = False):
    r"""Searches given paths for all lines containing regex matches.

    Examples
    --------

    Print all lines containing the string ``hello`` in *file.txt*:

    >>> hl.grep('hello','data/file.txt')

    Print all lines containing digits in *file1.txt* and *file2.txt*:

    >>> hl.grep('\\d', ['data/file1.txt','data/file2.txt'])

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
    regex : :class:`str`
        The regular expression to match.
    path : :class:`str` or :obj:`list` of :obj:`str`
        The files to search.
    max_count : :obj:`int`
        The maximum number of matches to return
    show : :obj:`bool`
        When `True`, show the values on stdout. When `False`, return a
        dictionary mapping file names to lines.
    force_bgz : :obj:`bool`
        If ``True``, read files as blocked gzip files, assuming
        that they were actually compressed using the BGZ codec. This option is
        useful when the file extension is not ``'.bgz'``, but the file is
        blocked gzip, so that the file can be read in parallel and not on a
        single node.
    force : :obj:`bool`
        If ``True``, read gzipped files serially on one core. This should
        be used only when absolutely necessary, as processing time will be
        increased due to lack of parallelism.

    Returns
    ---
    :obj:`dict` of :class:`str` to :obj:`list` of :obj:`str`
    """
    from hail.backend.spark_backend import SparkBackend

    if isinstance(hl.current_backend(), SparkBackend):
        jfs = Env.spark_backend('grep').fs._jfs
        if show:
            Env.backend()._jhc.grepPrint(jfs, regex, jindexed_seq_args(path), max_count)
            return
        else:
            jarr = Env.backend()._jhc.grepReturn(jfs, regex, jindexed_seq_args(path), max_count)
            return {x._1(): list(x._2()) for x in jarr}

    ht = hl.import_lines(path, force=force, force_bgz=force_bgz)
    ht = ht.filter(ht.text.matches(regex))
    ht = ht.head(max_count)
    lines = ht.collect()
    if show:
        print('\n'.join(line.file + ': ' + line.text for line in lines))
        return

    results = defaultdict(list)
    for line in lines:
        results[line.file].append(line.text)
    return results


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
    path : :class:`str` or :obj:`list` of :obj:`str`
        BGEN file(s) to read.
    entry_fields : :obj:`list` of :class:`str`
        List of entry fields to create.
        Options: ``'GT'``, ``'GP'``, ``'dosage'``.
    sample_file : :class:`str`, optional
        Sample file to read the sample ids from. If specified, the number of
        samples in the file must match the number in the BGEN file(s).
    n_partitions : :obj:`int`, optional
        Number of partitions.
    block_size : :obj:`int`, optional
        Block size, in MB.
    index_file_map : :obj:`dict` of :class:`str` to :obj:`str`, optional
        Dict of BGEN file to index file location. Cannot use Hadoop glob
        patterns in file names.
    variants : :class:`.StructExpression` or :class:`.LocusExpression` or :obj:`list` of :class:`.Struct` or :obj:`list` of :class:`.Locus` or :class:`.Table`
        Variants to filter to. The underlying type of the input (row key in the case of a :class:`.Table`)
        must be either a :class:`.tlocus`, a struct with one field `locus`, or a struct with two fields:
        `locus` and `alleles`. The type of `locus` can either be a :class:`.tlocus` or a :class:`.tstruct`
        with two fields: `contig` of type :obj:`.tstr` and `position` of type :obj:`.tint`. If the
        type of `locus` is :class:`.tlocus`, the reference genome must match that used to index the BGEN
        file(s). The type of `alleles` is a :class:`.tarray` of :obj:`.tstr`.
    _row_fields : :obj:`list` of :class:`str`
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
            ir.MatrixRead(ir.MatrixBGENReader(path, sample_file, index_file_map, n_partitions, block_size, None)))
        lt = mt_type.row_type['locus']

        expected_vtype = tstruct(locus=lt, alleles=tarray(tstr))

        if isinstance(variants, StructExpression) or isinstance(variants, LocusExpression):
            if isinstance(variants, LocusExpression):
                variants = hl.struct(locus=variants)

            if len(variants.dtype) == 0 or not variants.dtype._is_prefix_of(expected_vtype):
                raise TypeError(
                    "'import_bgen' requires the expression type for 'variants' is a non-empty prefix of the BGEN key type: \n"
                    + f"\tFound: {repr(variants.dtype)}\n"
                    + f"\tExpected: {repr(expected_vtype)}\n")

            uid = Env.get_uid()
            fnames = list(variants.dtype)
            name, variants = variants._to_table(
                uid)  # This will add back the other key fields of the source, which we don't want
            variants = variants.key_by(**{fname: variants[name][fname] for fname in fnames})
            variants = variants.select()
        elif isinstance(variants, Table):
            if len(variants.key) == 0 or not variants.key.dtype._is_prefix_of(expected_vtype):
                raise TypeError(
                    "'import_bgen' requires the row key type for 'variants' is a non-empty prefix of the BGEN key type: \n"
                    + f"\tFound: {repr(variants.key.dtype)}\n"
                    + f"\tExpected: {repr(expected_vtype)}\n")
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
            except Exception:
                raise TypeError(
                    f"'import_bgen' requires all elements in 'variants' are a non-empty prefix of the BGEN key type: {repr(expected_vtype)}")

        vir = variants._tir
        if isinstance(vir, ir.TableRead) \
                and isinstance(vir.reader, ir.TableNativeReader) \
                and vir.reader.intervals is None \
                and variants.count() == variants.distinct().count():
            variants_path = vir.reader.path
        else:
            variants_path = new_temp_file(prefix='bgen_included_vars', extension='ht')
            variants.distinct().write(variants_path)
    else:
        variants_path = None

    reader = ir.MatrixBGENReader(path, sample_file, index_file_map, n_partitions, block_size, variants_path)

    mt = (MatrixTable(ir.MatrixRead(reader))
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
      given by `chromosome`) and position (4th column if `chromosome` is not
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
    path : :class:`str` or :obj:`list` of :obj:`str`
        GEN files to import.
    sample_file : :class:`str`
        Sample file to import.
    tolerance : :obj:`float`
        If the sum of the genotype probabilities for a genotype differ from 1.0
        by more than the tolerance, set the genotype to missing.
    min_partitions : :obj:`int`, optional
        Number of partitions.
    chromosome : :class:`str`, optional
        Chromosome if not included in the GEN file
    reference_genome : :class:`str` or :class:`.ReferenceGenome`, optional
        Reference genome to use.
    contig_recoding : :obj:`dict` of :class:`str` to :obj:`str`, optional
        Dict of old contig name to new contig name. The new contig name must be
        in the reference genome given by `reference_genome`.
    skip_invalid_loci : :obj:`bool`
        If ``True``, skip loci that are not consistent with `reference_genome`.

    Returns
    -------
    :class:`.MatrixTable`
    """
    gen_table = import_lines(path, min_partitions)
    sample_table = import_lines(sample_file)
    rg = reference_genome.name if reference_genome else None
    contig_recoding = contig_recoding
    if contig_recoding is None:
        contig_recoding = hl.empty_dict(hl.tstr, hl.tstr)
    else:
        contig_recoding = hl.dict(contig_recoding)

    gen_table = gen_table.transmute(data=gen_table.text.split(' '))

    if chromosome is None:
        last_rowf_idx = 5
        contig_holder = gen_table.data[0]
    else:
        last_rowf_idx = 4
        contig_holder = chromosome

    contig_holder = contig_recoding.get(contig_holder, contig_holder)

    position = hl.int(gen_table.data[last_rowf_idx - 2])
    alleles = hl.array([hl.str(gen_table.data[last_rowf_idx - 1]), hl.str(gen_table.data[last_rowf_idx])])
    rsid = gen_table.data[last_rowf_idx - 3]
    varid = gen_table.data[last_rowf_idx - 4]
    if rg is None:
        locus = hl.struct(contig=contig_holder, position=position)
    else:
        if skip_invalid_loci:
            locus = hl.if_else(hl.is_valid_locus(contig_holder, position, rg),
                               hl.locus(contig_holder, position, rg),
                               hl.missing(hl.tlocus(rg)))
        else:
            locus = hl.locus(contig_holder, position, rg)

    gen_table = gen_table.annotate(locus=locus, alleles=alleles, rsid=rsid, varid=varid)
    gen_table = gen_table.annotate(entries=gen_table.data[last_rowf_idx + 1:].map(lambda x: hl.float64(x))
                                   .grouped(3).map(lambda x: hl.struct(GP=x)))
    if skip_invalid_loci:
        gen_table = gen_table.filter(hl.is_defined(gen_table.locus))

    sample_table_count = sample_table.count() - 2  # Skipping first 2 unneeded rows in sample file
    gen_table = gen_table.annotate_globals(cols=hl.range(sample_table_count).map(lambda x: hl.struct(col_idx=x)))
    mt = gen_table._unlocalize_entries('entries', 'cols', ['col_idx'])

    sample_table = sample_table.tail(sample_table_count).add_index()
    sample_table = sample_table.annotate(s=sample_table.text.split(' ')[0])
    sample_table = sample_table.key_by(sample_table.idx)
    mt = mt.annotate_cols(s=sample_table[hl.int64(mt.col_idx)].s)

    mt = mt.annotate_entries(GP=hl.rbind(hl.sum(mt.GP), lambda gp_sum: hl.if_else(hl.abs(1.0 - gp_sum) > tolerance,
                                                                                  hl.missing(hl.tarray(hl.tfloat64)),
                                                                                  hl.abs((1 / gp_sum) * mt.GP))))
    mt = mt.annotate_entries(GT=hl.rbind(hl.argmax(mt.GP),
                                         lambda max_idx: hl.if_else(
                                             hl.len(mt.GP.filter(lambda y: y == mt.GP[max_idx])) == 1,
                                             hl.switch(max_idx)
                                             .when(0, hl.call(0, 0))
                                             .when(1, hl.call(0, 1))
                                             .when(2, hl.call(1, 1))
                                             .or_error("error creating gt field."),
                                             hl.missing(hl.tcall))))
    mt = mt.filter_entries(hl.is_defined(mt.GP))

    mt = mt.key_cols_by('s').drop('col_idx', 'file', 'data')
    mt = mt.key_rows_by('locus', 'alleles').select_entries('GT', 'GP')
    return mt


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
           force=bool,
           source_file_field=nullable(str))
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
                 force=False,
                 source_file_field=None) -> Table:
    """Import delimited text file (text table) as :class:`.Table`.

    The resulting :class:`.Table` will have no key fields. Use
    :meth:`.Table.key_by` to specify keys. See also:
    :func:`.import_matrix_table`.

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

        $ cat data/samples2.csv
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

    >>> table = hl.import_table('data/samples2.csv', delimiter=',', missing='.')

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

    Let's import a table from a file where one of the fields is a JSON object.

    .. code-block: text

        $cat data/table_with_json.tsv
        id     json_field
        1      {"foo": "bar", "x": 7}
        4      {"foo": "baz", "x": 100}

    To import, we need to specify the types argument.

    >>> my_types = {"id": hl.tint32, "json_field":hl.tstruct(foo=hl.tstr, x=hl.tint32)}
    >>> ht_with_json = hl.import_table('data/table_with_json.tsv', types=my_types)

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
    table. It is an :obj:`dict` keyed by :class:`str`, with :class:`.HailType` values.
    See the examples above for a standard usage. Additionally, this option can
    be used to override type imputation. For example, if the field
    ``Chromosome`` only contains the values ``1`` through ``22``, it will be
    imputed to have type :py:data:`.tint32`, whereas most Hail methods expect
    that a chromosome field will be of type :py:data:`.tstr`. Setting
    ``impute=True`` and ``types={'Chromosome': hl.tstr}`` solves this problem.

    Parameters
    ----------

    paths : :class:`str` or :obj:`list` of :obj:`str`
        Files to import.
    key : :class:`str` or :obj:`list` of :obj:`str`
        Key fields(s).
    min_partitions : :obj:`int` or :obj:`None`
        Minimum number of partitions.
    no_header : :obj:`bool`
        If ``True```, assume the file has no header and name the N fields `f0`,
        `f1`, ... `fN` (0-indexed).
    impute : :obj:`bool`
        If ``True``, Impute field types from the file.
    comment : :class:`str` or :obj:`list` of :obj:`str`
        Skip lines beginning with the given string if the string is a single
        character. Otherwise, skip lines that match the regex specified. Multiple
        comment characters or patterns should be passed as a list.
    delimiter : :class:`str`
        Field delimiter regex.
    missing : :class:`str` or :obj:`list` [:obj:`str`]
        Identifier(s) to be treated as missing.
    types : :obj:`dict` mapping :class:`str` to :class:`.HailType`
        Dictionary defining field types.
    quote : :class:`str` or :obj:`None`
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
    filter : :class:`str`, optional
        Line filter regex. A partial match results in the line being removed
        from the file. Applies before `find_replace`, if both are defined.
    find_replace : (:class:`str`, :obj:`str`)
        Line substitution regex. Functions like ``re.sub``, but obeys the exact
        semantics of Java's
        `String.replaceAll <https://docs.oracle.com/javase/8/docs/api/java/lang/String.html#replaceAll-java.lang.String-java.lang.String->`__.
    force : :obj:`bool`
        If ``True``, load gzipped files serially on one core. This should
        be used only when absolutely necessary, as processing time will be
        increased due to lack of parallelism.
    source_file_field : :class:`str`, optional
        If defined, the source file name for each line will be a field of the table
        with this name. Can be useful when importing multiple tables using glob patterns.
    Returns
    -------
    :class:`.Table`
    """
    if len(delimiter) < 1:
        raise ValueError('import_table: empty delimiter is not supported')

    paths = wrap_to_list(paths)
    comment = wrap_to_list(comment)
    missing = wrap_to_list(missing)

    ht = hl.import_lines(paths, min_partitions, force_bgz, force)

    should_remove_line_expr = should_remove_line(
        ht.text, filter=filter, comment=comment, skip_blank_lines=skip_blank_lines
    )
    if should_remove_line_expr is not None:
        ht = ht.filter(should_remove_line_expr, keep=False)

    try:
        if len(paths) <= 1:
            # With zero or one files and no filters, the first row, if it exists must be in the first
            # partition, so we take this one-pass fast-path.
            first_row_ht = ht._filter_partitions([0]).head(1)
        else:
            first_row_ht = ht.head(1)

        if find_replace is not None:
            ht = ht.annotate(text=ht['text'].replace(*find_replace))

        first_rows = first_row_ht.annotate(
            header=first_row_ht.text._split_line(
                delimiter, missing=hl.empty_array(hl.tstr), quote=quote, regex=len(delimiter) > 1)
        ).collect()
    except FatalError as err:
        if '_filter_partitions: no partition with index 0' in err.args[0]:
            first_rows = []
        else:
            raise

    if len(first_rows) == 0:
        raise ValueError(f"Invalid file: no lines remaining after filters\n Files provided: {', '.join(paths)}")
    first_row = first_rows[0]

    if no_header:
        fields = [f'f{index}' for index in range(0, len(first_row.header))]
    else:
        maybe_duplicated_fields = first_row.header
        renamings, fields = deduplicate(maybe_duplicated_fields)
        ht = ht.filter(ht.text == first_row.text, keep=False)  # FIXME: seems wrong. Could easily fix with partition index and row_within_partition_index.
        if renamings:
            hl.utils.warning(
                f'import_table: renamed the following {plural("field", len(renamings))} to avoid name conflicts:'
                + ''.join(f'\n    {repr(k)} -> {repr(v)}' for k, v in renamings)
            )

    ht = ht.annotate(
        split_text=(
            hl.case()
            .when(
                hl.len(ht.text) > 0,
                split_lines(ht, fields, delimiter=delimiter, missing=missing, quote=quote)
            )
            .or_error(hl.str("Blank line found in file ") + ht.file)
        )
    )
    ht = ht.drop('text')

    fields_to_value = {}
    strs = []
    if impute:
        fields_to_impute_idx = []
        fields_to_guess = []
        for idx, field in enumerate(fields):
            if types.get(field) is None:
                fields_to_impute_idx.append(idx)
                fields_to_guess.append(field)

        hl.utils.info('Reading table to impute column types')
        guessed = ht.aggregate(hl.agg.array_agg(lambda x: hl.agg._impute_type(x),
                                                [ht.split_text[i] for i in fields_to_impute_idx]))

        reasons = {f: 'user-supplied type' for f in types}
        imputed_types = dict()
        for field, s in zip(fields_to_guess, guessed):
            if not s['anyNonMissing']:
                imputed_types[field] = hl.tstr
                reasons[field] = 'no non-missing observations'
            else:
                if s['supportsBool']:
                    imputed_types[field] = hl.tbool
                elif s['supportsInt32']:
                    imputed_types[field] = hl.tint32
                elif s['supportsInt64']:
                    imputed_types[field] = hl.tint64
                elif s['supportsFloat64']:
                    imputed_types[field] = hl.tfloat64
                else:
                    imputed_types[field] = hl.tstr
                reasons[field] = 'imputed'

        strs.append('Finished type imputation')

        all_types = dict(**types, **imputed_types)

        for f_idx, field in enumerate(fields):
            strs.append(f'  Loading field {field!r} as type {all_types[field]} ({reasons[field]})')
            fields_to_value[field] = parse_type(ht.split_text[f_idx], all_types[field])
    else:
        strs.append('Reading table without type imputation')
        for f_idx, field in enumerate(fields):
            reason = 'user-supplied' if field in types else 'not specified'
            t = types.get(field, hl.tstr)
            fields_to_value[field] = parse_type(ht.split_text[f_idx], t)
            strs.append(f'  Loading field {field!r} as type {t} ({reason})')

    ht = ht.annotate(**fields_to_value).drop('split_text')
    if source_file_field is not None:
        source_file = {source_file_field: ht.file}
        ht = ht.annotate(**source_file)
    ht = ht.drop('file')

    if len(fields) < 30:
        hl.utils.info('\n'.join(strs))
    else:
        from collections import Counter
        strs2 = [f'Loading {ht.row} fields. Counts by type:']
        for name, count in Counter(ht[f].dtype for f in fields).most_common():
            strs2.append(f'  {name}: {count}')
        hl.utils.info('\n'.join(strs2))

    if key:
        key = wrap_to_list(key)
        ht = ht.key_by(*key)
    return ht


@typecheck(paths=oneof(str, sequenceof(str)), min_partitions=nullable(int), force_bgz=bool,
           force=bool, file_per_partition=bool)
def import_lines(paths, min_partitions=None, force_bgz=False, force=False, file_per_partition=False) -> Table:
    """Import lines of file(s) as a :class:`.Table` of strings.

    Examples
    --------

    To import a file as a table of strings:

    >>> ht = hl.import_lines('data/matrix2.tsv')
    >>> ht.describe()
    ----------------------------------------
    Global fields:
        None
    ----------------------------------------
    Row fields:
        'file': str
        'text': str
    ----------------------------------------
    Key: []
    ----------------------------------------

    Parameters
    ----------
    paths: :class:`str` or :obj:`list` of :obj:`str`
        Files to import.
    min_partitions: :obj:`int` or :obj:`None`
        Minimum number of partitions.
    force_bgz : :obj:`bool`
        If ``True``, load files as blocked gzip files, assuming
        that they were actually compressed using the BGZ codec. This option is
        useful when the file extension is not ``'.bgz'``, but the file is
        blocked gzip, so that the file can be read in parallel and not on a
        single node.
    force : :obj:`bool`
        If ``True``, load gzipped files serially on one core. This should
        be used only when absolutely necessary, as processing time will be
        increased due to lack of parallelism.
    file_per_partition : :obj:`bool`
        If ``True``, each file will be in a seperate partition. Not recommended
        for most uses. Error thrown if ``True`` and `min_partitions` is less than
        the number of files

    Returns
    -------
    :class:`.Table`
        Table constructed from imported data.
    """

    paths = wrap_to_list(paths)

    if file_per_partition and min_partitions is not None:
        if min_partitions > len(paths):
            raise FatalError(f'file_per_partition is True while min partitions is {min_partitions} ,which is greater'
                             f' than the number of files, {len(paths)}')

    st_reader = ir.StringTableReader(paths, min_partitions, force_bgz, force, file_per_partition)
    table_type = hl.ttable(
        global_type=hl.tstruct(),
        row_type=hl.tstruct(file=hl.tstr, text=hl.tstr),
        row_key=[]
    )
    string_table = Table(ir.TableRead(st_reader, _assert_type=table_type))
    return string_table


@typecheck(paths=oneof(str, sequenceof(str)),
           row_fields=dictof(str, hail_type),
           row_key=oneof(str, sequenceof(str)),
           entry_type=enumeration(tint32, tint64, tfloat32, tfloat64, tstr),
           missing=str,
           min_partitions=nullable(int),
           no_header=bool,
           force_bgz=bool,
           sep=nullable(str),
           delimiter=nullable(str),
           comment=oneof(str, sequenceof(str)),
           )
def import_matrix_table(paths,
                        row_fields={},
                        row_key=[],
                        entry_type=tint32,
                        missing="NA",
                        min_partitions=None,
                        no_header=False,
                        force_bgz=False,
                        sep=None,
                        delimiter=None,
                        comment=()) -> MatrixTable:
    """Import tab-delimited file(s) as a :class:`.MatrixTable`.

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

    The field ``Days`` contains floating-point numbers and each of the ``GENE`` fields contain integers.
    contains integers.

    To import this matrix:

    >>> matrix1 = hl.import_matrix_table('data/matrix1.tsv',
    ...                                  row_fields={'Barcode': hl.tstr, 'Tissue': hl.tstr, 'Days':hl.tfloat32},
    ...                                  row_key='Barcode')
    >>> matrix1.describe()  # doctest: +SKIP_OUTPUT_CHECK
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
          `row_id` of type :obj:`int`, whose values 0, 1, ... index the original
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

    The column IDs will never be missing, even if the `missing` string appears
    in the column IDs.

    Parameters
    ----------
    paths: :class:`str` or :obj:`list` of :obj:`str`
        Files to import.
    row_fields: :obj:`dict` of :class:`str` to :class:`.HailType`
        Columns to take as row fields in the MatrixTable. They must be located
        before all entry columns.
    row_key: :class:`str` or :obj:`list` of :obj:`str`
        Key fields(s). If empty, creates an index `row_id` to use as key.
    entry_type: :class:`.HailType`
        Type of entries in matrix table. Must be one of: :py:data:`.tint32`,
        :py:data:`.tint64`, :py:data:`.tfloat32`, :py:data:`.tfloat64`, or
        :py:data:`.tstr`. Default: :py:data:`.tint32`.
    missing: :class:`str`
        Identifier to be treated as missing. Default: NA
    min_partitions: :obj:`int` or :obj:`None`
        Minimum number of partitions.
    no_header: :obj:`bool`
        If ``True``, assume the file has no header and name the row fields `f0`,
        `f1`, ... `fK` (0-indexed) and the column keys 0, 1, ... N.
    force_bgz : :obj:`bool`
        If ``True``, load **.gz** files as blocked gzip files, assuming
        that they were actually compressed using the BGZ codec.
    sep : :class:`str`
        This parameter is a deprecated name for `delimiter`, please use that
        instead.
    delimiter : :class:`str`
        A single character string which separates values in the file.
    comment : :class:`str` or :obj:`list` of :obj:`str`
        Skip lines beginning with the given string if the string is a single
        character. Otherwise, skip lines that match the regex specified. Multiple
        comment characters or patterns should be passed as a list.

    Returns
    -------
    :class:`.MatrixTable`
        MatrixTable constructed from imported data.
    """
    row_key = wrap_to_list(row_key)
    comment = wrap_to_list(comment)
    paths = [hl.current_backend().fs.canonicalize_path(p) for p in wrap_to_list(paths)]
    missing_list = wrap_to_list(missing)

    def comment_filter(table):
        return hl.rbind(hl.array(comment),
                        lambda hl_comment: hl_comment.any(lambda com: hl.if_else(hl.len(com) == 1,
                                                                                 table.text.startswith(com),
                                                                                 table.text.matches(com, False)))) \
            if len(comment) > 0 else False

    def truncate(string_array, delim=", "):
        if len(string_array) > 10:
            string_array = string_array[:10]
            string_array.append("...")
        return delim.join(string_array)

    path_to_index = {path: idx for idx, path in enumerate(paths)}

    def format_file(file_name, hl_value=False):
        if hl_value:
            return hl.rbind(file_name.split('/'), lambda split_file:
                            hl.if_else(hl.len(split_file) <= 4, hl.str("/").join(file_name.split('/')[-4:]),
                                       hl.str("/") + hl.str("/").join(file_name.split('/')[-4:])))
        else:
            return "/".join(file_name.split('/')[-3:]) if len(file_name) <= 4 else \
                "/" + "/".join(file_name.split('/')[-3:])

    file_start_array = None

    def get_file_start(row):
        nonlocal file_start_array
        if file_start_array is None:
            collect_expr = first_lines_table.collect(_localize=False).map(lambda line: (line.file, line.idx))
            file_start_array = hl.literal(hl.eval(collect_expr), dtype=collect_expr.dtype)
        return hl.coalesce(
            file_start_array.filter(lambda line_tuple: line_tuple[0] == row.file).map(
                lambda line_tuple: line_tuple[1]).first(),
            0)

    def validate_row_fields():
        unique_fields = {}
        duplicates = []
        header_idx = 0
        for header_rowf in header_dict['row_fields']:
            rowf_type = row_fields.get(header_rowf)
            if rowf_type is None:
                import itertools as it
                row_fields_string = '\n'.join(list(it.starmap(
                    lambda row_field, row_type: f"      '{row_field}': {str(row_type)}", row_fields.items())))
                header_fields_string = "\n      ".join(map(lambda field: f"'{field}'", header_dict['row_fields']))
                raise FatalError(f"in file {format_file(header_dict['path'])} found row field '{header_rowf}' that's"
                                 f" not in 'row fields'\nrow fields found in file:\n      {header_fields_string}"
                                 f"\n'row fields':\n{row_fields_string}")
            if header_rowf in unique_fields:
                duplicates.append(header_rowf)
            else:
                unique_fields[header_rowf] = True
            header_idx += 1
        if len(duplicates) > 0:
            raise FatalError("Found following duplicate row fields in header:\n" + '\n'.join(duplicates))

    def parse_entries(row):
        return hl.range(num_of_row_fields, len(header_dict['column_ids']) + num_of_row_fields).map(
            lambda entry_idx: parse_type_or_error(entry_type, row, entry_idx, not_entries=False))

    def parse_rows(row):
        rows_list = list(row_fields.items())
        return {rows_list[idx][0]:
                parse_type_or_error(rows_list[idx][1], row, idx) for idx in range(num_of_row_fields)}

    def error_msg(row, idx, msg):
        return (hl.str("in file ") + hl.str(format_file(row.file, True))
                + hl.str(" on line ") + hl.str(row.row_id - get_file_start(row) + 1)
                + hl.str(" at value '") + hl.str(row.split_array[idx]) + hl.str("':\n") + hl.str(msg))

    def parse_type_or_error(hail_type, row, idx, not_entries=True):
        value = row.split_array[idx]
        if hail_type == hl.tint32:
            parsed_type = hl.parse_int32(value)
        elif hail_type == hl.tint64:
            parsed_type = hl.parse_int64(value)
        elif hail_type == hl.tfloat32:
            parsed_type = hl.parse_float32(value)
        elif hail_type == hl.tfloat64:
            parsed_type = hl.parse_float64(value)
        else:
            parsed_type = value

        if not_entries:
            error_clarify_msg = hl.str(" at row field '") + hl.str(hl_row_fields[idx]) + hl.str("'")
        else:
            error_clarify_msg = (hl.str(" at column id '") + hl.str(hl_columns[idx - num_of_row_fields])
                                 + hl.str("' for entry field 'x' "))

        return hl.if_else(hl.is_missing(value), hl.missing(hail_type),
                          hl.case().when(~hl.is_missing(parsed_type), parsed_type)
                          .or_error(
                              error_msg(row, idx, f"error parsing value into {str(hail_type)}" + error_clarify_msg)))

    num_of_row_fields = len(row_fields.keys())
    add_row_id = False
    if len(row_key) == 0:
        add_row_id = True
        row_key = ['row_id']

    if sep is not None:
        if delimiter is not None:
            raise ValueError(
                f'expecting either sep or delimiter but received both: '
                f'{sep}, {delimiter}')
        delimiter = sep
    del sep

    if delimiter is None:
        delimiter = '\t'
    if len(delimiter) != 1:
        raise FatalError('delimiter or sep must be a single character')

    if add_row_id:
        if 'row_id' in row_fields:
            raise FatalError(
                "import_matrix_table reserves the field name 'row_id' for"
                'its own use, please use a different name')

    for k, v in row_fields.items():
        if v not in {tint32, tint64, tfloat32, tfloat64, tstr}:
            raise FatalError(
                f'import_matrix_table expects field types to be one of:'
                f"'int32', 'int64', 'float32', 'float64', 'str': field {repr(k)} had type '{v}'")

    if entry_type not in {tint32, tint64, tfloat32, tfloat64, tstr}:
        raise FatalError("""import_matrix_table expects entry types to be one of:
        'int32', 'int64', 'float32', 'float64', 'str': found '{}'""".format(entry_type))

    if missing in delimiter:
        raise FatalError(f"Missing value {missing} contains delimiter {delimiter}")

    ht = import_lines(paths, min_partitions, force_bgz=force_bgz).add_index(name='row_id')
    # for checking every header matches
    file_per_partition = import_lines(paths, force_bgz=force_bgz, file_per_partition=True)
    file_per_partition = file_per_partition.filter(hl.bool(hl.len(file_per_partition.text) == 0)
                                                   | comment_filter(file_per_partition), False)
    first_lines_table = file_per_partition._map_partitions(lambda rows: rows.take(1))
    first_lines_table = first_lines_table.annotate(split_array=first_lines_table.text.split(delimiter)).add_index()

    if not no_header:
        def validate_header_get_info_dict():
            two_first_lines = file_per_partition.head(2)
            two_first_lines = two_first_lines.annotate(split_array=two_first_lines.text.split(delimiter)).collect()
            header_line = two_first_lines[0] if two_first_lines else None
            first_data_line = two_first_lines[1] if len(two_first_lines) > 1 else None
            num_of_data_line_values = len(first_data_line.split_array) if len(two_first_lines) > 1 else 0
            num_of_header_values = len(header_line.split_array) if two_first_lines else 0
            if header_line is None or path_to_index[header_line.file] != 0:
                raise ValueError(f"Expected header in every file but found empty file: {format_file(paths[0])}")
            elif not first_data_line or first_data_line.file != header_line.file:
                hl.utils.warning(f"File {format_file(header_line.file)} contains a header, but no lines of data")
                if num_of_header_values < num_of_data_line_values:
                    raise ValueError(f"File {format_file(header_line.file)} contains one line assumed to be the header."
                                     f"The header had a length of {num_of_header_values} while the number"
                                     f"of row fields is {num_of_row_fields}")
                user_row_fields = header_line.split_array[:num_of_row_fields]
                column_ids = header_line.split_array[num_of_row_fields:]
            elif num_of_data_line_values != num_of_header_values:
                if num_of_data_line_values == num_of_header_values + num_of_row_fields:
                    user_row_fields = ["f" + str(f_idx) for f_idx in list(range(0, num_of_row_fields))]
                    column_ids = header_line.split_array
                else:
                    raise ValueError(
                        f"In file {format_file(header_line.file)}, expected the header line to match either:\n"
                        f"rowField0 rowField1 ... rowField${num_of_row_fields} colId0 colId1 ...\nor\n"
                        f" colId0 colId1 ...\nInstead the first two lines were:\nInstead the first two lin"
                        f"es were:\n{header_line.text}\n{first_data_line.text}\nThe first line contained"
                        f" {num_of_header_values} separated values and the second line"
                        f" contained {num_of_data_line_values}")
            else:
                user_row_fields = header_line.split_array[:num_of_row_fields]
                column_ids = header_line.split_array[num_of_row_fields:]
            return {'text': header_line.text, 'header_values': header_line.split_array, 'path': header_line.file,
                    'row_fields': user_row_fields, 'column_ids': column_ids}

        def warn_if_duplicate_col_ids():
            time_col_id_encountered_dict = {}
            duplicate_cols = []
            for item in header_dict['column_ids']:
                if time_col_id_encountered_dict.get(item) is not None:
                    duplicate_cols.append(item)
                    time_col_id_encountered_dict[item] = time_col_id_encountered_dict[item] + 1
                time_col_id_encountered_dict[item] = 1
            if len(duplicate_cols) == 0:
                return

            import itertools as it
            duplicates_to_print = sorted(
                [('"' + dup_field + '"', '(' + str(time_col_id_encountered_dict[dup_field]) + ')')
                 for dup_field in duplicate_cols], key=lambda dup_values: dup_values[1])

            duplicates_to_print = truncate(duplicates_to_print)
            duplicates_to_print_formatted = it.starmap(lambda dup, time_found: time_found
                                                       + " " + dup, duplicates_to_print)
            ht.utils.warning(f"Found {len(duplicate_cols)} duplicate column id"
                             + f"{'s' if len(duplicate_cols) > 1 else ''}\n" + '\n'.join(duplicates_to_print_formatted))

        def validate_all_headers():
            all_headers = first_lines_table.collect()
            for header in all_headers:
                if header_dict['text'] != header.text:
                    if len(header_dict['header_values']) == len(header.split_array):
                        zipped_headers = list(zip(header_dict['header_values'], header.split_array))
                        for header_idx, header_values in enumerate(zipped_headers):
                            main_header_value = header_values[0]
                            error_header_value = header_values[1]
                            if main_header_value != error_header_value:
                                raise ValueError("invalid header: expected elements to be identical for all input paths"
                                                 f". Found different elements at position {header_idx + 1}"
                                                 f"\n in file {format_file(header.file)} with value "
                                                 f"'{error_header_value}' when expecting value '{main_header_value}'")
                    else:
                        raise ValueError(f"invalid header: lengths of headers differ. \n"
                                         f"{len(header_dict['header_values'])} elements in "
                                         f"{format_file(header_dict['path'])}:\n"
                                         + truncate(["'{}'".format(value) for value in header_dict['header_values']])
                                         + f" {len(header.split_array)} elements in {format_file(header.file)}:\n"
                                         + truncate(["'{}'".format(value) for value in header.split_array]))

        header_dict = validate_header_get_info_dict()
        warn_if_duplicate_col_ids()
        validate_all_headers()

    else:
        first_line = first_lines_table.head(1).collect()
        if not first_line or path_to_index[first_line[0].file] != 0:
            hl.utils.warning(
                f"File {format_file(paths[0])} is empty and has no header, so we assume no columns")
            header_dict = {'header_values': [],
                           'row_fields': ["f" + str(f_idx) for f_idx in list(range(0, num_of_row_fields))],
                           'column_ids': []
                           }
        else:
            first_line = first_line[0]
            header_dict = {'header_values': [],
                           'row_fields': ["f" + str(f_idx) for f_idx in list(range(0, num_of_row_fields))],
                           'column_ids':
                               [col_id for col_id in list(range(0, len(first_line.split_array) - num_of_row_fields))]
                           }

    validate_row_fields()
    header_filter = ht.text == header_dict['text'] if not no_header else False

    ht = ht.filter(hl.bool(hl.len(ht.text) == 0) | comment_filter(ht) | header_filter, False)

    hl_columns = hl.array(header_dict['column_ids']) if len(header_dict['column_ids']) > 0 else hl.empty_array(hl.tstr)
    hl_row_fields = hl.array(header_dict['row_fields']) if len(header_dict['row_fields']) > 0 \
        else hl.empty_array(hl.tstr)
    ht = ht.annotate(split_array=ht.text._split_line(delimiter, missing_list, quote=None, regex=False)).add_index(
        'row_id')

    ht = ht.annotate(split_array=hl.case().when(hl.len(ht.split_array) >= num_of_row_fields, ht.split_array)
                     .or_error(error_msg(ht, hl.len(ht.split_array) - 1,
                                         " unexpected end of line while reading row field")))

    n_column_ids = len(header_dict['column_ids'])
    n_in_split_array = hl.len(ht.split_array[num_of_row_fields:(num_of_row_fields + n_column_ids)])
    ht = ht.annotate(split_array=hl.case().when(
        n_column_ids <= n_in_split_array,
        ht.split_array
    ).or_error(
        error_msg(
            ht,
            hl.len(ht.split_array) - 1,
            " unexpected end of line while reading entries"
        )
    ))

    ht = ht.annotate(**parse_rows(ht), entries=parse_entries(ht).map(lambda entry: hl.struct(x=entry)))\
        .drop('text', 'split_array', 'file')

    ht = ht.annotate_globals(cols=hl.range(0, len(header_dict['column_ids']))
                             .map(lambda col_idx: hl.struct(col_id=hl_columns[col_idx])))

    if not add_row_id:
        ht = ht.drop('row_id')

    mt = ht._unlocalize_entries('entries', 'cols', ['col_id'])
    mt = mt.key_rows_by(*row_key)
    return mt


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
           skip_invalid_loci=bool,
           n_partitions=nullable(int),
           block_size=nullable(int))
def import_plink(bed, bim, fam,
                 min_partitions=None,
                 delimiter='\\\\s+',
                 missing='NA',
                 quant_pheno=False,
                 a2_reference=True,
                 reference_genome='default',
                 contig_recoding=None,
                 skip_invalid_loci=False,
                 n_partitions=None,
                 block_size=None) -> MatrixTable:
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
        * `is_case` (:py:data:`.tbool`) -- Column 6 in the FAM file. Only
          present if `quant_pheno` equals False. Set to missing if value equals
          "-9", "0", "N/A", or the value specified by `missing`. Set to true if
          value equals "2". Set to false if value equals "1".
        * `quant_pheno` (:py:data:`.tfloat`) -- Column 6 in the FAM file. Only
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
    bed : :class:`str`
        PLINK BED file.

    bim : :class:`str`
        PLINK BIM file.

    fam : :class:`str`
        PLINK FAM file.

    min_partitions : :obj:`int`, optional
        Minimum number of partitions.  Useful in conjunction with `block_size`.

    missing : :class:`str`
        String used to denote missing values **only** for the phenotype field.
        This is in addition to "-9", "0", and "N/A" for case-control
        phenotypes.

    delimiter : :class:`str`
        FAM file field delimiter regex.

    quant_pheno : :obj:`bool`
        If ``True``, FAM phenotype is interpreted as quantitative.

    a2_reference : :obj:`bool`
        If ``True``, A2 is treated as the reference allele. If False, A1 is treated
        as the reference allele.

    reference_genome : :class:`str` or :class:`.ReferenceGenome`, optional
        Reference genome to use.

    contig_recoding : :obj:`dict` of :class:`str` to :obj:`str`, optional
        Dict of old contig name to new contig name. The new contig name must be
        in the reference genome given by ``reference_genome``. If ``None``, the
        default is dependent on the ``reference_genome``. For "GRCh37", the default
        is ``{'23': 'X', '24': 'Y', '25': 'X', '26': 'MT'}``. For "GRCh38", the
        default is ``{'1': 'chr1', ..., '22': 'chr22', '23': 'chrX', '24': 'chrY', '25': 'chrX', '26': 'chrM'}``.

    skip_invalid_loci : :obj:`bool`
        If ``True``, skip loci that are not consistent with `reference_genome`.

    n_partitions : :obj:`int`, optional
        Number of partitions.  If both `n_partitions` and `block_size`
        are specified, `n_partitions` will be used.

    block_size : :obj:`int`, optional
        Block size, in MB.  Default: 128MB blocks.

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
            contig_recoding = {**{str(i): f'chr{i}' for i in range(1, 23)},
                               **{'23': 'chrX', '24': 'chrY', '25': 'chrX', '26': 'chrM'}}
        else:
            contig_recoding = {}

    reader = ir.MatrixPLINKReader(bed, bim, fam,
                                  n_partitions, block_size, min_partitions,
                                  missing, delimiter, quant_pheno, a2_reference, reference_genome,
                                  contig_recoding, skip_invalid_loci)
    return MatrixTable(ir.MatrixRead(reader, drop_cols=False, drop_rows=False))


@typecheck(path=str,
           _intervals=nullable(sequenceof(anytype)),
           _filter_intervals=bool,
           _drop_cols=bool,
           _drop_rows=bool,
           _create_row_uids=bool,
           _create_col_uids=bool,
           _n_partitions=nullable(int),
           _assert_type=nullable(hl.tmatrix),
           _load_refs=bool)
def read_matrix_table(path, *, _intervals=None, _filter_intervals=False, _drop_cols=False,
                      _drop_rows=False, _create_row_uids=False, _create_col_uids=False,
                      _n_partitions=None, _assert_type=None, _load_refs=True) -> MatrixTable:
    """Read in a :class:`.MatrixTable` written with :meth:`.MatrixTable.write`.

    Parameters
    ----------
    path : :class:`str`
        File to read.

    Returns
    -------
    :class:`.MatrixTable`
    """
    if _load_refs:
        for rg_config in Env.backend().load_references_from_dataset(path):
            hl.ReferenceGenome._from_config(rg_config)

    if _intervals is not None and _n_partitions is not None:
        raise ValueError("'read_matrix_table' does not support both _intervals and _n_partitions")

    mt = MatrixTable(ir.MatrixRead(ir.MatrixNativeReader(path, _intervals, _filter_intervals),
                                   _drop_cols,
                                   _drop_rows,
                                   drop_row_uids=not _create_row_uids,
                                   drop_col_uids=not _create_col_uids,
                                   _assert_type=_assert_type))
    if _n_partitions:
        intervals = mt._calculate_new_partitions(_n_partitions)
        return read_matrix_table(
            path,
            _drop_rows=_drop_rows,
            _drop_cols=_drop_cols,
            _intervals=intervals,
            _assert_type=_assert_type,
            _load_refs=_load_refs
        )
    return mt


@typecheck(path=str)
def get_vcf_metadata(path):
    """Extract metadata from VCF header.

    Examples
    --------

    >>> hl.get_vcf_metadata('data/example2.vcf.bgz')  # doctest: +SKIP_OUTPUT_CHECK
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
    path : :class:`str`
        VCF file(s) to read. If more than one file is given, the first
        file is used.

    Returns
    -------
    :obj:`dict` of :class:`str` to (:obj:`dict` of :obj:`str` to (:obj:`dict` of :obj:`str` to :obj:`str`))
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
           n_partitions=nullable(int),
           block_size=nullable(int))
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
               n_partitions=None,
               block_size=None) -> MatrixTable:
    """Import VCF file(s) as a :class:`.MatrixTable`.

    Examples
    --------

    Import a standard bgzipped VCF with GRCh37 as the reference genome.

    >>> ds = hl.import_vcf('data/example2.vcf.bgz', reference_genome='GRCh37')

    Import a VCF with GRCh38 as the reference genome that incorrectly uses the
    contig names from GRCh37 (i.e. uses contig name "1" instead of "chr1").

    >>> recode = {f"{i}":f"chr{i}" for i in (list(range(1, 23)) + ['X', 'Y'])}
    >>> ds = hl.import_vcf('data/grch38_bad_contig_names.vcf', reference_genome='GRCh38', contig_recoding=recode)

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
    path : :class:`str` or :obj:`list` of :obj:`str`
        VCF file(s) to read.
    force : :obj:`bool`
        If ``True``, load **.vcf.gz** files serially. No downstream operations
        can be parallelized, so this mode is strongly discouraged.
    force_bgz : :obj:`bool`
        If ``True``, load **.vcf.gz** files as blocked gzip files, assuming
        that they were actually compressed using the BGZ codec.
    header_file : :class:`str`, optional
        Optional header override file. If not specified, the first file in
        `path` is used.
    min_partitions : :obj:`int`, optional
        Minimum partitions to load per file.
    drop_samples : :obj:`bool`
        If ``True``, create sites-only dataset. Don't load sample IDs or
        entries.
    call_fields : :obj:`list` of :class:`str`
        List of FORMAT fields to load as :py:data:`.tcall`. "GT" is
        loaded as a call automatically.
    reference_genome: :class:`str` or :class:`.ReferenceGenome`, optional
        Reference genome to use.
    contig_recoding: :obj:`dict` of (:class:`str`, :obj:`str`), optional
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
    filter : :class:`str`, optional
        Line filter regex. A partial match results in the line being removed
        from the file. Applies before `find_replace`, if both are defined.
    find_replace : (:class:`str`, :obj:`str`)
        Line substitution regex. Functions like ``re.sub``, but obeys the exact
        semantics of Java's
        `String.replaceAll <https://docs.oracle.com/javase/8/docs/api/java/lang/String.html#replaceAll-java.lang.String-java.lang.String->`__.
    n_partitions : :obj:`int`, optional
        Number of partitions.  If both `n_partitions` and `block_size`
        are specified, `n_partitions` will be used.
    block_size : :obj:`int`, optional
        Block size, in MB.  Default: 128MB blocks.

    Returns
    -------
    :class:`.MatrixTable`
    """

    reader = ir.MatrixVCFReader(path, call_fields, entry_float_type, header_file,
                                n_partitions, block_size, min_partitions,
                                reference_genome, contig_recoding, array_elements_required,
                                skip_invalid_loci, force_bgz, force, filter, find_replace)
    return MatrixTable(ir.MatrixRead(reader, drop_cols=drop_samples))


@typecheck(path=sequenceof(str),
           partitions=expr_any,
           call_fields=oneof(str, sequenceof(str)),
           entry_float_type=enumeration(tfloat32, tfloat64),
           reference_genome=nullable(reference_genome_type),
           contig_recoding=nullable(dictof(str, str)),
           array_elements_required=bool,
           skip_invalid_loci=bool,
           filter=nullable(str),
           find_replace=nullable(sized_tupleof(str, str)),
           _external_sample_ids=nullable(sequenceof(sequenceof(str))),
           _external_header=nullable(str))
def import_gvcfs(path,
                 partitions,
                 call_fields=['PGT'],
                 entry_float_type=tfloat64,
                 reference_genome='default',
                 contig_recoding=None,
                 array_elements_required=True,
                 skip_invalid_loci=False,
                 filter=None,
                 find_replace=None,
                 _external_sample_ids=None,
                 _external_header=None) -> List[MatrixTable]:
    """(Experimental) Import multiple vcfs as multiple :class:`.MatrixTable`.

    .. include:: ../_templates/experimental.rst

    All files described by the ``path`` argument must be block gzipped VCF
    files. They must all be tabix indexed. Because of this requirement, no
    ``force`` or ``force_bgz`` arguments are present. Otherwise, the arguments
    to this function are almost identical to :func:`.import_vcf`.  However, this
    function also requrires a ``partitions`` argument, which is used to divide
    and filter the vcfs.  It must be an expression or literal of type
    ``array<interval<struct{locus:locus<RG>}>>``. A partition will be created
    for every element of the array. Loci that fall outside of any interval will
    not be imported. For example:

    .. code-block:: python

        [hl.Interval(hl.Locus("chr22", 1), hl.Locus("chr22", 5332423), includes_end=True)]

    The ``includes_start`` and ``includes_end`` keys must be ``True``. The
    ``contig`` fields must be the same.

    One difference between :func:`.import_gvcfs` and :func:`.import_vcf` is that
    :func:`.import_gvcfs` only keys the resulting matrix tables by ``locus``
    rather than ``locus, alleles``.
    """
    hl.utils.no_service_backend('import_gvcfs')
    rg = reference_genome.name if reference_genome else None

    partitions, partitions_type = hl.utils._dumps_partitions(partitions, hl.tstruct(locus=hl.tlocus(rg),
                                                                                    alleles=hl.tarray(hl.tstr)))

    vector_ref_s = Env.spark_backend('import_vcfs')._jbackend.pyImportVCFs(
        wrap_to_list(path),
        wrap_to_list(call_fields),
        entry_float_type._parsable_string(),
        rg,
        contig_recoding,
        array_elements_required,
        skip_invalid_loci,
        partitions, partitions_type._parsable_string(),
        filter,
        find_replace[0] if find_replace is not None else None,
        find_replace[1] if find_replace is not None else None,
        _external_sample_ids,
        _external_header)
    vector_ref = json.loads(vector_ref_s)
    jir_vref = ir.JIRVectorReference(vector_ref['vector_ir_id'],
                                     vector_ref['length'],
                                     hl.tmatrix._from_json(vector_ref['type']))

    return [MatrixTable(ir.JavaMatrixVectorRef(jir_vref, idx)) for idx in range(len(jir_vref))]


def import_vcfs(path,
                partitions,
                call_fields=['PGT'],
                entry_float_type=tfloat64,
                reference_genome='default',
                contig_recoding=None,
                array_elements_required=True,
                skip_invalid_loci=False,
                filter=None,
                find_replace=None,
                _external_sample_ids=None,
                _external_header=None) -> List[MatrixTable]:
    """This function is deprecated, use :func:`.import_gvcfs` instead"""
    return import_gvcfs(path,
                        partitions,
                        call_fields,
                        entry_float_type,
                        reference_genome,
                        contig_recoding,
                        array_elements_required,
                        skip_invalid_loci,
                        filter,
                        find_replace,
                        _external_sample_ids,
                        _external_header)


@typecheck(path=oneof(str, sequenceof(str)),
           index_file_map=nullable(dictof(str, str)),
           reference_genome=nullable(reference_genome_type),
           contig_recoding=nullable(dictof(str, str)),
           skip_invalid_loci=bool,
           _buffer_size=int)
def index_bgen(path,
               index_file_map=None,
               reference_genome='default',
               contig_recoding=None,
               skip_invalid_loci=False,
               _buffer_size=16_000_000):
    """Index BGEN files as required by :func:`.import_bgen`.

    If `index_file_map` is unspecified, then, for each BGEN file, the index file is written in the
    same directory and as the associated BGEN file with the same filename appended by
    `.idx2`. Otherwise, the `index_file_map` must specify a distinct `idx2` path for each BGEN file.

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
    path : :class:`str` or :obj:`list` of :obj:`str`
        The .bgen files to index. May be one of: a BGEN file path, a list of BGEN file paths, or the
        path of a directory that contains BGEN files.
    index_file_map : :obj:`dict` of :class:`str` to :obj:`str`, optional
        Dict of BGEN file to index file location. Index file location must have
        a `.idx2` file extension. Cannot use Hadoop glob patterns in file names.
    reference_genome : :class:`str` or :class:`.ReferenceGenome`, optional
        Reference genome to use.
    contig_recoding : :obj:`dict` of :class:`str` to :obj:`str`, optional
        Dict of old contig name to new contig name. The new contig name must be
        in the reference genome given by `reference_genome`.
    skip_invalid_loci : :obj:`bool`
        If ``True``, skip loci that are not consistent with `reference_genome`.

    """
    rg_t = hl.tlocus(reference_genome) if reference_genome else hl.tstruct(contig=hl.tstr, position=hl.tint32)
    if index_file_map is None:
        index_file_map = {}
    if contig_recoding is None:
        contig_recoding = {}
    raw_paths = wrap_to_list(path)

    fs = hl.current_backend().fs
    paths = []
    for p in raw_paths:
        if fs.is_file(p):
            paths.append(p)
        else:
            if not fs.is_dir(p):
                raise ValueError(f'index_bgen: no file or directory at {p}')
            for stat_result in fs.ls(p):
                if re.match(r"^.*part-[0-9]+(-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})?$",
                            os.path.basename(stat_result.path)):
                    paths.append(stat_result.path)

    paths_lit = hl.literal(paths, hl.tarray(hl.tstr))
    index_file_map_lit = hl.literal(index_file_map, hl.tdict(hl.tstr, hl.tstr))
    for k, v in index_file_map.items():
        if not v.endswith('.idx2'):
            raise FatalError(f"index file for {k} is missing a .idx2 file extension")
    contig_recoding_lit = hl.literal(contig_recoding, hl.tdict(hl.tstr, hl.tstr))
    ht = hl.utils.range_table(len(paths), len(paths))
    path_fd = paths_lit[ht.idx]
    ht = ht.annotate(n_indexed=hl.expr.functions._func(
        "index_bgen",
        hl.tint64,
        path_fd,
        index_file_map_lit.get(path_fd, path_fd + ".idx2"),
        contig_recoding_lit,
        hl.bool(skip_invalid_loci),
        hl.int32(_buffer_size),
        type_args=(rg_t,)))

    for r in ht.collect():
        idx = r.idx
        n = r.n_indexed
        path = paths[idx]
        idx_path = index_file_map.get(path, path)
        info(f"indexed {n} sites in {path} at {idx_path}")


@typecheck(path=str,
           _intervals=nullable(sequenceof(anytype)),
           _filter_intervals=bool,
           _n_partitions=nullable(int),
           _assert_type=nullable(hl.ttable),
           _load_refs=bool,
           _create_row_uids=bool)
def read_table(path,
               *,
               _intervals=None,
               _filter_intervals=False,
               _n_partitions=None,
               _assert_type=None,
               _load_refs=True,
               _create_row_uids=False) -> Table:
    """Read in a :class:`.Table` written with :meth:`.Table.write`.

    Parameters
    ----------
    path : :class:`str`
        File to read.

    Returns
    -------
    :class:`.Table`
    """
    if _load_refs:
        for rg_config in Env.backend().load_references_from_dataset(path):
            hl.ReferenceGenome._from_config(rg_config)

    if _intervals is not None and _n_partitions is not None:
        raise ValueError("'read_table' does not support both _intervals and _n_partitions")
    tr = ir.TableNativeReader(path, _intervals, _filter_intervals)
    ht = Table(ir.TableRead(tr, False, drop_row_uids=not _create_row_uids, _assert_type=_assert_type))

    if _n_partitions:
        intervals = ht._calculate_new_partitions(_n_partitions)
        return read_table(path, _intervals=intervals, _assert_type=_assert_type, _load_refs=_load_refs, _create_row_uids=_create_row_uids)
    return ht


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

    By default, this method supports Elasticsearch versions 6.8.x - 7.x.x. Older versions of elasticsearch will require
    recompiling hail.

    .. warning::
        :func:`.export_elasticsearch` is EXPERIMENTAL.

    .. note::
        Table rows may be exported more than once. For example, if a task has to be retried after being preempted
        midway through processing a partition. To avoid duplicate documents in Elasticsearch, use a `config` with the
        `es.mapping.id <https://www.elastic.co/guide/en/elasticsearch/hadoop/current/configuration.html#cfg-mapping>`__
        option set to a field that contains a unique value for each row.
    """

    jdf = t.expand_types().to_spark(flatten=False)._jdf
    Env.hail().io.ElasticsearchConnector.export(jdf, host, port, index, index_type, block_size, config, verbose)


@typecheck(paths=sequenceof(str), key=nullable(sequenceof(str)), intervals=nullable(sequenceof(anytype)))
def import_avro(paths, *, key=None, intervals=None):
    if not paths:
        raise ValueError('import_avro requires at least one path')
    if (key is None) != (intervals is None):
        raise ValueError('key and intervals must either be both defined or both undefined')

    with hl.current_backend().fs.open(paths[0], 'rb') as avro_file:

        # monkey patch DataFileReader.determine_file_length to account for bug in Google HadoopFS

        def patched_determine_file_length(self) -> int:
            remember_pos = self.reader.tell()
            self.reader.seek(-1, 2)
            file_length = self.reader.tell() + 1
            self.reader.seek(remember_pos)
            return file_length

        original_determine_file_length = DataFileReader.determine_file_length

        try:
            DataFileReader.determine_file_length = patched_determine_file_length

            with DataFileReader(avro_file, DatumReader()) as data_file_reader:
                tr = ir.AvroTableReader(avro.schema.parse(data_file_reader.schema), paths, key, intervals)

        finally:
            DataFileReader.determine_file_length = original_determine_file_length

    return Table(ir.TableRead(tr))


@typecheck(paths=oneof(str, sequenceof(str)),
           key=table_key_type,
           min_partitions=nullable(int),
           impute=bool,
           no_header=bool,
           comment=oneof(str, sequenceof(str)),
           missing=oneof(str, sequenceof(str)),
           types=dictof(str, hail_type),
           skip_blank_lines=bool,
           force_bgz=bool,
           filter=nullable(str),
           find_replace=nullable(sized_tupleof(str, str)),
           force=bool,
           source_file_field=nullable(str))
def import_csv(paths,
               key=None,
               min_partitions=None,
               impute=False,
               no_header=False,
               comment=(),
               missing="NA",
               types={},
               skip_blank_lines=False,
               force_bgz=False,
               filter=None,
               find_replace=None,
               force=False,
               source_file_field=None) -> Table:
   """Import a csv file as a :class:`.Table`.

    Examples
    --------

    Let's import fields from a CSV file with missing data:

    .. code-block:: text

        $ cat data/samples2.csv
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

    - Pass the non-default missing value ``.``

    >>> table = hl.import_csv('data/samples2.csv', missing='.')

    Notes
    -----

    The `impute` parameter tells Hail to scan the file an extra time to gather
    information about possible field types. While this is a bit slower for large
    files because the file is parsed twice, the convenience is often worth this
    cost.

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
    table. It is an :obj:`dict` keyed by :class:`str`, with :class:`.HailType` values.
    See the examples above for a standard usage. Additionally, this option can
    be used to override type imputation. For example, if the field
    ``Chromosome`` only contains the values ``1`` through ``22``, it will be
    imputed to have type :py:data:`.tint32`, whereas most Hail methods expect
    that a chromosome field will be of type :py:data:`.tstr`. Setting
    ``impute=True`` and ``types={'Chromosome': hl.tstr}`` solves this problem.

    Parameters
    ----------

    paths : :class:`str` or :obj:`list` of :obj:`str`
        Files to import.
    key : :class:`str` or :obj:`list` of :obj:`str`
        Key fields(s).
    min_partitions : :obj:`int` or :obj:`None`
        Minimum number of partitions.
    no_header : :obj:`bool`
        If ``True```, assume the file has no header and name the N fields `f0`,
        `f1`, ... `fN` (0-indexed).
    impute : :obj:`bool`
        If ``True``, Impute field types from the file.
    comment : :class:`str` or :obj:`list` of :obj:`str`
        Skip lines beginning with the given string if the string is a single
        character. Otherwise, skip lines that match the regex specified. Multiple
        comment characters or patterns should be passed as a list.
    missing : :class:`str` or :obj:`list` [:obj:`str`]
        Identifier(s) to be treated as missing.
    types : :obj:`dict` mapping :class:`str` to :class:`.HailType`
        Dictionary defining field types.
    skip_blank_lines : :obj:`bool`
        If ``True``, ignore empty lines. Otherwise, throw an error if an empty
        line is found.
    force_bgz : :obj:`bool`
        If ``True``, load files as blocked gzip files, assuming
        that they were actually compressed using the BGZ codec. This option is
        useful when the file extension is not ``'.bgz'``, but the file is
        blocked gzip, so that the file can be read in parallel and not on a
        single node.
    filter : :class:`str`, optional
        Line filter regex. A partial match results in the line being removed
        from the file. Applies before `find_replace`, if both are defined.
    find_replace : (:class:`str`, :obj:`str`)
        Line substitution regex. Functions like ``re.sub``, but obeys the exact
        semantics of Java's
        `String.replaceAll <https://docs.oracle.com/javase/8/docs/api/java/lang/String.html#replaceAll-java.lang.String-java.lang.String->`__.
    force : :obj:`bool`
        If ``True``, load gzipped files serially on one core. This should
        be used only when absolutely necessary, as processing time will be
        increased due to lack of parallelism.
    source_file_field : :class:`str`, optional
        If defined, the source file name for each line will be a field of the table
        with this name. Can be useful when importing multiple tables using glob patterns.
    Returns
    -------
    :class:`.Table`
    """

   ht = hl.import_table(paths, key=None, min_partitions=None, impute=False, no_header=False, comment=(), missing="NA",
                        types={}, skip_blank_lines=False, force_bgz=False, filter=None, find_replace=None,
                        force=False, source_file_field=None, delimiter = ",", quote ='"')
   return ht