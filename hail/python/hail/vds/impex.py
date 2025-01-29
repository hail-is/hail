import functools

import hail as hl
from hail import ir
from hail.expr.types import tarray, tcall, tfloat32, tfloat64, tint32, tstruct
from hail.genetics.reference_genome import reference_genome_type
from hail.typecheck import dictof, enumeration, nullable, oneof, sequenceof, sized_tupleof, typecheck
from hail.utils.java import info, warning

from .functions import lgt_to_gt
from .methods import to_merged_sparse_mt
from .variant_dataset import VariantDataset


@typecheck(
    dataset=VariantDataset,
    output=str,
    append_to_header=nullable(str),
    parallel=nullable(ir.ExportType.checker),
    metadata=nullable(dictof(str, dictof(str, dictof(str, str)))),
    tabix=bool,
)
def export_vcf(
    dataset,
    output,
    *,
    append_to_header=None,
    parallel=None,
    metadata=None,
    tabix=False,
):
    """Export a :class:`.VariantDataset` as an SVCR-VCF file.

    .. include:: _templates/experimental.rst

    Examples
    --------
    Export to VCF as a block-compressed file:

    >>> hl.vds.export_vcf(dataset, 'output/example.svcr.vcf.bgz')  # doctest: +SKIP

    Notes
    -----
    While the innovations present in VDS/SVCR have been standardized in the
    `VCF 4.5 spec <https://samtools.github.io/hts-specs/VCFv4.5.pdf>`__, at current, the library
    that we use to parse VCF headers does not support VCF 4.5, so we instead output a standard,
    widely supported version 4.2 VCF like :func:`hail.export_vcf`. We expect this to change in the
    future as support for VCF 4.5 improves.

    All recommendations of :func:`hail.export_vcf` apply to this method, and the arguments are
    identical. Please refer to the documentation of that method as all recommendations and warnings
    apply to this method as well.

    Note
    ----
    Some data transformations take place before export.

    #. If the local alleles field, ``LA`` is present, it is converted to the standardized local
       alternate alleles field, ``LAA``.
    #. Both of ``LGT``, and ``LPGT`` are converted to ``GT`` and ``PGT``, using
       :func:`.vds.lgt_to_gt`, if necessary.
    #. The ``gvcf_info`` field produced by the :class:`hail.vds.combiner.VariantDatasetCombiner` is
       dropped. VCF does not support ``struct`` type entry fields. To preserve this data, flatten
       the struct onto the entries::

            flat_gvcf_info = {f'gi_{name}': vds.variant_data.gvcf_info[name] for name in vds.variant_data.gvcf_info}
            vds.variant_data = vds.variant_data.transmute_entries(**flat_gvcf_info)

    Parameters
    ----------
    dataset : :class:`.VariantDataset`
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
    ref, var = dataset.reference_data, dataset.variant_data

    # fix up ref data
    if 'END' in ref.entry:
        ref = ref.drop('END')

    if 'gvcf_info' in var.entry and isinstance(var.gvcf_info.dtype, tstruct):
        var = var.drop('gvcf_info')

    if 'LGT' in var.entry:
        if 'GT' not in var.entry:
            var = var.annotate_entries(GT=lgt_to_gt(var.LGT, var.LA))
        var = var.drop('LGT')
    if 'LPGT' in var.entry:
        if 'PGT' not in var.entry:
            var = var.annotate_entries(PGT=lgt_to_gt(var.LPGT, var.LA))
        var = var.drop('LPGT')

    if 'LA' in var.entry:
        var = var.transmute_entries(LAA=var.LA[1:])

    # TODO(chrisvittal):
    #   In a future enhancement, add appropriate metadata for all 'local' fields. We do not do this
    #   at this time (Oct. 2024), as htsjdk, which we use for VCF header parsing, does not support
    #   the vcf 4.5 changes necessary for completely correct metadata.

    extra_header = (
        '##SVCR="This is a VCF that implements hail\'s Scalable Variant Call '
        'Representation. See https://doi.org/10.1101/2024.01.09.574205 for more '
        'information."\n'
    )
    if VariantDataset.ref_block_max_length_field in ref.globals:
        rbml = hl.eval(ref[VariantDataset.ref_block_max_length_field])
        extra_header += f'##ref_block_max_length={rbml}\n'

    fs = hl.current_backend().fs
    if append_to_header:
        with fs.open(append_to_header, 'r') as header_file:
            extra_header += header_file.read()

    new_header_file_path = hl.utils.new_temp_file('vds-append-to-header', 'txt')
    with fs.open(new_header_file_path, 'w') as new_header_file:
        new_header_file.write(extra_header)

    vcf = to_merged_sparse_mt(VariantDataset(reference_data=ref, variant_data=var))
    hl.export_vcf(vcf, output, append_to_header=new_header_file_path, parallel=parallel, metadata=metadata, tabix=tabix)


@typecheck(
    path=oneof(str, sequenceof(str)),
    is_split=bool,
    ref_block_fields=sequenceof(str),
    infer_ref_block_fields=bool,
    force=bool,
    force_bgz=bool,
    header_file=nullable(str),
    min_partitions=nullable(int),
    call_fields=oneof(str, sequenceof(str)),
    reference_genome=nullable(reference_genome_type),
    contig_recoding=nullable(dictof(str, str)),
    array_elements_required=bool,
    skip_invalid_loci=bool,
    entry_float_type=enumeration(tfloat32, tfloat64),
    filter=nullable(str),
    find_replace=nullable(sized_tupleof(str, str)),
    n_partitions=nullable(int),
    block_size=nullable(int),
    _create_row_uids=bool,
    _create_col_uids=bool,
)
def import_vcf(
    path,
    *,
    is_split=False,
    ref_block_fields=(),
    infer_ref_block_fields=True,
    force=False,
    force_bgz=False,
    header_file=None,
    min_partitions=None,
    call_fields=('LPGT', 'PGT'),
    reference_genome='default',
    contig_recoding=None,
    array_elements_required=False,
    skip_invalid_loci=False,
    entry_float_type=tfloat64,
    filter=None,
    find_replace=None,
    n_partitions=None,
    block_size=None,
    _create_row_uids=False,
    _create_col_uids=False,
) -> VariantDataset:
    """Import SVCR-VCF file(s) as a :class:`.VariantDataset`.

    .. include:: _templates/experimental.rst

    Notes
    -----
    The internal implementation of this method is a combination of :func:`hail.import_vcf` and
    :func:`VariantDataset.from_merged_representation`. Refer to the documentation of those methods
    for detailed usage of the parameters of both. The ``drop_samples`` parameter from
    :func:`hail.import_vcf` is not present as a 'sites only' VDS is nonsense.

    Note
    ----
    The following validations and transformations take place:

    #. The ``LEN`` FORMAT field must exist and be of type ``Integer``
    #. One of ``GT`` or ``LGT`` must be a FORMAT field of type ``String`` and must represent
       a :py:data:`.tcall`
    #. If ``is_split`` is ``False`` (the default), one of ``LA`` or ``LAA``, the local (alternate)
       alleles field must be a FORMAT field of type ``array<int32>``.
    #. ``LAA`` is transformed back into ``LA`` if ``LA`` is not already present, and then ``LAA`` is
       dropped.
    #. Entries are filtered to just those with present ``GT`` or ``LGT``.

    Parameters
    ----------
    path : :class:`str` or :obj:`list` of :obj:`str`
        One or more paths to VCF files to read. Each path may include glob expressions like ``*``,
        ``?``, or ``[abc123]``, which, if present, will be expanded.
    is_split : :obj:`bool`
        If ``True`` indicate that this is a dataset where multi-allelic variants have been split.
        When ``True``, the ``LA``/``LAA`` fields need not be present.
    ref_block_fields : :obj:`list` of :obj:`str`
        These fields are automatically added to the reference data.
    infer_ref_block_fields : :obj:`bool`
        If True, use a small sample of the data to determine what fields are defined for reference
        data.
    force : :obj:`bool`
        If ``True``, load **.vcf.gz** files serially. No downstream operations
        can be parallelized, so this mode is strongly discouraged.
    force_bgz : :obj:`bool`
        If ``True``, load **.vcf.gz** files as blocked gzip files, assuming that they were actually
        compressed using the BGZ codec.
    header_file : :class:`str`, optional
        Optional header override file. If not specified, the first file in
        `path` is used. Glob patterns are not allowed in the `header_file`.
    min_partitions : :obj:`int`, optional
        Minimum partitions to load per file.
    call_fields : :obj:`list` of :class:`str`
        List of FORMAT fields to load as :py:data:`.tcall`. "GT" and "LGT" are
        loaded as calls automatically.
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
        `String.replaceAll <https://docs.oracle.com/en/java/javase/11/docs/api/java.base/java/lang/String.html#replaceAll(java.lang.String,java.lang.String)>`__.
    n_partitions : :obj:`int`, optional
        Number of partitions.  If both `n_partitions` and `block_size`
        are specified, `n_partitions` will be used.
    block_size : :obj:`int`, optional
        Block size, in MB.  Default: 128MB blocks.

    Returns
    -------
    :class:`.VariantDataset`
    """
    if isinstance(call_fields, str):
        call_fields = tuple({'LGT', call_fields})
    else:  # call_fields is a sequence
        call_fields = tuple({'LGT', *call_fields})

    # read ref_block_max_length from header
    if header_file is None:
        if isinstance(path, str):
            paths = [entry['path'] for entry in hl.hadoop_ls(path)]
        else:
            paths = [entry['path'] for item in path for entry in hl.hadoop_ls(item)]
        if not paths:
            raise ValueError(f'File(s) in {path} refer to no files')
        header_file = paths[0]

    ref_block_max_length = None
    with hl.hadoop_open(header_file) as header:  # use hadoop_open to handle gzip
        for _line in header:
            line = _line.strip()
            if not line:
                continue
            if not line.startswith('##'):
                break
            if (newline := line.removeprefix('##ref_block_max_length=')) != line:
                try:
                    ref_block_max_length = int(newline)
                    if ref_block_max_length <= 0:
                        raise ValueError
                except ValueError:
                    ref_block_max_length = None  # ensure this is None
                    warning(f"invalid ref_block_max_length '{newline}', ignoring it")

    vcf = hl.import_vcf(
        path=path,
        force=force,
        force_bgz=force_bgz,
        header_file=header_file,
        min_partitions=min_partitions,
        drop_samples=False,
        call_fields=call_fields,
        reference_genome=reference_genome,
        contig_recoding=contig_recoding,
        array_elements_required=array_elements_required,
        skip_invalid_loci=skip_invalid_loci,
        entry_float_type=entry_float_type,
        filter=filter,
        find_replace=find_replace,
        n_partitions=n_partitions,
        block_size=block_size,
        _create_row_uids=_create_row_uids,
        _create_col_uids=_create_col_uids,
    )

    if 'LEN' not in vcf.entry or vcf.LEN.dtype != tint32:
        raise ValueError('Invalid SVCR-VCF: expected `LEN` of type `int32` in FORMAT fields')

    gt_fields = [field for field in ('GT', 'LGT') if field in vcf.entry and vcf[field].dtype == tcall]
    if not gt_fields:
        raise ValueError('Invalid SVCR-VCF: expected at least one field of type `call` named `GT` or `LGT`')

    if not is_split:
        has_la = 'LA' in vcf.entry
        has_laa = 'LAA' in vcf.entry
        if not (has_la or has_laa):
            raise ValueError('Invalid SVCR-VCF: expected one of `LA` or `LAA` in FORMAT fields')
        if has_la and vcf.LA.dtype != tarray(tint32):
            raise ValueError('Invalid SVCR-VCF: `LA` field must have type `array<int32>`')
        if has_laa and vcf.LAA.dtype != tarray(tint32):
            raise ValueError('Invalid SVCR-VCF: `LAA` field must have type `array<int32>`')
        if has_laa:
            if has_la:
                info('hail.vds.import_vcf: SVCR-VCF has both `LA` and `LAA`, keeping only `LA`')
                vcf = vcf.drop('LAA')
            else:
                vcf = vcf.transmute_entries(LA=hl.array([0]).extend(vcf.LAA))

    vcf = vcf.filter_entries(functools.reduce(lambda p, q: p | q, (hl.is_defined(vcf[gt]) for gt in gt_fields)))

    vds = VariantDataset.from_merged_representation(
        vcf,
        ref_block_indicator_field='LEN',
        ref_block_fields=ref_block_fields,
        infer_ref_block_fields=infer_ref_block_fields,
        is_split=is_split,
    )

    if ref_block_max_length is not None:
        vds.reference_data = vds.reference_data.annotate_globals(**{
            VariantDataset.ref_block_max_length_field: ref_block_max_length
        })

    return vds
