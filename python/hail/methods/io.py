from hail.typecheck import *
from hail.utils.java import Env, handle_py4j, joption
from hail.api2 import MatrixTable
from hail.history import *
from hail.expr.types import *
from .misc import require_biallelic

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