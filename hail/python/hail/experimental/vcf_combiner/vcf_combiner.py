"""An experimental library for combining (g)VCFS into sparse matrix tables"""
# these are necessary for the diver script included at the end of this file
import math
import os
import uuid
from typing import Optional, List, Tuple, Dict

import hail as hl
from hail import MatrixTable, Table
from hail.experimental.function import Function
from hail.expr import StructExpression
from hail.expr.expressions import expr_bool, expr_str
from hail.genetics.reference_genome import reference_genome_type
from hail.ir import Apply, TableMapRows, MatrixKeyRowsBy
from hail.typecheck import oneof, sequenceof, typecheck
from hail.utils.java import info, warning, Env

_transform_rows_function_map: Dict[Tuple[hl.HailType], Function] = {}
_merge_function_map: Dict[Tuple[hl.HailType, hl.HailType], Function] = {}


@typecheck(string=expr_str, has_non_ref=expr_bool)
def parse_as_ints(string, has_non_ref):
    ints = string.split(r'\|')
    ints = hl.if_else(has_non_ref, ints[:-1], ints)
    return ints.map(lambda i: hl.if_else((hl.len(i) == 0) | (i == '.'), hl.missing(hl.tint32), hl.int32(i)))


@typecheck(string=expr_str, has_non_ref=expr_bool)
def parse_as_doubles(string, has_non_ref):
    ints = string.split(r'\|')
    ints = hl.if_else(has_non_ref, ints[:-1], ints)
    return ints.map(lambda i: hl.if_else((hl.len(i) == 0) | (i == '.'), hl.missing(hl.tfloat64), hl.float64(i)))


@typecheck(string=expr_str, has_non_ref=expr_bool)
def parse_as_sb_table(string, has_non_ref):
    ints = string.split(r'\|')
    ints = hl.if_else(has_non_ref, ints[:-1], ints)
    return ints.map(lambda xs: xs.split(",").map(hl.int32))


@typecheck(string=expr_str, has_non_ref=expr_bool)
def parse_as_ranksum(string, has_non_ref):
    typ = hl.ttuple(hl.tfloat64, hl.tint32)
    items = string.split(r'\|')
    items = hl.if_else(has_non_ref, items[:-1], items)
    return items.map(lambda s: hl.if_else(
        (hl.len(s) == 0) | (s == '.'),
        hl.missing(typ),
        hl.rbind(s.split(','), lambda ss: hl.if_else(
            hl.len(ss) != 2,  # bad field, possibly 'NaN', just set it null
            hl.missing(hl.ttuple(hl.tfloat64, hl.tint32)),
            hl.tuple([hl.float64(ss[0]), hl.int32(ss[1])])))))


_as_function_map = {
    'AS_QUALapprox': parse_as_ints,
    'AS_RAW_MQ': parse_as_doubles,
    'AS_RAW_MQRankSum': parse_as_ranksum,
    'AS_RAW_ReadPosRankSum': parse_as_ranksum,
    'AS_SB_TABLE': parse_as_sb_table,
    'AS_VarDP': parse_as_ints,
}


def parse_as_fields(info, has_non_ref):
    return hl.struct(**{f: info[f] if f not in _as_function_map
                        else _as_function_map[f](info[f], has_non_ref) for f in info})


def localize(mt):
    if isinstance(mt, MatrixTable):
        return mt._localize_entries('__entries', '__cols')
    return mt


def unlocalize(mt):
    if isinstance(mt, Table):
        return mt._unlocalize_entries('__entries', '__cols', ['s'])
    return mt


@typecheck(mt=oneof(Table, MatrixTable), info_to_keep=sequenceof(str))
def transform_gvcf(mt, info_to_keep=[]) -> Table:
    """Transforms a gvcf into a sparse matrix table

    The input to this should be some result of either :func:`.import_vcf` or
    :func:`.import_gvcfs` with ``array_elements_required=False``.

    There is an assumption that this function will be called on a matrix table
    with one column (or a localized table version of the same).

    Parameters
    ----------
    mt : :obj:`Union[Table, MatrixTable]`
        The gvcf being transformed, if it is a table, then it must be a localized matrix table with
        the entries array named ``__entries``
    info_to_keep : :obj:`List[str]`
        Any ``INFO`` fields in the gvcf that are to be kept and put in the ``gvcf_info`` entry
        field. By default, all ``INFO`` fields except ``END`` and ``DP`` are kept.

    Returns
    -------
    :obj:`.Table`
        A localized matrix table that can be used as part of the input to `combine_gvcfs`.

    Notes
    -----
    This function will parse the following allele specific annotations from
    pipe delimited strings into proper values. ::

        AS_QUALapprox
        AS_RAW_MQ
        AS_RAW_MQRankSum
        AS_RAW_ReadPosRankSum
        AS_SB_TABLE
        AS_VarDP

    """
    if not info_to_keep:
        info_to_keep = [name for name in mt.info if name not in ['END', 'DP']]
    mt = localize(mt)

    transform_row = _transform_rows_function_map.get(mt.row.dtype)
    if transform_row is None or not hl.current_backend()._is_registered_ir_function_name(transform_row._name):
        def get_lgt(e, n_alleles, has_non_ref, row):
            index = e.GT.unphased_diploid_gt_index()
            n_no_nonref = n_alleles - hl.int(has_non_ref)
            triangle_without_nonref = hl.triangle(n_no_nonref)
            return (hl.case()
                    .when(e.GT.is_haploid(),
                          hl.or_missing(e.GT[0] < n_no_nonref, e.GT))
                    .when(index < triangle_without_nonref, e.GT)
                    .when(index < hl.triangle(n_alleles), hl.missing('call'))
                    .or_error('invalid GT ' + hl.str(e.GT) + ' at site ' + hl.str(row.locus)))

        def make_entry_struct(e, alleles_len, has_non_ref, row):
            handled_fields = dict()
            handled_names = {'LA', 'gvcf_info',
                             'END',
                             'LAD', 'AD',
                             'LGT', 'GT',
                             'LPL', 'PL',
                             'LPGT', 'PGT'}

            if 'END' not in row.info:
                raise hl.utils.FatalError("the Hail GVCF combiner expects GVCFs to have an 'END' field in INFO.")
            if 'GT' not in e:
                raise hl.utils.FatalError("the Hail GVCF combiner expects GVCFs to have a 'GT' field in FORMAT.")

            handled_fields['LA'] = hl.range(0, alleles_len - hl.if_else(has_non_ref, 1, 0))
            handled_fields['LGT'] = get_lgt(e, alleles_len, has_non_ref, row)
            if 'AD' in e:
                handled_fields['LAD'] = hl.if_else(has_non_ref, e.AD[:-1], e.AD)
            if 'PGT' in e:
                handled_fields['LPGT'] = e.PGT
            if 'PL' in e:
                handled_fields['LPL'] = hl.if_else(has_non_ref,
                                                   hl.if_else(alleles_len > 2,
                                                              e.PL[:-alleles_len],
                                                              hl.missing(e.PL.dtype)),
                                                   hl.if_else(alleles_len > 1,
                                                              e.PL,
                                                              hl.missing(e.PL.dtype)))
                handled_fields['RGQ'] = hl.if_else(
                    has_non_ref,
                    hl.if_else(e.GT.is_haploid(),
                               e.PL[alleles_len - 1],
                               e.PL[hl.call(0, alleles_len - 1).unphased_diploid_gt_index()]),
                    hl.missing(e.PL.dtype.element_type))

            handled_fields['END'] = row.info.END
            handled_fields['gvcf_info'] = (hl.case()
                                           .when(hl.is_missing(row.info.END),
                                                 hl.struct(**(
                                                     parse_as_fields(
                                                         row.info.select(*info_to_keep),
                                                         has_non_ref)
                                                 )))
                                           .or_missing())

            pass_through_fields = {k: v for k, v in e.items() if k not in handled_names}
            return hl.struct(**handled_fields, **pass_through_fields)

        transform_row = hl.experimental.define_function(
            lambda row: hl.rbind(
                hl.len(row.alleles), '<NON_REF>' == row.alleles[-1],
                lambda alleles_len, has_non_ref: hl.struct(
                    locus=row.locus,
                    alleles=hl.if_else(has_non_ref, row.alleles[:-1], row.alleles),
                    rsid=row.rsid,
                    __entries=row.__entries.map(
                        lambda e: make_entry_struct(e, alleles_len, has_non_ref, row)))),
            mt.row.dtype)
        _transform_rows_function_map[mt.row.dtype] = transform_row
    return Table(TableMapRows(mt._tir, Apply(transform_row._name, transform_row._ret_type, mt.row._ir)))


def transform_one(mt, info_to_keep=[]) -> Table:
    return transform_gvcf(mt, info_to_keep)


def combine(ts):
    def merge_alleles(alleles):
        from hail.expr.functions import _num_allele_type, _allele_ints
        return hl.rbind(
            alleles.map(lambda a: hl.or_else(a[0], ''))
            .fold(lambda s, t: hl.if_else(hl.len(s) > hl.len(t), s, t), ''),
            lambda ref:
            hl.rbind(
                alleles.map(
                    lambda al: hl.rbind(
                        al[0],
                        lambda r:
                        hl.array([ref]).extend(
                            al[1:].map(
                                lambda a:
                                hl.rbind(
                                    _num_allele_type(r, a),
                                    lambda at:
                                    hl.if_else(
                                        (_allele_ints['SNP'] == at)
                                        | (_allele_ints['Insertion'] == at)
                                        | (_allele_ints['Deletion'] == at)
                                        | (_allele_ints['MNP'] == at)
                                        | (_allele_ints['Complex'] == at),
                                        a + ref[hl.len(r):],
                                        a)))))),
                lambda lal:
                hl.struct(
                    globl=hl.array([ref]).extend(hl.array(hl.set(hl.flatten(lal)).remove(ref))),
                    local=lal)))

    def renumber_entry(entry, old_to_new) -> StructExpression:
        # global index of alternate (non-ref) alleles
        return entry.annotate(LA=entry.LA.map(lambda lak: old_to_new[lak]))

    merge_function = _merge_function_map.get((ts.row.dtype, ts.globals.dtype))
    if merge_function is None or not hl.current_backend()._is_registered_ir_function_name(merge_function._name):
        merge_function = hl.experimental.define_function(
            lambda row, gbl:
            hl.rbind(
                merge_alleles(row.data.map(lambda d: d.alleles)),
                lambda alleles:
                hl.struct(
                    locus=row.locus,
                    alleles=alleles.globl,
                    rsid=hl.find(hl.is_defined, row.data.map(lambda d: d.rsid)),
                    __entries=hl.bind(
                        lambda combined_allele_index:
                        hl.range(0, hl.len(row.data)).flatmap(
                            lambda i:
                            hl.if_else(hl.is_missing(row.data[i].__entries),
                                       hl.range(0, hl.len(gbl.g[i].__cols))
                                       .map(lambda _: hl.missing(row.data[i].__entries.dtype.element_type)),
                                       hl.bind(
                                           lambda old_to_new: row.data[i].__entries.map(
                                               lambda e: renumber_entry(e, old_to_new)),
                                           hl.range(0, hl.len(alleles.local[i])).map(
                                               lambda j: combined_allele_index[alleles.local[i][j]])))),
                        hl.dict(hl.range(0, hl.len(alleles.globl)).map(
                            lambda j: hl.tuple([alleles.globl[j], j])))))),
            ts.row.dtype, ts.globals.dtype)
        _merge_function_map[(ts.row.dtype, ts.globals.dtype)] = merge_function
    ts = Table(TableMapRows(ts._tir, Apply(merge_function._name,
                                           merge_function._ret_type,
                                           ts.row._ir,
                                           ts.globals._ir)))
    return ts.transmute_globals(__cols=hl.flatten(ts.g.map(lambda g: g.__cols)))


@typecheck(mts=sequenceof(oneof(Table, MatrixTable)))
def combine_gvcfs(mts):
    """Merges gvcfs and/or sparse matrix tables

    Parameters
    ----------
    mts : :obj:`List[Union[Table, MatrixTable]]`
        The matrix tables (or localized versions) to combine

    Returns
    -------
    :class:`.MatrixTable`

    Notes
    -----
    All of the input tables/matrix tables must have the same partitioning. This
    module provides no method of repartitioning data.
    """
    ts = hl.Table.multi_way_zip_join([localize(mt) for mt in mts], 'data', 'g')
    combined = combine(ts)
    return unlocalize(combined)


@typecheck(mt=hl.MatrixTable, desired_average_partition_size=int, tmp_path=str)
def calculate_new_intervals(mt, desired_average_partition_size: int, tmp_path: str):
    """takes a table, keyed by ['locus', ...] and produces a list of intervals suitable
    for repartitioning a combiner matrix table.

    Parameters
    ----------
    mt : :class:`.MatrixTable`
        Sparse MT intermediate.
    desired_average_partition_size : :obj:`int`
        Average target number of rows for each partition.
    tmp_path : :obj:`str`
        Temporary path for scan checkpointing.

    Returns
    -------
    (:obj:`List[Interval]`, :obj:`.Type`)
    """
    assert list(mt.row_key) == ['locus']
    assert isinstance(mt.locus.dtype, hl.tlocus)
    reference_genome = mt.locus.dtype.reference_genome
    end = hl.Locus(reference_genome.contigs[-1],
                   reference_genome.lengths[reference_genome.contigs[-1]],
                   reference_genome=reference_genome)

    (n_rows, n_cols) = mt.count()

    if n_rows == 0:
        raise ValueError('empty table!')

    # split by a weight function that takes into account the number of
    # dense entries per row. However, give each row some base weight
    # to prevent densify computations from becoming unbalanced (these
    # scale roughly linearly with N_ROW * N_COL)
    ht = mt.select_rows(weight=hl.agg.count() + (n_cols // 25) + 1).rows().checkpoint(tmp_path)

    total_weight = ht.aggregate(hl.agg.sum(ht.weight))
    partition_weight = int(total_weight / (n_rows / desired_average_partition_size))

    ht = ht.annotate(cumulative_weight=hl.scan.sum(ht.weight),
                     last_weight=hl.scan._prev_nonnull(ht.weight),
                     row_idx=hl.scan.count())

    def partition_bound(x):
        return x - (x % hl.int64(partition_weight))

    at_partition_bound = partition_bound(ht.cumulative_weight) != partition_bound(ht.cumulative_weight - ht.last_weight)

    ht = ht.filter(at_partition_bound | (ht.row_idx == n_rows - 1))
    ht = ht.annotate(start=hl.or_else(
        hl.scan._prev_nonnull(hl.locus_from_global_position(ht.locus.global_position() + 1,
                                                            reference_genome=reference_genome)),
        hl.locus_from_global_position(0, reference_genome=reference_genome)))
    ht = ht.select(
        interval=hl.interval(start=hl.struct(locus=ht.start), end=hl.struct(locus=ht.locus), includes_end=True))

    intervals_dtype = hl.tarray(ht.interval.dtype)
    intervals = ht.aggregate(hl.agg.collect(ht.interval))
    last_st = hl.eval(
        hl.locus_from_global_position(hl.literal(intervals[-1].end.locus).global_position() + 1,
                                      reference_genome=reference_genome))
    interval = hl.Interval(start=hl.Struct(locus=last_st), end=hl.Struct(locus=end), includes_end=True)
    intervals.append(interval)
    return intervals, intervals_dtype


@typecheck(reference_genome=reference_genome_type, interval_size=int)
def calculate_even_genome_partitioning(reference_genome, interval_size) -> List[hl.utils.Interval]:
    """create a list of locus intervals suitable for importing and merging gvcfs.

    Parameters
    ----------
    reference_genome: :class:`str` or :class:`.ReferenceGenome`,
        Reference genome to use. NOTE: only GRCh37 and GRCh38 references
        are supported.
    interval_size: :obj:`int` The ceiling and rough target of interval size.
        Intervals will never be larger than this, but may be smaller.

    Returns
    -------
    :obj:`List[Interval]`
    """
    def calc_parts(contig):
        def locus_interval(start, end):
            return hl.Interval(
                start=hl.Locus(contig=contig, position=start, reference_genome=reference_genome),
                end=hl.Locus(contig=contig, position=end, reference_genome=reference_genome),
                includes_end=True)

        contig_length = reference_genome.lengths[contig]
        n_parts = math.ceil(contig_length / interval_size)
        real_size = math.ceil(contig_length / n_parts)
        n = 1
        intervals = []
        while n < contig_length:
            start = n
            end = min(n + real_size, contig_length)
            intervals.append(locus_interval(start, end))
            n = end + 1

        return intervals

    if reference_genome.name == 'GRCh37':
        contigs = [f'{i}' for i in range(1, 23)] + ['X', 'Y', 'MT']
    elif reference_genome.name == 'GRCh38':
        contigs = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY', 'chrM']
    else:
        raise ValueError(
            f"Unsupported reference genome '{reference_genome.name}', "
            "only 'GRCh37' and 'GRCh38' are supported")

    intervals = []
    for ctg in contigs:
        intervals.extend(calc_parts(ctg))
    return intervals


# END OF VCF COMBINER LIBRARY, BEGINNING OF BEST PRACTICES SCRIPT #


class Merge(object):
    def __init__(self,
                 inputs: List[int],
                 input_total_size: int):
        self.inputs: List[int] = inputs
        self.input_total_size: int = input_total_size


class Job(object):
    def __init__(self, merges: List[Merge]):
        self.merges: List[Merge] = merges
        self.input_total_size = sum(m.input_total_size for m in merges)


class Phase(object):
    def __init__(self, jobs: List[Job]):
        self.jobs: List[Job] = jobs


class CombinerPlan(object):
    def __init__(self,
                 file_size: List[List[int]],
                 phases: List[Phase]):
        self.file_size = file_size
        self.phases = phases
        self.merge_per_phase = len(file_size[0])
        self.total_merge = self.merge_per_phase * len(phases)


class CombinerConfig(object):
    default_max_partitions_per_job = 75_000
    default_branch_factor = 100
    default_phase1_batch_size = 100
    default_target_records = 30_000

    # These are used to calculate intervals for reading GVCFs in the combiner
    # The genome interval size results in 2568 partions for GRCh38. The exome
    # interval size assumes that they are around 2% the size of a genome and
    # result in 65 partitions for GRCh38.
    default_genome_interval_size = 1_200_000
    default_exome_interval_size = 60_000_000

    def __init__(self,
                 branch_factor: int = default_branch_factor,
                 batch_size: int = default_phase1_batch_size,
                 target_records: int = default_target_records):
        self.branch_factor: int = branch_factor
        self.batch_size: int = batch_size
        self.target_records: int = target_records

    @classmethod
    def default(cls) -> 'CombinerConfig':
        return CombinerConfig()

    def plan(self, n_inputs: int) -> CombinerPlan:
        assert n_inputs > 0

        def int_ceil(x):
            return int(math.ceil(x))

        tree_height = int_ceil(math.log(n_inputs, self.branch_factor))
        phases: List[Phase] = []
        file_size: List[List[int]] = []  # List of file size per phase

        file_size.append([1 for _ in range(n_inputs)])
        while len(file_size[-1]) > 1:
            batch_size_this_phase = self.batch_size if len(file_size) == 1 else 1
            last_stage_files = file_size[-1]
            n = len(last_stage_files)
            i = 0
            jobs = []
            while (i < n):
                job = []
                job_i = 0
                while job_i < batch_size_this_phase and i < n:
                    merge = []
                    merge_i = 0
                    merge_size = 0
                    while merge_i < self.branch_factor and i < n:
                        merge_size += last_stage_files[i]
                        merge.append(i)
                        merge_i += 1
                        i += 1
                    job.append(Merge(merge, merge_size))
                    job_i += 1
                jobs.append(Job(job))
            file_size.append([merge.input_total_size for job in jobs for merge in job.merges])
            phases.append(Phase(jobs))

        assert len(phases) == tree_height
        for layer in file_size:
            assert sum(layer) == n_inputs

        phase_strs = []
        total_jobs = 0
        for i, phase in enumerate(phases):
            n = len(phase.jobs)
            job_str = hl.utils.misc.plural('job', n)
            n_files_produced = len(file_size[i + 1])
            adjective = 'final' if n_files_produced == 1 else 'intermediate'
            file_str = hl.utils.misc.plural('file', n_files_produced)
            phase_strs.append(
                f'\n        Phase {i + 1}: {n} {job_str} corresponding to {n_files_produced} {adjective} output {file_str}.')
            total_jobs += n

        info(f"GVCF combiner plan:\n"
             f"    Branch factor: {self.branch_factor}\n"
             f"    Phase 1 batch size: {self.batch_size}\n"
             f"    Combining {n_inputs} input files in {tree_height} phases with {total_jobs} total jobs.{''.join(phase_strs)}\n")
        return CombinerPlan(file_size, phases)


def run_combiner(sample_paths: List[str],
                 out_file: str,
                 tmp_path: str,
                 *,
                 intervals: Optional[List[hl.utils.Interval]] = None,
                 import_interval_size: Optional[int] = None,
                 use_genome_default_intervals: bool = False,
                 use_exome_default_intervals: bool = False,
                 header: Optional[str] = None,
                 sample_names: Optional[List[str]] = None,
                 branch_factor: int = CombinerConfig.default_branch_factor,
                 batch_size: int = CombinerConfig.default_phase1_batch_size,
                 target_records: int = CombinerConfig.default_target_records,
                 overwrite: bool = False,
                 reference_genome: str = 'default',
                 contig_recoding: Optional[Dict[str, str]] = None,
                 key_by_locus_and_alleles: bool = False):
    """Run the Hail VCF combiner, performing a hierarchical merge to create a combined sparse matrix table.

    **Partitioning**

    The partitioning of input GVCFs, which determines the maximum parallelism per file,
    is determined the four parameters below. One of these parameters must be passed to
    this function.

    - `intervals` -- User-supplied intervals.
    - `import_interval_size` -- Use intervals of this uniform size across the genome.
    - `use_genome_default_intervals` -- Use intervals of typical uniform size for whole
      genome GVCFs.
    - `use_exome_default_intervals` -- Use intervals of typical uniform size for exome
      GVCFs.

    It is recommended that new users include either `use_genome_default_intervals` or
    `use_exome_default_intervals`.

    Note also that the partitioning of the final, combined matrix table does not depend
    the GVCF input partitioning.

    Parameters
    ----------
    sample_paths : :obj:`list` of :class:`str`
        Paths to individual GVCFs.
    out_file : :class:`str`
        Path to final combined matrix table.
    tmp_path : :class:`str`
        Path for intermediate output.
    intervals : list of :class:`.Interval` or None
        Import GVCFs with specified partition intervals.
    import_interval_size : :obj:`int` or None
        Import GVCFs with uniform partition intervals of specified size.
    use_genome_default_intervals : :obj:`bool`
        Import GVCFs with uniform partition intervals of default size for
        whole-genome data.
    use_exome_default_intervals : :obj:`bool`
        Import GVCFs with uniform partition intervals of default size for
        exome data.
    header : :class:`str` or None
        External header file to use as GVCF header for all inputs. If defined, `sample_names` must be defined as well.
    sample_names: list of :class:`str` or None
        Sample names, to be used with `header`.
    branch_factor : :obj:`int`
        Combiner branch factor.
    batch_size : :obj:`int`
        Combiner batch size.
    target_records : :obj:`int`
        Target records per partition in each combiner phase after the first.
    overwrite : :obj:`bool`
        Overwrite output file, if it exists.
    reference_genome : :class:`str`
        Reference genome for GVCF import.
    contig_recoding: :obj:`dict` of (:class:`str`, :obj:`str`), optional
        Mapping from contig name in gVCFs to contig name the reference
        genome.  All contigs must be present in the
        `reference_genome`, so this is useful for mapping
        differently-formatted data onto known references.
    key_by_locus_and_alleles : :obj:`bool`
        Key by both locus and alleles in the final output.

    Returns
    -------
    None

    """
    hl.utils.no_service_backend('vcf_combiner')
    flagname = 'no_ir_logging'
    prev_flag_value = hl._get_flags(flagname).get(flagname)
    hl._set_flags(**{flagname: '1'})
    tmp_path += f'/combiner-temporary/{uuid.uuid4()}/'
    if header is not None:
        assert sample_names is not None
        assert len(sample_names) == len(sample_paths)

    n_partition_args = (int(intervals is not None)
                        + int(import_interval_size is not None)
                        + int(use_genome_default_intervals)
                        + int(use_exome_default_intervals))

    if n_partition_args == 0:
        raise ValueError("'run_combiner': require one argument from 'intervals', 'import_interval_size', "
                         "'use_genome_default_intervals', or 'use_exome_default_intervals' to choose GVCF partitioning")
    if n_partition_args > 1:
        warning("'run_combiner': multiple colliding arguments found from 'intervals', 'import_interval_size', "
                "'use_genome_default_intervals', or 'use_exome_default_intervals'."
                "\n  The argument found first in the list in this warning will be used, and others ignored.")

    if intervals is not None:
        info(f"Using {len(intervals)} user-supplied intervals as partitioning for GVCF import")
    elif import_interval_size is not None:
        intervals = calculate_even_genome_partitioning(reference_genome, import_interval_size)
        info(f"Using {len(intervals)} intervals with user-supplied size"
             f" {import_interval_size} as partitioning for GVCF import")
    elif use_genome_default_intervals:
        size = CombinerConfig.default_genome_interval_size
        intervals = calculate_even_genome_partitioning(reference_genome, size)
        info(f"Using {len(intervals)} intervals with default whole-genome size"
             f" {size} as partitioning for GVCF import")
    elif use_exome_default_intervals:
        size = CombinerConfig.default_exome_interval_size
        intervals = calculate_even_genome_partitioning(reference_genome, size)
        info(f"Using {len(intervals)} intervals with default exome size"
             f" {size} as partitioning for GVCF import")

    assert intervals is not None

    config = CombinerConfig(branch_factor=branch_factor,
                            batch_size=batch_size,
                            target_records=target_records)
    plan = config.plan(len(sample_paths))

    files_to_merge = sample_paths
    n_phases = len(plan.phases)
    total_ops = len(files_to_merge) * n_phases
    total_work_done = 0
    for phase_i, phase in enumerate(plan.phases):
        phase_i += 1  # used for info messages, 1-indexed for readability

        n_jobs = len(phase.jobs)
        merge_str = 'input GVCFs' if phase_i == 1 else 'intermediate sparse matrix tables'
        job_str = hl.utils.misc.plural('job', n_jobs)
        info(f"Starting phase {phase_i}/{n_phases}, merging {len(files_to_merge)} {merge_str} in {n_jobs} {job_str}.")

        if phase_i > 1:
            intervals, intervals_dtype = calculate_new_intervals(hl.read_matrix_table(files_to_merge[0]),
                                                                 config.target_records,
                                                                 os.path.join(tmp_path,
                                                                              f'phase{phase_i}_interval_checkpoint.ht'))

        new_files_to_merge = []

        for job_i, job in enumerate(phase.jobs):
            job_i += 1  # used for info messages, 1-indexed for readability

            n_merges = len(job.merges)
            merge_str = hl.utils.misc.plural('file', n_merges)
            pct_total = 100 * job.input_total_size / total_ops
            info(
                f"Starting phase {phase_i}/{n_phases}, job {job_i}/{len(phase.jobs)} to create {n_merges} merged {merge_str}, corresponding to ~{pct_total:.1f}% of total I/O.")
            merge_mts: List[MatrixTable] = []
            for merge in job.merges:
                inputs = [files_to_merge[i] for i in merge.inputs]

                if phase_i == 1:
                    mts = [transform_gvcf(vcf)
                           for vcf in hl.import_gvcfs(inputs, intervals, array_elements_required=False,
                                                      _external_header=header,
                                                      _external_sample_ids=[[sample_names[i]] for i in
                                                                            merge.inputs] if header is not None else None,
                                                      reference_genome=reference_genome,
                                                      contig_recoding=contig_recoding)]
                else:
                    mts = Env.spark_backend("vcf_combiner").read_multiple_matrix_tables(inputs, intervals,
                                                                                        intervals_dtype)

                merge_mts.append(combine_gvcfs(mts))

            if phase_i == n_phases:  # final merge!
                assert n_jobs == 1
                assert len(merge_mts) == 1
                [final_mt] = merge_mts

                if key_by_locus_and_alleles:
                    final_mt = MatrixTable(MatrixKeyRowsBy(final_mt._mir, ['locus', 'alleles'], is_sorted=True))
                final_mt.write(out_file, overwrite=overwrite)
                new_files_to_merge = [out_file]
                info(f"Finished phase {phase_i}/{n_phases}, job {job_i}/{len(phase.jobs)}, 100% of total I/O finished.")
                break

            tmp = f'{tmp_path}_phase{phase_i}_job{job_i}/'
            hl.experimental.write_matrix_tables(merge_mts, tmp, overwrite=True)
            pad = len(str(len(merge_mts) - 1))
            new_files_to_merge.extend(tmp + str(n).zfill(pad) + '.mt' for n in range(len(merge_mts)))
            total_work_done += job.input_total_size
            info(
                f"Finished {phase_i}/{n_phases}, job {job_i}/{len(phase.jobs)}, {100 * total_work_done / total_ops:.1f}% of total I/O finished.")

        info(f"Finished phase {phase_i}/{n_phases}.")

        files_to_merge = new_files_to_merge

    assert files_to_merge == [out_file]

    info("Finished!")
    hl._set_flags(**{flagname: prev_flag_value})


def parse_sample_mapping(sample_map_path: str) -> Tuple[List[str], List[str]]:
    sample_names: List[str] = list()
    sample_paths: List[str] = list()

    with hl.hadoop_open(sample_map_path) as f:
        for line in f:
            [name, path] = line.strip().split('\t')
            sample_names.append(name)
            sample_paths.append(path)

    return sample_names, sample_paths
