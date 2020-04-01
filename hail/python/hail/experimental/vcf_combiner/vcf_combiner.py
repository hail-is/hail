"""An experimental library for combining (g)VCFS into sparse matrix tables"""
# these are necessary for the diver script included at the end of this file
import math
import uuid
from typing import Optional, List, Tuple

import hail as hl
from hail import MatrixTable, Table
from hail.expr import StructExpression
from hail.expr.expressions import expr_bool, expr_str
from hail.genetics.reference_genome import reference_genome_type
from hail.ir import Apply, TableMapRows, TopLevelReference
from hail.typecheck import oneof, sequenceof, typecheck
from hail.utils.java import info

_transform_rows_function_map = {}
_merge_function_map = {}


@typecheck(string=expr_str, has_non_ref=expr_bool)
def parse_as_ints(string, has_non_ref):
    ints = string.split(r'\|')
    ints = hl.cond(has_non_ref, ints[:-1], ints)
    return ints.map(lambda i: hl.cond((hl.len(i) == 0) | (i == '.'), hl.null(hl.tint32), hl.int32(i)))


@typecheck(string=expr_str, has_non_ref=expr_bool)
def parse_as_doubles(string, has_non_ref):
    ints = string.split(r'\|')
    ints = hl.cond(has_non_ref, ints[:-1], ints)
    return ints.map(lambda i: hl.cond((hl.len(i) == 0) | (i == '.'), hl.null(hl.tfloat64), hl.float64(i)))


@typecheck(string=expr_str, has_non_ref=expr_bool)
def parse_as_sb_table(string, has_non_ref):
    ints = string.split(r'\|')
    ints = hl.cond(has_non_ref, ints[:-1], ints)
    return ints.map(lambda xs: xs.split(",").map(hl.int32))


@typecheck(string=expr_str, has_non_ref=expr_bool)
def parse_as_ranksum(string, has_non_ref):
    typ = hl.ttuple(hl.tfloat64, hl.tint32)
    items = string.split(r'\|')
    items = hl.cond(has_non_ref, items[:-1], items)
    return items.map(lambda s: hl.cond(
        (hl.len(s) == 0) | (s == '.'),
        hl.null(typ),
        hl.rbind(s.split(','), lambda ss: hl.cond(
            hl.len(ss) != 2,  # bad field, possibly 'NaN', just set it null
            hl.null(hl.ttuple(hl.tfloat64, hl.tint32)),
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
        A localized matrix table that can be used as part of the input to :func:`.combine_gvcfs`

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

    if mt.row.dtype not in _transform_rows_function_map:
        f = hl.experimental.define_function(
            lambda row: hl.rbind(
                hl.len(row.alleles), '<NON_REF>' == row.alleles[-1],
                lambda alleles_len, has_non_ref: hl.struct(
                    locus=row.locus,
                    alleles=hl.cond(has_non_ref, row.alleles[:-1], row.alleles),
                    rsid=row.rsid,
                    __entries=row.__entries.map(
                        lambda e:
                        hl.struct(
                            DP=e.DP,
                            END=row.info.END,
                            GQ=e.GQ,
                            LA=hl.range(0, alleles_len - hl.cond(has_non_ref, 1, 0)),
                            LAD=hl.cond(has_non_ref, e.AD[:-1], e.AD),
                            LGT=e.GT,
                            LPGT=e.PGT,
                            LPL=hl.cond(has_non_ref,
                                        hl.cond(alleles_len > 2,
                                                e.PL[:-alleles_len],
                                                hl.null(e.PL.dtype)),
                                        hl.cond(alleles_len > 1,
                                                e.PL,
                                                hl.null(e.PL.dtype))),
                            MIN_DP=e.MIN_DP,
                            PID=e.PID,
                            RGQ=hl.cond(
                                has_non_ref,
                                e.PL[hl.call(0, alleles_len - 1).unphased_diploid_gt_index()],
                                hl.null(e.PL.dtype.element_type)),
                            SB=e.SB,
                            gvcf_info=hl.case()
                                .when(hl.is_missing(row.info.END),
                                      hl.struct(**(
                                          parse_as_fields(
                                              row.info.select(*info_to_keep),
                                              has_non_ref)
                                      )))
                                .or_missing()
                        ))),
            ),
            mt.row.dtype)
        _transform_rows_function_map[mt.row.dtype] = f
    transform_row = _transform_rows_function_map[mt.row.dtype]
    return Table(TableMapRows(mt._tir, Apply(transform_row._name, transform_row._ret_type, TopLevelReference('row'))))


def transform_one(mt, info_to_keep=[]) -> Table:
    return transform_gvcf(mt, info_to_keep)


def combine(ts):
    def merge_alleles(alleles):
        from hail.expr.functions import _num_allele_type, _allele_ints
        return hl.rbind(
            alleles.map(lambda a: hl.or_else(a[0], ''))
                .fold(lambda s, t: hl.cond(hl.len(s) > hl.len(t), s, t), ''),
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
                                    hl.cond(
                                        (_allele_ints['SNP'] == at) |
                                        (_allele_ints['Insertion'] == at) |
                                        (_allele_ints['Deletion'] == at) |
                                        (_allele_ints['MNP'] == at) |
                                        (_allele_ints['Complex'] == at),
                                        a + ref[hl.len(r):],
                                        a)))))),
                lambda lal:
                hl.struct(
                    globl=hl.array([ref]).extend(hl.array(hl.set(hl.flatten(lal)).remove(ref))),
                    local=lal)))

    def renumber_entry(entry, old_to_new) -> StructExpression:
        # global index of alternate (non-ref) alleles
        return entry.annotate(LA=entry.LA.map(lambda lak: old_to_new[lak]))

    if (ts.row.dtype, ts.globals.dtype) not in _merge_function_map:
        f = hl.experimental.define_function(
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
                            hl.cond(hl.is_missing(row.data[i].__entries),
                                    hl.range(0, hl.len(gbl.g[i].__cols))
                                    .map(lambda _: hl.null(row.data[i].__entries.dtype.element_type)),
                                    hl.bind(
                                        lambda old_to_new: row.data[i].__entries.map(
                                            lambda e: renumber_entry(e, old_to_new)),
                                        hl.range(0, hl.len(alleles.local[i])).map(
                                            lambda j: combined_allele_index[alleles.local[i][j]])))),
                        hl.dict(hl.range(0, hl.len(alleles.globl)).map(
                            lambda j: hl.tuple([alleles.globl[j], j])))))),
            ts.row.dtype, ts.globals.dtype)
        _merge_function_map[(ts.row.dtype, ts.globals.dtype)] = f
    merge_function = _merge_function_map[(ts.row.dtype, ts.globals.dtype)]
    ts = Table(TableMapRows(ts._tir, Apply(merge_function._name,
                                           merge_function._ret_type,
                                           TopLevelReference('row'),
                                           TopLevelReference('global'))))
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


@typecheck(ht=hl.Table, n=int, reference_genome=reference_genome_type)
def calculate_new_intervals(ht, n, reference_genome):
    """takes a table, keyed by ['locus', ...] and produces a list of intervals suitable
    for repartitioning a combiner matrix table

    Parameters
    ----------
    ht : :class:`.Table`
        Table / Rows Table to compute new intervals for
    n : :obj:`int`
        Number of rows each partition should have, (last partition may be smaller)
    reference_genome: :obj:`str` or :class:`.ReferenceGenome`, optional
        Reference genome to use.

    Returns
    -------
    :obj:`List[Interval]`
    """
    assert list(ht.key) == ['locus']
    assert ht.locus.dtype == hl.tlocus(reference_genome=reference_genome)
    end = hl.Locus(reference_genome.contigs[-1],
                   reference_genome.lengths[reference_genome.contigs[-1]],
                   reference_genome=reference_genome)

    n_rows = ht.count()

    if n_rows == 0:
        raise ValueError('empty table!')

    ht = ht.select()
    ht = ht.annotate(x=hl.scan.count())
    ht = ht.annotate(y=ht.x + 1)
    ht = ht.filter((ht.x // n != ht.y // n) | (ht.x == (n_rows - 1)))
    ht = ht.select()
    ht = ht.annotate(start=hl.or_else(
        hl.scan._prev_nonnull(hl.locus_from_global_position(ht.locus.global_position() + 1,
                                                            reference_genome=reference_genome)),
        hl.locus_from_global_position(0, reference_genome=reference_genome)))
    ht = ht.key_by()
    ht = ht.select(interval=hl.interval(start=ht.start, end=ht.locus, includes_end=True))

    intervals = ht.aggregate(hl.agg.collect(ht.interval))

    last_st = hl.eval(
        hl.locus_from_global_position(hl.literal(intervals[-1].end).global_position() + 1,
                                      reference_genome=reference_genome))
    interval = hl.Interval(start=last_st, end=end, includes_end=True)
    intervals.append(interval)
    return intervals


@typecheck(reference_genome=reference_genome_type)
def default_exome_intervals(reference_genome) -> List[hl.utils.Interval]:
    """create a list of locus intervals suitable for importing and merging exome gvcfs. As exomes
    are small. One partition per chromosome works well here.

    Parameters
    ----------
    reference_genome: :obj:`str` or :class:`.ReferenceGenome`, optional
        Reference genome to use. NOTE: only GRCh37 and GRCh38 references
        are supported.

    Returns
    -------
    :obj:`List[Interval]`
    """
    if reference_genome.name == 'GRCh37':
        contigs = [f'{i}' for i in range(1, 23)] + ['X', 'Y', 'MT']
    elif reference_genome.name == 'GRCh38':
        contigs = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY', 'chrM']
    else:
        raise ValueError(
            f"Invalid reference genome '{reference_genome.name}', only 'GRCh37' and 'GRCh38' are supported")
    return [hl.Interval(start=hl.Locus(contig=contig, position=1, reference_genome=reference_genome),
                        end=hl.Locus.parse(f'{contig}:END', reference_genome=reference_genome),
                        includes_end=True) for contig in contigs]


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
    default_branch_factor = 100
    default_batch_size = 100
    default_target_records = 30_000

    def __init__(self,
                 branch_factor: int = default_branch_factor,
                 batch_size: int = default_batch_size,
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
            last_stage_files = file_size[-1]
            n = len(last_stage_files)
            i = 0
            jobs = []
            while (i < n):
                job = []
                job_i = 0
                while job_i < self.batch_size and i < n:
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
             f"    Batch size: {self.batch_size}\n"
             f"    Combining {n_inputs} input files in {tree_height} phases with {total_jobs} total jobs.{''.join(phase_strs)}\n")
        return CombinerPlan(file_size, phases)


def run_combiner(sample_paths: List[str],
                 out_file: str,
                 tmp_path: str,
                 intervals: Optional[List[hl.utils.Interval]] = None,
                 header: Optional[str] = None,
                 sample_names: Optional[List[str]] = None,
                 branch_factor: int = CombinerConfig.default_branch_factor,
                 batch_size: int = CombinerConfig.default_batch_size,
                 target_records: int = CombinerConfig.default_target_records,
                 overwrite: bool = False,
                 reference_genome: str = 'default'):
    """Run the Hail VCF combiner, performing a hierarchical merge to create a combined sparse matrix table.

    Parameters
    ----------
    sample_paths : :obj:`list` of :obj:`str`
        Paths to individual GVCFs.
    out_file : :obj:`str`
        Path to final combined matrix table.
    tmp_path : :obj:`str`
        Path for intermediate output.
    intervals : list of :class:`.Interval` or None
        Partitioning with which to import GVCFs in first phase of combiner.
    header : :obj:`str` or None
        External header file to use as GVCF header for all inputs. If defined, `sample_names` must be defined as well.
    sample_names: list of :obj:`str` or None
        Sample names, to be used with `header`.
    branch_factor : :obj:`int`
        Combiner branch factor.
    batch_size : :obj:`int`
        Combiner batch size.
    target_records : :obj:`int`
        Target records per partition in each combiner phase after the first.
    overwrite : :obj:`bool`
        Overwrite output file, if it exists.
    reference_genome : :obj:`str`
        Reference genome for GVCF import.

    Returns
    -------
    None
    """
    tmp_path += f'/combiner-temporary/{uuid.uuid4()}/'
    if header is not None:
        assert sample_names is not None
        assert len(sample_names) == len(sample_paths)

    # FIXME: this should be hl.default_reference().even_intervals_contig_boundary
    intervals = intervals or default_exome_intervals(reference_genome)

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
            intervals = calculate_new_intervals(hl.read_matrix_table(files_to_merge[0]).rows(),
                                                config.target_records,
                                                reference_genome=reference_genome)

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
                                                      _external_sample_ids=[sample_names[i] for i in
                                                                            merge.inputs] if header is not None else None,
                                                      reference_genome=reference_genome)]
                else:
                    mts = [hl.read_matrix_table(path, _intervals=intervals) for path in inputs]

                merge_mts.append(combine_gvcfs(mts))

            if phase_i == n_phases:  # final merge!
                assert n_jobs == 1
                assert len(merge_mts) == 1
                [final_mt] = merge_mts
                final_mt.write(out_file, overwrite=overwrite)
                new_files_to_merge = [out_file]
                info(f"Finished phase {phase_i}/{n_phases}, job {job_i}/{len(phase.jobs)}, 100% of total I/O finished.")
                break

            tmp = f'{tmp_path}_phase{phase_i}_job{job_i}/'
            hl.experimental.write_matrix_tables(merge_mts, tmp, overwrite=True)
            pad = len(str(len(merge_mts)))
            new_files_to_merge.extend(tmp + str(n).zfill(pad) + '.mt' for n in range(len(merge_mts)))
            total_work_done += job.input_total_size
            info(
                f"Finished {phase_i}/{n_phases}, job {job_i}/{len(phase.jobs)}, {100 * total_work_done / total_ops:.1f}% of total I/O finished.")

        info(f"Finished phase {phase_i}/{n_phases}.")

        files_to_merge = new_files_to_merge

    assert files_to_merge == [out_file]

    info("Finished!")


def parse_sample_mapping(sample_map_path: str) -> Tuple[List[str], List[str]]:
    sample_names: List[str] = list()
    sample_paths: List[str] = list()

    with hl.hadoop_open(sample_map_path) as f:
        for line in f:
            [name, path] = line.strip().split('\t')
            sample_names.append(name)
            sample_paths.append(path)

    return sample_names, sample_paths
