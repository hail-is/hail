"""An experimental library for combining (g)VCFS into sparse matrix tables"""
# these are necessary for the diver script included at the end of this file
import argparse
import time
import uuid

import hail as hl
from hail import MatrixTable, Table
from hail.expr import StructExpression
from hail.expr.expressions import expr_bool, expr_str
from hail.genetics.reference_genome import reference_genome_type
from hail.ir import Apply, TableMapRows, TopLevelReference
from hail.typecheck import oneof, sequenceof, typecheck

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
def calculate_new_intervals(ht, n, reference_genome='default'):
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

    ht = ht.select()
    ht = ht.annotate(x=hl.scan.count())
    ht = ht.annotate(y=ht.x + 1)
    ht = ht.filter(ht.x // n != ht.y // n)
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
def default_exome_intervals(reference_genome='default'):
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
        raise ValueError(f"Invalid reference genome '{reference_genome.name}', only 'GRCh37' and 'GRCh38' are supported")
    return [hl.Interval(start=hl.Locus(contig=contig, position=1, reference_genome=reference_genome),
                        end=hl.Locus.parse(f'{contig}:END', reference_genome=reference_genome),
                        includes_end=True) for contig in contigs]


# END OF VCF COMBINER LIBRARY, BEGINNING OF BEST PRACTICES SCRIPT #

DEFAULT_REF = 'GRCh38'
MAX_MULTI_WRITE_NUMBER = 100
MAX_COMBINE_NUMBER = 100
# The target number of rows per partition during each round of merging
TARGET_RECORDS = 30_000

def chunks(seq, size):
    """iterate through a list size elements at a time"""
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def stage_one(paths, sample_names, tmp_path, intervals, header, out_path):
    """stage one of the combiner, responsible for importing gvcfs, transforming them
       into what the combiner expects, and writing intermediates."""
    def h(paths, sample_names, tmp_path, intervals, header, out_path, i, first):
        vcfs = [transform_gvcf(vcf)
                for vcf in hl.import_gvcfs(paths, intervals, array_elements_required=False,
                                           _external_header=header,
                                           _external_sample_ids=sample_names if header is not None else None)]
        combined = [combine_gvcfs(mts) for mts in chunks(vcfs, MAX_COMBINE_NUMBER)]
        if first and len(paths) <= MAX_COMBINE_NUMBER:  # only 1 item, just write it, unless we have already written other items
            combined[0].write(out_path, overwrite=True)
            return []
        pad = len(str(len(combined)))
        hl.experimental.write_matrix_tables(combined, tmp_path + f'{i}/', overwrite=True)
        return [tmp_path + f'{i}/' + str(n).zfill(pad) + '.mt' for n in range(len(combined))]

    assert len(paths) == len(sample_names)
    tmp_path += f'{uuid.uuid4()}/'
    out_paths = []
    i = 0
    size = MAX_MULTI_WRITE_NUMBER * MAX_COMBINE_NUMBER
    first = True
    for pos in range(0, len(paths), size):
        tmp = h(paths[pos:pos + size], sample_names[pos:pos + size], tmp_path, intervals, header,
                out_path, i, first)
        if not tmp:
            return tmp
        out_paths.extend(tmp)
        first = False
        i += 1
    return out_paths

def run_combiner(sample_names, sample_paths, intervals, out_file, tmp_path, header, overwrite):
    tmp_path += f'/combiner-temporary/{uuid.uuid4()}/'
    assert len(sample_names) == len(sample_paths)
    out_paths = stage_one(sample_paths, sample_names, tmp_path, intervals, header, out_file)
    if not out_paths:
        return
    tmp_path += f'{uuid.uuid4()}/'

    ht = hl.read_matrix_table(out_paths[0]).rows()
    intervals = calculate_new_intervals(ht, TARGET_RECORDS)

    mts = [hl.read_matrix_table(path, _intervals=intervals) for path in out_paths]
    combined_mts = [combine_gvcfs(mt) for mt in chunks(mts, MAX_COMBINE_NUMBER)]
    i = 0
    while len(combined_mts) > 1:
        tmp = tmp_path + f'{i}/'
        pad = len(str(len(combined_mts)))
        hl.experimental.write_matrix_tables(combined_mts, tmp, overwrite=True)
        paths = [tmp + str(n).zfill(pad) + '.mt' for n in range(len(combined_mts))]

        ht = hl.read_matrix_table(out_paths[0]).rows()
        intervals = calculate_new_intervals(ht, TARGET_RECORDS)

        mts = [hl.read_matrix_table(path, _intervals=intervals) for path in paths]
        combined_mts = [combine_gvcfs(mts) for mt in chunks(mts, MAX_COMBINE_NUMBER)]
        i += 1
    combined_mts[0].write(out_file, overwrite=overwrite)

def drive_combiner(sample_map_path, intervals, out_file, tmp_path, header, overwrite=False):
    with open(sample_map_path) as sample_map:
        samples = [l.strip().split('\t') for l in sample_map]
    sample_names, sample_paths = [list(x) for x in zip(*samples)]
    sample_names = [[n] for n in sample_names]
    run_combiner(sample_names, sample_paths, intervals, out_file, tmp_path, header, overwrite)

def main():
    parser = argparse.ArgumentParser(description="Driver for hail's GVCF combiner")
    parser.add_argument('sample_map',
                        help='path to the sample map (must be readable by this script). '
                             'The sample map should be tab separated with two columns. '
                             'The first column is the sample ID, and the second column '
                             'is the GVCF path.')
    parser.add_argument('out_file', help='path to final combiner output')
    parser.add_argument('--tmp-path', help='path to folder for intermediate output (can be a cloud bucket)',
                        default='/tmp')
    parser.add_argument('--log', help='path to hail log file',
                        default='/hail-joint-caller-' + time.strftime('%Y%m%d-%H%M') + '.log')
    parser.add_argument('--header',
                        help='external header, must be readable by all executors. '
                             'WARNING: if this option is used, the sample names in the '
                             'gvcfs will be overridden by the names in sample map.',
                        required=False)
    parser.add_argument('--overwrite', help='overwrite the output path', action='store_true')
    args = parser.parse_args()
    hl.init(default_reference=DEFAULT_REF,
            log=args.log)

    # NOTE: This will need to be changed to support genomes as well
    intervals = default_exome_intervals()
    with open(args.sample_map) as sample_map:
        samples = [l.strip().split('\t') for l in sample_map]
    if not args.overwrite and hl.utils.hadoop_exists(args.out_file):
        raise FileExistsError(f"path '{args.out_file}' already exists, use --overwrite to overwrite this path")

    drive_combiner(samples, intervals, args.out_file, args.tmp_path, args.header, args.overwrite)


if __name__ == '__main__':
    main()
