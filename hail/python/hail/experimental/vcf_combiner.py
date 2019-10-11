"""A work in progress pipeline to combine (g)VCFs into an alternate format"""

import hail as hl
from hail import MatrixTable, Table
from hail.expr import StructExpression
from hail.expr.expressions import expr_call, expr_array, expr_int32
from hail.genetics.reference_genome import reference_genome_type
from hail.ir import Apply, TableMapRows, TopLevelReference
from hail.typecheck import oneof, sequenceof, typecheck

_transform_rows_function_map = {}
_merge_function_map = {}

def localize(mt):
    if isinstance(mt, MatrixTable):
        return mt._localize_entries('__entries', '__cols')
    return mt

def unlocalize(mt):
    if isinstance(mt, Table):
        return mt._unlocalize_entries('__entries', '__cols', ['s'])
    return mt

@typecheck(mt=oneof(Table, MatrixTable), info_to_keep=sequenceof(str))
def transform_one(mt, info_to_keep=[]) -> Table:
    """transforms a gvcf into a form suitable for combining

    The input to this should be some result of either :func:`.import_vcf` or
    :func:`.import_vcfs` with `array_elements_required=False`.

    There is a strong assumption that this function will be called on a matrix
    table with one column.
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
                                      hl.struct(**(row.info.select(*info_to_keep))))
                                .or_missing()
                        ))),
            ),
            mt.row.dtype)
        _transform_rows_function_map[mt.row.dtype] = f
    transform_row = _transform_rows_function_map[mt.row.dtype]
    return Table(TableMapRows(mt._tir, Apply(transform_row._name, transform_row._ret_type, TopLevelReference('row'))))

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

def combine_gvcfs(mts):
    """merges vcfs using multi way join"""
    ts = hl.Table.multi_way_zip_join([localize(mt) for mt in mts], 'data', 'g')
    combined = combine(ts)
    return unlocalize(combined)

@typecheck(lgt=expr_call, la=expr_array(expr_int32))
def lgt_to_gt(lgt, la):
    """A method for transforming Local GT and Local Alleles into the true GT"""
    return hl.call(la[lgt[0]], la[lgt[1]])

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
                   reference_genome.lengths[reference_genome.contigs[-1]])

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
