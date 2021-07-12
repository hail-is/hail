import hail as hl
import os

from hail.utils.java import Env
from hail.utils.misc import divide_null
from typing import Sequence


def read_vds(path):
    """

    Parameters
    ----------
    path: :obj:`str`

    Returns
    -------
    :class:`.VariantDataset`
    """
    reference_data = hl.read_matrix_table(VariantDataset._reference_path(path))
    variant_data = hl.read_matrix_table(VariantDataset._variants_path(path))

    return VariantDataset(reference_data, variant_data)

class VariantDataset(object):
    """
    Class for representing cohort-level genomic data.
    """

    @staticmethod
    def _reference_path(base: str) -> str:
        return os.path.join(base, 'reference_data')

    @staticmethod
    def _variants_path(base: str) -> str:
        return os.path.join(base, 'variant_data')


    @staticmethod
    def from_merged_representation(mt, ref_block_fields = ('GQ', 'DP', 'MIN_DP')):
        gt_field = 'LGT' if 'LGT' in mt.entry else 'GT'
        rmt = mt.filter_entries(mt[gt_field].is_hom_ref())
        rmt = rmt.select_entries(*(x for x in ref_block_fields if x in rmt.entry), 'END')
        rmt = rmt.filter_rows(hl.agg.count() > 0)

        # drop other alleles
        rmt = rmt.key_rows_by(rmt.locus)
        rmt = rmt.annotate_rows(alleles=rmt.alleles[:1])

        vmt = mt.filter_entries(mt[gt_field].is_non_ref() | hl.is_missing(mt[gt_field]))
        vmt = vmt.filter_rows(hl.agg.count() > 0)

        return VariantDataset(rmt, vmt)

    def __init__(self, reference_data: 'hl.MatrixTable', variant_data: 'hl.MatrixTable'):
        self.reference_data: 'hl.MatrixTable' = reference_data
        self.variant_data: 'hl.MatrixTable' = variant_data

    def write(self, path, **kwargs):
        self.reference_data.write(VariantDataset._reference_path(path), **kwargs)
        self.variant_data.write(VariantDataset._variants_path(path), **kwargs)

    def checkpoint(self, path, **kwargs):
        self.write(path, **kwargs)
        return hl.read_vds(path)

    def filter_intervals(self, intervals, *, keep=False):
        # for now, don't touch reference data.
        # should remove large regions and scan forward ref blocks to the start of the next kept region
        self.variant_data = hl.filter_intervals(self.variant_data, intervals, keep)

    def merged_sparse_mt(self):
        rht = self.reference_data \
            .localize_entries('ref_entries', 'ref_cols')
        vht = self.variant_data.localize_entries('var_entries', 'var_cols').rename({'alleles': 'var_alleles'})

        merged_schema = {}
        for e in self.reference_data.entry:
            merged_schema[e] = self.reference_data[e].dtype
        for e in self.variant_data.entry:
            if e in merged_schema:
                if not merged_schema[e] == self.variant_data[e].dtype:
                    raise TypeError(f"cannot unify field {e!r}: {merged_schema[e]}, {self.variant_data[e].dtype}")
            else:
                merged_schema[e] = self.variant_data[e].dtype

        ht = rht.join(vht, how='outer').drop('ref_cols')

        def merge_arrays(r_array, v_array):

            def rewrite_ref(r):
                ref_block_selector = {}
                for k, t in merged_schema.items():
                    if k == 'LA':
                        ref_block_selector[k] = hl.literal([0])
                    elif k in ('LGT', 'GT'):
                        ref_block_selector[k] = hl.call(0, 0)
                    else:
                        ref_block_selector[k] = r[k] if k in r else hl.missing(t)
                return r.select(**ref_block_selector)

            def rewrite_var(v):
                return v.select(**{
                    k: v[k] if k in v else hl.missing(t)
                    for k, t in merged_schema.items()
                })

            return hl.case() \
                .when(hl.is_missing(r_array), v_array.map(rewrite_var)) \
                .when(hl.is_missing(v_array), r_array.map(rewrite_ref)) \
                .default(hl.zip(r_array, v_array).map(lambda t: hl.coalesce(rewrite_ref(t[0]), rewrite_var(t[1]))))

        ht = ht.select(alleles=hl.coalesce(ht['var_alleles'], ht['alleles']),
                       **{k: ht[k] for k in self.variant_data.row_value if k != 'alleles'}, # handle cases where vmt is not keyed by alleles
                       entries=merge_arrays(ht['ref_entries'], ht['var_entries']))
        return ht._unlocalize_entries('entries', 'var_cols', list(self.variant_data.col_key))


    def dense_mt(self):
        # we can do something more efficient here!
        return hl.experimental.densify(self.merged_sparse_mt())

    def split_multi(self, *, filter_changed_loci: bool = False):
        self.variant_data = hl.experimental.sparse_split_multi(self.variant_data, filter_changed_loci=filter_changed_loci)

    def sample_qc(self, *, name='sample_qc', gq_bins: 'Sequence[int]' = (0, 20, 60)) -> 'hl.Table':

        from hail.expr.functions import _num_allele_type, _allele_types

        allele_types = _allele_types[:]
        allele_types.extend(['Transition', 'Transversion'])
        allele_enum = {i: v for i, v in enumerate(allele_types)}
        allele_ints = {v: k for k, v in allele_enum.items()}

        def allele_type(ref, alt):
            return hl.bind(lambda at: hl.if_else(at == allele_ints['SNP'],
                                                 hl.if_else(hl.is_transition(ref, alt),
                                                            allele_ints['Transition'],
                                                            allele_ints['Transversion']),
                                                 at),
                           _num_allele_type(ref, alt))

        variant_ac = Env.get_uid()
        variant_atypes = Env.get_uid()

        vmt = self.variant_data
        if not 'GT' in vmt.entry:
            vmt = vmt.annotate_entries(GT=hl.experimental.lgt_to_gt(vmt.LGT, vmt.LA))


        vmt = vmt.annotate_rows(**{variant_ac: hl.agg.call_stats(vmt.GT, vmt.alleles).AC,
                                 variant_atypes: vmt.alleles[1:].map(lambda alt: allele_type(vmt.alleles[0], alt))})

        bound_exprs = {}

        bound_exprs['n_het'] = hl.agg.count_where(vmt['GT'].is_het())
        bound_exprs['n_hom_var'] = hl.agg.count_where(vmt['GT'].is_hom_var())
        bound_exprs['n_singleton'] = hl.agg.sum(hl.sum(hl.range(0, vmt['GT'].ploidy).map(lambda i: vmt[variant_ac][vmt['GT'][i]] == 1)))

        def get_allele_type(allele_idx):
            return hl.if_else(allele_idx > 0, vmt[variant_atypes][allele_idx - 1], hl.missing(hl.tint32))

        bound_exprs['allele_type_counts'] = hl.agg.explode(
            lambda elt: hl.agg.counter(elt),
            hl.range(0, vmt['GT'].ploidy).map(lambda i: get_allele_type(vmt['GT'][i])))

        zero = hl.int64(0)

        gq_exprs = hl.agg.filter(hl.is_defined(vmt.GT),
                                 hl.struct(**{f'gq_over_{x}': hl.agg.count_where(vmt.GQ > x)
                                              for x in gq_bins}))

        result_struct = hl.rbind(
            hl.struct(**bound_exprs),
            lambda x: hl.rbind(
                hl.struct(**{
                    'gq_exprs': gq_exprs,
                    'n_het': x.n_het,
                    'n_hom_var': x.n_hom_var,
                    'n_non_ref': x.n_het + x.n_hom_var,
                    'n_singleton': x.n_singleton,
                    'n_snp': (x.allele_type_counts.get(allele_ints["Transition"], zero)
                              + x.allele_type_counts.get(allele_ints["Transversion"], zero)),
                    'n_insertion': x.allele_type_counts.get(allele_ints["Insertion"], zero),
                    'n_deletion': x.allele_type_counts.get(allele_ints["Deletion"], zero),
                    'n_transition': x.allele_type_counts.get(allele_ints["Transition"], zero),
                    'n_transversion': x.allele_type_counts.get(allele_ints["Transversion"], zero),
                    'n_star': x.allele_type_counts.get(allele_ints["Star"], zero)
                }),
                lambda s: s.annotate(
                    r_ti_tv=divide_null(hl.float64(s.n_transition), s.n_transversion),
                    r_het_hom_var=divide_null(hl.float64(s.n_het), s.n_hom_var),
                    r_insertion_deletion=divide_null(hl.float64(s.n_insertion), s.n_deletion)
                )))
        variant_results = vmt.select_cols(**result_struct).cols()

        rmt = self.reference_data
        ref_results = rmt.select_cols(gq_exprs=hl.struct(**{
            f'gq_over_{x}': hl.agg.filter(rmt.GQ > x, hl.agg.sum(1 + rmt.END - rmt.locus.position))
            for x in gq_bins
        })).cols()

        joined = ref_results[variant_results.key].gq_exprs
        return variant_results.transmute(**{
            f'gq_over_{x}': variant_results.gq_exprs[f'gq_over_{x}'] + joined[f'gq_over_{x}']
                for x in gq_bins
        })

    def filter_samples(self, samples_table: 'hl.Table', *, keep: bool = True):
        if not list(samples_table[x].dtype for x in samples_table.key) == [hl.tstr]:
            raise TypeError(f'invalid key: {samples_table.key.dtype}')
        samples_to_keep = samples_table.aggregate(hl.agg.collect_as_set(samples_table.key[0]), _localize=False)._persist()
        self.reference_data = self.reference_data.filter_cols(samples_to_keep.contains(self.reference_data.key[0]), keep=keep)
        self.variant_data = self.variant_data.filter_cols(samples_to_keep.contains(self.variant_data.key[0]), keep=keep)

    def filter_variants(self, variants_table: 'hl.Table', *, keep: bool = True):
        # don't remove reference data
        if keep:
            self.variant_data = self.variant_data.semi_join_rows(variants_table)
        else:
            self.variant_data = self.variant_data.anti_join_rows(variants_table)

