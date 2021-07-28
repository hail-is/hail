import os

import hail as hl
from hail.matrixtable import MatrixTable
from hail.utils.java import info


def read_vds(path) -> 'VariantDataset':
    """Read in a :class:`.VariantDataset` written with :meth:`.VariantDataset.write`.

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


class VariantDataset:
    """Class for representing cohort-level genomic data.

    This class facilitates a sparse, split representation of genomic data in
    which reference block data and variant data are contained in separate
    :class:`.MatrixTable` objects.

    Parameters
    ----------
    reference_data : :class:`.MatrixTable`
        MatrixTable containing only reference block data.
    variant_data : :class:`.MatrixTable`
        MatrixTable containing only variant data.
    """

    @staticmethod
    def _reference_path(base: str) -> str:
        return os.path.join(base, 'reference_data')

    @staticmethod
    def _variants_path(base: str) -> str:
        return os.path.join(base, 'variant_data')

    @staticmethod
    def from_merged_representation(mt,
                                   *,
                                   ref_block_fields=(),
                                   infer_ref_block_fields: bool = True):

        if 'END' not in mt.entry:
            raise ValueError("VariantDataset.from_merged_representation: expect field 'END' in matrix table entry")

        if 'LA' not in mt.entry:
            raise ValueError("VariantDataset.from_merged_representation: expect field 'LA' in matrix table entry")

        if 'GT' not in mt.entry and 'LGT' not in mt.entry:
            raise ValueError(
                "VariantDataset.from_merged_representation: expect field 'LGT' or 'GT' in matrix table entry")

        n_rows_to_use = 100
        info(f"inferring reference block fields from missingness patterns in first {n_rows_to_use} rows")
        used_ref_block_fields = set(ref_block_fields)
        used_ref_block_fields.add('END')

        if infer_ref_block_fields:
            mt_head = mt.head(n_rows=n_rows_to_use)
            for k, any_present in zip(
                    list(mt_head.entry),
                    mt_head.aggregate_entries(hl.agg.filter(hl.is_defined(mt_head.END), tuple(
                        hl.agg.any(hl.is_defined(mt_head[x])) for x in mt_head.entry)))):
                if any_present:
                    used_ref_block_fields.add(k)

        gt_field = 'LGT' if 'LGT' in mt.entry else 'GT'

        # remove LGT/GT and LA fields, which are trivial for reference blocks and do not need to be represented
        if gt_field in used_ref_block_fields:
            used_ref_block_fields.remove(gt_field)
        if 'LA' in used_ref_block_fields:
            used_ref_block_fields.remove('LA')

        info("Including the following fields in reference block table:" + "".join(
            f"\n  {k!r}" for k in mt.entry if k in used_ref_block_fields))

        rmt = mt.filter_entries(hl.case()
                                .when(hl.is_missing(mt.END), False)
                                .when(hl.is_defined(mt.END) & mt[gt_field].is_hom_ref(), True)
                                .or_error(hl.str('cannot create VDS from merged representation -'
                                                 ' found END field with non-reference genotype at ')
                                          + hl.str(mt.locus) + hl.str(' / ') + hl.str(mt.col_key[0])))
        rmt = rmt.select_entries(*(x for x in rmt.entry if x in used_ref_block_fields))
        rmt = rmt.filter_rows(hl.agg.count() > 0)

        # drop other alleles
        rmt = rmt.key_rows_by(rmt.locus)
        rmt = rmt.select_rows(ref_allele=rmt.alleles[0])

        vmt = mt.filter_entries(hl.is_missing(mt.END))
        vmt = vmt.filter_rows(hl.agg.count() > 0)

        return VariantDataset(rmt, vmt)

    def __init__(self, reference_data: 'MatrixTable', variant_data: 'MatrixTable'):
        self.reference_data: 'MatrixTable' = reference_data
        self.variant_data: 'MatrixTable' = variant_data

    def write(self, path, **kwargs):
        self.reference_data.write(VariantDataset._reference_path(path), **kwargs)
        self.variant_data.write(VariantDataset._variants_path(path), **kwargs)

    def checkpoint(self, path, **kwargs) -> 'VariantDataset':
        self.write(path, **kwargs)
        return read_vds(path)
