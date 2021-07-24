import os

import hail as hl
from hail.matrixtable import MatrixTable


def read_vds(path):
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
    def from_merged_representation(mt, ref_block_fields=('GQ', 'DP', 'MIN_DP')):
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

    def __init__(self, reference_data: 'MatrixTable', variant_data: 'MatrixTable'):
        self.reference_data: 'MatrixTable' = reference_data
        self.variant_data: 'MatrixTable' = variant_data

    def write(self, path, **kwargs):
        self.reference_data.write(VariantDataset._reference_path(path), **kwargs)
        self.variant_data.write(VariantDataset._variants_path(path), **kwargs)

    def checkpoint(self, path, **kwargs):
        self.write(path, **kwargs)
        return read_vds(path)
