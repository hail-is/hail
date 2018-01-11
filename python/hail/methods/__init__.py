from .family_methods import trio_matrix
from .statgen import linreg, sample_rows, ld_matrix, pca, hwe_normalized_pca, split_multi_hts
from .qc import sample_qc, variant_qc
from .misc import rename_duplicates

__all__ = ['trio_matrix',
           'linreg',
           'sample_rows',
           'ld_matrix',
           'sample_qc',
           'variant_qc',
           'pca',
           'hwe_normalized_pca',
           'rename_duplicates',
           'split_multi_hts']
