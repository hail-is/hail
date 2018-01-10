from .family_methods import trio_matrix
from .statgen import linreg, ld_matrix, pca, hwe_normalized_pca
from .qc import sample_qc, variant_qc
from .misc import rename_duplicates

__all__ = ['trio_matrix',
           'linreg',
           'ld_matrix',
           'sample_qc',
           'variant_qc',
           'pca',
           'hwe_normalized_pca',
           'rename_duplicates']
