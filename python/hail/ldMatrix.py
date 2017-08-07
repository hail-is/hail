from hail.representation import Variant
from hail.eigendecomposition import Eigendecomposition
from hail.typecheck import *
from hail.java import *

class LDMatrix:
    """
    Represents a symmetric matrix encoding the Pearson correlation between each pair of variants in the accompanying variant list.
    """
    def __init__(self, jldm):
        self._jldm = jldm

    def variant_list(self):
        """
        Gets the list of variants. The (i, j) entry of the matrix encodes the Pearson correlation between the ith and jth variants.

        :return: List of variants.
        :rtype: list of Variant
        """
        jvars = self._jldm.variants()
        return list(map(lambda jrep: Variant._from_java(jrep), jvars))

    def matrix(self):
        """
        Gets the distributed matrix backing this LD matrix.

        :return: Matrix of Pearson correlation values.
        :rtype: `IndexedRowMatrix <https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.linalg.distributed.IndexedRowMatrix>`__
        """
        from pyspark.mllib.linalg.distributed import IndexedRowMatrix

        return IndexedRowMatrix(self._jldm.matrix())

    def to_local_matrix(self):
        """
        Converts the LD matrix to a local Spark matrix.
        
        .. caution::
        
            Only call this method when the LD matrix is small enough to fit in local memory on the driver.
            The product of the dimensions can be at most :math:`2^31 - 1` (about 2 billion).
        
        :return: Matrix of Pearson correlation values.
        :rtype: `Matrix <https://spark.apache.org/docs/2.1.0/api/python/pyspark.mllib.html#pyspark.mllib.linalg.Matrix>`__
        """
        from pyspark.mllib.linalg import DenseMatrix

        j_local_mat = self._jldm.toLocalMatrix()
        return DenseMatrix(j_local_mat.numRows(), j_local_mat.numCols(), list(j_local_mat.values()), j_local_mat.isTransposed())
    
    @typecheck_method(vds=anytype,
                      k=nullable(integral))
    def eigen_rrm(self, vds, k=None):
        """
        Compute an eigendecomposition of the Realized Relationship Matrix (RRM) of the variant dataset via an
        eigendecomposition of the LD matrix.
        
        *Notes*

        This method computes and then uses eigendecomposition of the LD matrix to derive an eigendecomposition
        of the corresponding RRM. All variants in the LD matrix must be present in the VDS. The number of eigenvectors
        returned is the minimum of variants, the number of samples used to form the LD matrix, and k.
        
        .. caution::
        
            This method collects the LD matrix to a local matrix on the driver in order to compute the full
            eigendecomposition using LAPACK. Only call this method when the LD matrix and the resulting matrix
            of eigenvectors are small enough to fit in local memory. The absolute limit on the number of variants
            is 32k. The absolute limit on the number of elements in the eigenvector matrix is :math:`2^{31} - 1` (about 2 billion).
                    
        :param vds: Variant dataset
        :type vds: :py:class:`.VariantDataset`
        
        :param k: Upper bound on the number of eigenvectors to return.
        :type k: int or None
        
        :return: Eigendecomposition of the kinship matrix.
        :rtype: Eigendecomposition
        """
        
        return Eigendecomposition(self._jldm.eigenRRM(vds._jvds, joption(k)))