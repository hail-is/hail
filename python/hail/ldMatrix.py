from hail.representation import Variant
from hail.history import *


class LDMatrix(HasHistory):
    """
    Represents a symmetric matrix encoding the Pearson correlation between each pair of variants in the accompanying variant list.
    """
    def __init__(self, jldm):
        self._jldm = jldm
        super(LDMatrix, self).__init__()

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
        
        :return: Matrix of Pearson correlation values.
        :rtype: `Matrix <https://spark.apache.org/docs/2.1.0/api/python/pyspark.mllib.html#pyspark.mllib.linalg.Matrix>`__
        """
        from pyspark.mllib.linalg import DenseMatrix

        j_local_mat = self._jldm.toLocalMatrix()
        return DenseMatrix(j_local_mat.numRows(), j_local_mat.numCols(), list(j_local_mat.toArray()), j_local_mat.isTransposed())
