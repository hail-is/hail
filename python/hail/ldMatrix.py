from hail.java import *
from hail.representation import Variant
from hail.eigen import Eigen
from hail.typecheck import *

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
        return DenseMatrix(j_local_mat.numRows(), j_local_mat.numCols(), list(j_local_mat.toArray()), j_local_mat.isTransposed())

    def write(self, path):
        """
        Writes the LD matrix to a file.

        **Examples**

        Write an LD matrix to a file.

        >>> vds.ld_matrix().write('output/ld_matrix')

        :param path: the path to which to write the LD matrix
        :type path: str
        """

        self._jldm.write(path)

    @staticmethod
    @typecheck(path=strlike)
    def read(path):
        """
        Reads the LD matrix from a file.

        **Examples**

        Read an LD matrix from a file.

        >>> ld_matrix = LDMatrix.read('data/ld_matrix')

        :param path: the path from which to read the LD matrix
        :type path: str
        """

        jldm = Env.hail().methods.LDMatrix.read(Env.hc()._jhc, path)
        return LDMatrix(jldm)
    
    def eigen(self):
        """
        Compute an eigendecomposition of the LD matrix. The number of eigenvectors returned is the minimum of
        the number of samples used to form the LD matrix and the number of variants.
        
        .. caution::
        
            This method collects the LD matrix to a local matrix on the driver in order to compute the full
            eigendecomposition using LAPACK. Only call this method when the LD matrix is small enough to fit in
            local memory; the absolute limit on the number of variants is 32k.
        
        :return: Eigendecomposition of the LD matrix.
        :rtype: :py:class:`.Eigen`
        """
        return Eigen(self._jldm.eigen())
