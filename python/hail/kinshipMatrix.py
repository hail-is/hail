from pyspark.mllib.linalg.distributed import IndexedRowMatrix

class KinshipMatrix:
    """
    Represents a matrix whose values indicate how related two samples in the accompanying sample list are. This matrix should
    always be symmetric.
    """
    def __init__(self, jkm):
        self._jkm = jkm

    def sample_list(self):
        """
        Gets the list of samples that are represented in this matrix. The relationship between the ith and jth sample in this list
        is represented at index (i, j) of the matrix.

        :return: The list of samples represented in matrix.
        :rtype: list of str
        """
        return list(self._jkm.samples())

    def matrix(self):
        """
        Gets the actual matrix backing this KinshipMatrix.

        :return: The matrix of kinship values.
        :rtype: `IndexedRowMatrix <https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.linalg.distributed.IndexedRowMatrix>`_
        """
        return IndexedRowMatrix(self._jkm.matrix())
