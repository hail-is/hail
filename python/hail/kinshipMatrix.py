from pyspark.mllib.linalg.distributed import IndexedRowMatrix

class KinshipMatrix:
    """
    Represents a symmetric matrix encoding the relatedness of each pair of samples in the accompanying sample list.
    """
    def __init__(self, jkm):
        self._jkm = jkm

    def sample_list(self):
        """
        Gets the list of samples. The (i, j) entry of the matrix encodes the relatedness of the ith and jth samples.

        :return: List of samples.
        :rtype: list of str
        """
        return list(self._jkm.samples())

    def matrix(self):
        """
        Gets the matrix backing this kinship matrix.

        :return: Matrix of kinship values.
        :rtype: `IndexedRowMatrix <https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.linalg.distributed.IndexedRowMatrix>`_
        """
        return IndexedRowMatrix(self._jkm.matrix())
