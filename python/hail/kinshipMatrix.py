from pyspark.mllib.linalg.distributed import IndexedRowMatrix

class KinshipMatrix:
    """
    Represents a matrix whose values indicate how related two samples in the accompanying sample list are. This matrix should
    always be symmetric.
    """
    def __init__(self, jkm):
        self._jkm = jkm

    def sample_list(self):
        return list(self._jkm.samples())

    def matrix(self):
        return IndexedRowMatrix(self._jkm.matrix())
