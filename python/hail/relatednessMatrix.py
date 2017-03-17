from pyspark.mllib.linalg.distributed import BlockMatrix

class RelatednessMatrix:
    """
    Represents a matrix whose values indicate how related two samples in the accompanying sample list are.
    """
    def __init__(self, jrm):
        self._jrm = jrm

    def sample_list(self):
        return list(self._jrm.samples())

    def matrix(self):
        return BlockMatrix(self._jrm.matrix(), self._jrm.matrix().rowsPerBlock, self._jrm.matrix().colsPerBlock)
