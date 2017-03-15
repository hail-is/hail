class RelatednessMatrix:
    """
    Represents a matrix whose values indicate how related two samples in the accompanying sample list are.
    """
    def __init__(self, jrm):
        self._jrm = jrm

    def sample_list(self):
        return self._jrm.samples()

    def matrix(self):
        return BlockMatrix(jrm, jrm.rowsPerBlock(), jrm.colsPerBlock())