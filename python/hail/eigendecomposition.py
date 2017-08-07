from hail.typecheck import *

class Eigendecomposition:
    """
    Represents the eigendecomposition of a symmetric matrix.
    """
    def __init__(self, jeigen):
        self._jeigen = jeigen
        self._key_schema = None
    
    @property
    def key_schema(self):
        """
        Returns the signature of the key indexing the rows.

        :rtype: :class:`.Type`
        """

        if self._key_schema is None:
            self._key_schema = Type._from_java(self._jeigen.rowSignature())
        return self._key_schema
    
    def sample_list(self):
        """
        Gets the list of samples.

        :return: List of samples.
        :rtype: list of str
        """
        return [self.key_schema._convert_to_py(s) for s in self._jeigen.rowIds()]

    def evects(self):
        """
        Gets the matrix whose columns are eigenvectors, ordered by increasing eigenvalue.
                
        :return: Matrix of eigenvectors.
        :rtype: `Matrix <https://spark.apache.org/docs/2.1.0/api/python/pyspark.mllib.html#pyspark.mllib.linalg.Matrix>`__
        """
        from pyspark.mllib.linalg import DenseMatrix

        j_evects = self._jeigen.evectsSpark()
        return DenseMatrix(j_evects.numRows(), j_evects.numCols(), list(j_evects.values()), j_evects.isTransposed())


    def evals(self):
        """
        Gets the eigenvalues.

        :return: List of eigenvalues in increasing order.
        :rtype: list of float
        """
        return list(self._jeigen.evalsArray())
    
    def n_evects(self):
        """
        Gets the number of eigenvectors and eigenvalues.
        
        :return: Number of eigenvectors and eigenvalues.
        :rtype: int
        """
        return self._jeigen.nEvects()
    
    @typecheck_method(k=integral)
    def take_right(self, k):
        """
        Take the top k eigenvectors and eigenvalues.
        If k is greater than the number present, then the calling eigendecomposition is returned.

        :param int k: Number of eigenvectors and eigenvalues to return.

        :return: The top k eigenvectors and eigenvalues.
        :rtype: Eigendecomposition
        """
        
        return Eigendecomposition(self._jeigen.take_right(k))
