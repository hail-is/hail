class Eigendecomposition:
    """
    Represents the eigendecomposition of a symmetric matrix.
    """
    def __init__(self, jeigen):
        self._jeigen = jeigen
        
    def take(self, k):
        """
        Take the top k eigenvectors and eigenvalues.
        If k is greater than the number present, then the calling object is returned.

        :return: The top k eigenvectors and eigenvalues.
        :rtype: Eigendecomposition
        """
        
        return Eigendecomposition(self._jeigen.take(k))