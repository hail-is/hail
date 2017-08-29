from hail.typecheck import *
from hail.java import *
from hail.expr import Type

class Eigen:
    """
    Represents the eigenvectors and eigenvalues of a matrix.
    """
    def __init__(self, jeigen):
        self._jeigen = jeigen
        self._key_schema = None
    
    @property
    def key_schema(self):
        """Returns the signature of the key indexing the rows.

        :rtype: :class:`.Type`
        """

        if self._key_schema is None:
            self._key_schema = Type._from_java(self._jeigen.rowSignature())
        return self._key_schema
    
    def row_ids(self):
        """Gets the list of row IDs.

        :return: List of row IDs of type key_schema
        """
        return [self.key_schema._convert_to_py(s) for s in self._jeigen.rowIds()]

    def evects(self):
        """Gets the matrix whose columns are eigenvectors, ordered by increasing eigenvalue.
                
        :return: Matrix of whose columns are eigenvectors.
        :rtype: `Matrix <https://spark.apache.org/docs/2.1.0/api/python/pyspark.mllib.html#pyspark.mllib.linalg.Matrix>`__
        """
        from pyspark.mllib.linalg import DenseMatrix

        j_evects = self._jeigen.evectsSpark()
        return DenseMatrix(j_evects.numRows(), j_evects.numCols(), list(j_evects.values()), j_evects.isTransposed())


    def evals(self):
        """Gets the eigenvalues.

        :return: List of eigenvalues in increasing order.
        :rtype: list of float
        """
        return list(self._jeigen.evalsArray())
    
    def num_evects(self):
        """Gets the number of eigenvectors and eigenvalues.
        
        :return: Number of eigenvectors and eigenvalues.
        :rtype: int
        """
        return self._jeigen.nEvects()
    
    @typecheck_method(k=integral)
    def take_top(self, k):
        """Take the top k eigenvectors and eigenvalues.
        
       **Notes** 
       
        If k is greater than or equal to the number present, then the calling eigendecomposition is returned.

        :param int k: Number of eigenvectors and eigenvalues to return.

        :return: The top k eigenvectors and eigenvalues.
        :rtype: :py:class:`.Eigen`
        """
        
        return Eigen(self._jeigen.takeTop(k))
    
    @typecheck_method(proportion=numeric)
    def drop_proportion(self, proportion = 1e-6):
        """Drop the maximum number of eigenvectors without cumulatively exceeding the given proportion of
        the total variance.
        
        **Notes**
        
        The total variance is the sum of all eigenvalues.
        
        For example, if the eigenvalues are [0.0, 1.0, 2.0, 97.0] then the proportions 0.0, 0.01, 0.02, and 0.03 will
        drop 1, 2, 2, and 3 eigenvectors, respectively.

        :param float proportion: Proportion in the interval [0,1)

        :return: Eigendecomposition
        :rtype: :py:class:`.Eigen`
        """
        
        return Eigen(self._jeigen.dropProportion(proportion))

    @typecheck_method(threshold=numeric)
    def drop_threshold(self, threshold=1e-6):
        """Drop eigenvectors with eigenvalues at or below the threshold.
        
        **Notes**
        
        For example, if the eigenvalues are [0.0, 1.0, 2.0, 97.0] then the thresholds 0.0, 0.01, 0.02, and 0.03 will
        drop 1, 2, 3, and 3 eigenvectors, respectively.

        :param float threshold: Non-negative threshold 

        :return: Eigendecomposition
        :rtype: :py:class:`.Eigen`
        """
        
        return Eigen(self._jeigen.dropThreshold(threshold))
    
    def distribute(self):
        """Convert to a distributed eigendecomposition.
        
        :return: Distributed eigendecomposition.
        :rtype: :py:class:`.EigenDistributed`
        """
        
        return EigenDistributed(self._jeigen.distribute(Env.hc()._jsc))
    
    @typecheck_method(path=strlike)
    def write(self, path):
        """Writes the eigendecomposition to a path.

        >>> vds.rrm().eigen().write('output/example.eig')

        :param str path: path to directory ending in ``.eig`` to which to write the eigendecomposition
        """

        self._jeigen.write(Env.hc()._jhc, path)
        
    @staticmethod
    def read(path):
        """Reads the eigendecomposition from a path.

        >>> eig = Eigen.read('data/example.eig')

        :param str path: path to directory ending in ``.eig`` from which to read the LD matrix
        
        :return: Eigendecomposition
        :rtype: :py:class:`.Eigen`
        """

        jeigen = Env.hail().stats.Eigen.read(Env.hc()._jhc, path)
        return Eigen(jeigen)

class EigenDistributed:
    """
    Represents the eigenvectors and eigenvalues of a matrix. Eigenvectors are stored as columns of a distributed matrix.
    """
    def __init__(self, jeigen):
        self._jeigen = jeigen
        self._key_schema = None
    
    @property
    def key_schema(self):
        """Returns the signature of the key indexing the rows.

        :rtype: :class:`.Type`
        """

        if self._key_schema is None:
            self._key_schema = Type._from_java(self._jeigen.rowSignature())
        return self._key_schema
    
    def row_ids(self):
        """Gets the list of row IDs.

        :return: List of rows.
        :rtype: list of str
        """
        return [self.key_schema._convert_to_py(s) for s in self._jeigen.rowIds()]

    def evects(self):
        """Gets the block matrix whose columns are eigenvectors, ordered by increasing eigenvalue.
                
        :return: Matrix whose columns are eigenvectors.
        :rtype: `BlockMatrix <https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.linalg.distributed.BlockMatrix>`__
        """
        from pyspark.mllib.linalg.distributed import BlockMatrix

        return BlockMatrix(self._jeigen.evects())

    def evals(self):
        """Gets the eigenvalues.

        :return: List of eigenvalues in increasing order.
        :rtype: list of float
        """
        return list(self._jeigen.evalsArray())
    
    def num_evects(self):
        """Gets the number of eigenvectors and eigenvalues.
        
        :return: Number of eigenvectors and eigenvalues.
        :rtype: int
        """
        return self._jeigen.nEvects()
        
    @typecheck_method(path=strlike)
    def write(self, path):
        """Writes the eigendecomposition to a path.

        >>> vds.rrm().eigen().distribute().write('output/example.eigd')

        :param str path: path to directory ending in ``.eigd`` to which to write the eigendecomposition
        """

        self._jeigen.write(path)
        
    @staticmethod
    @typecheck(path=strlike)
    def read(path):
        """Reads the eigendecomposition from a path.

        >>> eig = EigenDistributed.read('data/example.eigd')

        :param str path: path to directory ending in ``.eigd`` from which to read the LD matrix
        
        :return: Eigendecomposition
        :rtype: :py:class:`.EigenDistributed`
        """

        jeigen = Env.hail().stats.EigenDistributed.read(Env.hc()._jhc, path)
        return EigenDistributed(jeigen)
