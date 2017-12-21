from hail.history import *
from hail.typecheck import *
from hail.utils.java import *

import hail

class LDMatrix(HistoryMixin):
    """
    Represents a symmetric matrix encoding the Pearson correlation between each pair of variants in the accompanying variant list.
    """
    def __init__(self, jldm):
        self._jldm = jldm
        self._rg = hail.genetics.GenomeReference._from_java(jldm.vTyp().gr())

    def variant_list(self):
        """
        Gets the list of variants. The (i, j) entry of the matrix encodes the Pearson correlation between the ith and jth variants.

        :return: List of variants.
        :rtype: list of Variant
        """
        jvars = self._jldm.variants()

        return list(map(lambda jrep: hail.genetics.Variant._from_java(jrep, self._rg), jvars))

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
        :rtype: `Matrix <https://spark.apache.org/docs/2.1.0/api/python/pyspark.mllib.html#pyspark.mllib.linalg.MatrixTable>`__
        """
        from pyspark.mllib.linalg import DenseMatrix

        j_local_mat = self._jldm.toLocalMatrix()
        assert j_local_mat.majorStride() == j_local_mat.rows()
        assert j_local_mat.offset() == 0
        assert j_local_mat.isTranspose() == False
        return DenseMatrix(j_local_mat.rows(), j_local_mat.cols(), list(j_local_mat.data()), False)

    @handle_py4j
    @write_history('path', is_dir=True)
    @typecheck_method(path=strlike)
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
    @handle_py4j
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

    @handle_py4j
    @write_history('path', is_dir=False, parallel='parallel')
    @typecheck_method(path=strlike,
                      column_delimiter=strlike,
                      header=nullable(strlike),
                      parallel=nullable(enumeration('separate_header', 'header_per_shard')),
                      entries=enumeration('full', 'lower', 'strict_lower', 'upper', 'strict_upper'))
    def export(self, path, column_delimiter, header=None, parallel=None, entries='full'):
        """Exports this matrix as a delimited text file.

        **Examples**

        Write a full LD matrix as a tab-separated file:

        >>> vds.ld_matrix().export('output/ld_matrix.tsv', column_delimiter='\t')

        Write a full LD matrix as a comma-separated file with the variant list as a header:

        >>> ldm = vds.ld_matrix()
        >>> ldm.export('output/ld_matrix.tsv',
        ...            column_delimiter=',',
        ...            header=','.join([str(v) for v in ldm.variant_list()]))

        Write a full LD matrix as a folder of comma-separated file shards:

        >>> ldm = vds.ld_matrix()
        >>> ldm.export('output/ld_matrix.tsv',
        ...            column_delimiter=',',
        ...            header=None,
        ...            parallel=True)

        Write the upper-triangle with the diagonal as a comma-separated file:

        >>> ldm = vds.ld_matrix()
        >>> ldm.export('output/ld_matrix.tsv',
        ...            column_delimiter=',',
        ...            entries='upper')

        **Notes**

        A matrix cannot be exported if it has more than ``2^31 - 1`` columns.

        A full, 3x3 LD matrix written as a comma-separated file looks like this:

        .. code-block:: text

            1.0,0.8,0.7
            0.8,1.0,0.3
            0.7,0.3,1.0

        The strict lower triangle:

        .. code-block:: text

            0.8
            0.7,0.3

        The lower triangle:

        .. code-block:: text

            1.0
            0.8,1.0
            0.7,0.3,1.0

        The strict upper triangle:

        .. code-block:: text

            0.8,0.7
            0.3

        The upper triangle:

        .. code-block:: text

            1.0,0.8,0.7
            1.0,0.3
            1.0

        :param path: the path at which to write the LD matrix
        :type path: str

        :param column_delimiter: the column delimiter
        :type column_delimiter: str

        :param header: a string to append before the first row of the matrix
        :type path: str or None

        :param parallel: If None, a single file is produced, otherwise a
                               folder of file shards is produced. If 'separate_header',
                               the header file is output separately from the file shards. If
                               'header_per_shard', each file shard has a header. If set to None
                               the export will be slower.
        :type parallel: str or None

        :param entries: describes what portion of the entries should be printed,
                        see the notes for a detailed description
        :type entries: str

        """

        parallel = Env.hail().utils.ExportType.getExportType(joption(parallel))

        if entries == 'full':
            self._jldm.export(path, column_delimiter, joption(header), parallel)
        elif entries == 'lower':
            self._jldm.exportLowerTriangle(path, column_delimiter, joption(header), parallel)
        elif entries == 'strict_lower':
            self._jldm.exportStrictLowerTriangle(path, column_delimiter, joption(header), parallel)
        elif entries == 'upper':
            self._jldm.exportUpperTriangle(path, column_delimiter, joption(header), parallel)
        else:
            self._jldm.exportStrictUpperTriangle(path, column_delimiter, joption(header), parallel)
