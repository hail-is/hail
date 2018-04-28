from hail.typecheck import *
from hail.utils.java import *


class KinshipMatrix(object):
    """
    Represents a symmetric matrix encoding the relatedness of each pair of samples in the accompanying sample list.

    **We plan to remove this class in Hail 0.2 while maintaining its functionality.**

    The output formats are consistent with `PLINK formats <https://www.cog-genomics.org/plink2/formats>`_ as created by the `make-rel and make-grm commands <https://www.cog-genomics.org/plink2/distance#make_rel>`_ and used by `GCTA <http://cnsgenomics.com/software/gcta/estimate_grm.html>`_.

    """

    def __init__(self, jkm):
        self._key_schema = None
        self._jkm = jkm

    @classmethod
    def _from_block_matrix(cls, bm, col_keys, n_rows):
        return cls(Env.hail().methods.KinshipMatrix.apply(
            Env.hc()._jhc,
            bm._jbm,
            col_keys._jt,
            n_rows))

    @property
    def key_schema(self):
        """
        Returns the signature of the key indexing this matrix.

        :rtype: :class:`.HailType`
        """
        from hail.expr import HailType
        if self._key_schema is None:
            self._key_schema = HailType._from_java(self._jkm.sampleSignature())
        return self._key_schema

    def sample_list(self):
        """
        Gets the list of samples. The (i, j) entry of the matrix encodes the relatedness of the ith and jth samples.

        :return: List of samples.
        :rtype: list of str
        """
        return [self.key_schema._convert_to_py(s) for s in self._jkm.sampleIds()]

    def matrix(self):
        """
        Gets the matrix backing this kinship matrix.

        :return: MatrixTable of kinship values.
        :rtype: `IndexedRowMatrix <https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.linalg.distributed.IndexedRowMatrix>`__
        """
        from pyspark.mllib.linalg.distributed import IndexedRowMatrix

        return IndexedRowMatrix(self._jkm.matrix())

    @typecheck_method(output=str)
    def export_tsv(self, output):
        """
        Export kinship matrix to tab-delimited text file with sample list as header.

        :param str output: File path for output. 
        """
        self._jkm.exportTSV(output)

    @typecheck_method(output=str)
    def export_rel(self, output):
        """
        Export kinship matrix as .rel file. See `PLINK formats <https://www.cog-genomics.org/plink2/formats>`_.

        :param str output: File path for output. 
        """
        self._jkm.exportRel(output)

    @typecheck_method(output=str)
    def export_gcta_grm(self, output):
        """
        Export kinship matrix as .grm file. See `PLINK formats <https://www.cog-genomics.org/plink2/formats>`_.

        :param str output: File path for output.
        """
        self._jkm.exportGctaGrm(output)

    @typecheck_method(output=str,
                      opt_n_file=nullable(str))
    def export_gcta_grm_bin(self, output, opt_n_file=None):
        """
        Export kinship matrix as .grm.bin file or as .grm.N.bin file, depending on whether an N file is specified. See `PLINK formats <https://www.cog-genomics.org/plink2/formats>`_.

        :param str output: File path for output. 
        
        :param opt_n_file: The file path to the N file. 
        :type opt_n_file: str or None
        """
        self._jkm.exportGctaGrmBin(output, joption(opt_n_file))

    @typecheck_method(output=str)
    def export_id_file(self, output):
        """
        Export samples as .id file. See `PLINK formats <https://www.cog-genomics.org/plink2/formats>`_.

        :param str output: File path for output.
        """
        self._jkm.exportIdFile(output)
