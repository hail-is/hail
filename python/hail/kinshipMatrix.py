from pyspark.mllib.linalg.distributed import IndexedRowMatrix
from hail.java import *

class KinshipMatrix:
    """
    Represents a symmetric matrix encoding the relatedness of each pair of samples in the accompanying sample list.
    
    The output formats are consistent with `PLINK formats <https://www.cog-genomics.org/plink2/formats>`_ as created by the `make-rel and make-grm commands <https://www.cog-genomics.org/plink2/distance#make_rel>`_ and used by `GCTA <http://cnsgenomics.com/software/gcta/estimate_grm.html>`_.

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
        :rtype: `IndexedRowMatrix <https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.linalg.distributed.IndexedRowMatrix>`__
        """
        return IndexedRowMatrix(self._jkm.matrix())

    def export_tsv(self, output):
        """
        Export kinship matrix to tab-delimited text file with sample list as header.
        
        :param string output: The file path to output the matrix to. 
        """
        self._jkm.exportTSV(output)

    def export_rel(self, output):
        """
        Export kinship matrix as .rel file. See `PLINK formats <https://www.cog-genomics.org/plink2/formats>`_.
        
        :param output: The file path to output to. 
        """
        self._jkm.exportRel(output)

    def export_gcta_grm(self, output):
        """
        Export kinship matrix as .grm file. See `PLINK formats <https://www.cog-genomics.org/plink2/formats>`_.
        
        :param output: The file path to output to.
        """
        self._jkm.exportGctaGrm(output)

    def export_gcta_grm_bin(self, output, opt_n_file=None):
        """
        Export kinship matrix as .grm.bin file or as .grm.N.bin file, depending on whether an N file is specified. See `PLINK formats <https://www.cog-genomics.org/plink2/formats>`_.
        
        :param output: The file path to output to. 
        
        :param opt_n_file: The file path to the N file. 
        """
        self._jkm.exportGctaGrmBin(output, joption(opt_n_file))

    def export_id_file(self, output):
        """
        Export samples as .id file. See `PLINK formats <https://www.cog-genomics.org/plink2/formats>`_.
        
        :param output: The file to output to.
        """
        self._jkm.exportIdFile(output)
