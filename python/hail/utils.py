import io

from hail.java import Env, handle_py4j, jiterable_to_list
from hail.typecheck import *


class Summary(object):
    """Class holding summary statistics about a dataset.
    
    :ivar int samples: Number of samples.
    
    :ivar int variants: Number of variants.
    
    :ivar float call_rate: Fraction of all genotypes called.
    
    :ivar contigs: Unique contigs found in dataset.
    :vartype contigs: list of str 
    
    :ivar int multiallelics: Number of multiallelic variants.
    
    :ivar int snps: Number of SNP alternate alleles.
    
    :ivar int mnps: Number of MNP alternate alleles.

    :ivar int insertions: Number of insertion alternate alleles.
    
    :ivar int deletions: Number of deletion alternate alleles.
    
    :ivar int complex: Number of complex alternate alleles.
    
    :ivar int star: Number of star (upstream deletion) alternate alleles.
    
    :ivar int max_alleles: Highest number of alleles at any variant.
    """

    @classmethod
    def _from_java(cls, jrep):
        summary = Summary.__new__(cls)
        summary.samples = jrep.samples()
        summary.variants = jrep.variants()
        summary.call_rate = jrep.callRate().get() if jrep.callRate().isDefined() else float('nan')
        summary.contigs = [str(x) for x in jiterable_to_list(jrep.contigs())]
        summary.multiallelics = jrep.multiallelics()
        summary.snps = jrep.snps()
        summary.mnps = jrep.mnps()
        summary.insertions = jrep.insertions()
        summary.deletions = jrep.deletions()
        summary.complex = jrep.complex()
        summary.star = jrep.star()
        summary.max_alleles = jrep.maxAlleles()
        return summary

    def __repr__(self):
        return 'Summary(samples=%d, variants=%d, call_rate=%f, contigs=%s, multiallelics=%d, snps=%d, ' \
               'mnps=%d, insertions=%d, deletions=%d, complex=%d, star=%d, max_alleles=%d)' % (
                   self.samples, self.variants, self.call_rate,
                   self.contigs, self.multiallelics, self.snps,
                   self.mnps, self.insertions, self.deletions,
                   self.complex, self.star, self.max_alleles)

    def __str__(self):
        return repr(self)

    def report(self):
        """Print the summary information."""
        print('')  # clear out pesky progress bar if necessary
        print('%16s: %d' % ('Samples', self.samples))
        print('%16s: %d' % ('Variants', self.variants))
        print('%16s: %f' % ('Call Rate', self.call_rate))
        print('%16s: %s' % ('Contigs', self.contigs))
        print('%16s: %d' % ('Multiallelics', self.multiallelics))
        print('%16s: %d' % ('SNPs', self.snps))
        print('%16s: %d' % ('MNPs', self.mnps))
        print('%16s: %d' % ('Insertions', self.insertions))
        print('%16s: %d' % ('Deletions', self.deletions))
        print('%16s: %d' % ('Complex Alleles', self.complex))
        print('%16s: %d' % ('Star Alleles', self.star))
        print('%16s: %d' % ('Max Alleles', self.max_alleles))


class FunctionDocumentation(object):
    @handle_py4j
    def types_rst(self, file_name):
        Env.hail().utils.FunctionDocumentation.makeTypesDocs(file_name)

    @handle_py4j
    def functions_rst(self, file_name):
        Env.hail().utils.FunctionDocumentation.makeFunctionsDocs(file_name)


@handle_py4j
@typecheck(path=strlike,
           buffer_size=integral)
def hadoop_read(path, buffer_size=8192):
    """Open a readable file through the Hadoop filesystem API. 
    Supports distributed file systems like hdfs, gs, and s3.
    
    **Examples**
    
    .. doctest::
        :options: +SKIP

        >>> with hadoop_read('gs://my-bucket/notes.txt') as f:
        ...     for line in f:
        ...         print(line.strip())
    
    **Notes**
    
    The provided source file path must be a URI (uniform resource identifier).

    .. caution::
    
        These file handles are slower than standard Python file handles.
        If you are reading a file larger than ~50M, it will be faster to 
        use :py:meth:`~hail.hadoop_copy` to copy the file locally, then read it
        with standard Python I/O tools.
    
    :param str path: Source file URI.
    
    :param int buffer_size: Size of internal buffer.
    
    :return: Iterable file reader.
    :rtype: `io.BufferedReader <https://docs.python.org/2/library/io.html#io.BufferedReader>`_
    """
    if not isinstance(path, str) and not isinstance(path, unicode):
        raise TypeError("expected parameter 'path' to be type str, but found %s" % type(path))
    if not isinstance(buffer_size, int):
        raise TypeError("expected parameter 'buffer_size' to be type int, but found %s" % type(buffer_size))
    return io.BufferedReader(HadoopReader(path), buffer_size=buffer_size)


@handle_py4j
@typecheck(path=strlike,
           buffer_size=integral)
def hadoop_write(path, buffer_size=8192):
    """Open a writable file through the Hadoop filesystem API. 
    Supports distributed file systems like hdfs, gs, and s3.
    
    **Examples**
    
    .. doctest::
        :options: +SKIP

        >>> with hadoop_write('gs://my-bucket/notes.txt') as f:
        ...     f.write('result1: %s\\n' % result1)
        ...     f.write('result2: %s\\n' % result2)
    
    **Notes**
    
    The provided destination file path must be a URI (uniform resource identifier).

    .. caution::
    
        These file handles are slower than standard Python file handles. If you
        are writing a large file (larger than ~50M), it will be faster to write
        to a local file using standard Python I/O and use :py:meth:`~hail.hadoop_copy` 
        to move your file to a distributed file system.
    
    :param str path: Destination file URI.
    
    :return: File writer object.
    :rtype: `io.BufferedWriter <https://docs.python.org/2/library/io.html#io.BufferedWriter>`_
    """
    if not isinstance(path, str) and not isinstance(path, unicode):
        raise TypeError("expected parameter 'path' to be type str, but found %s" % type(path))
    if not isinstance(buffer_size, int):
        raise TypeError("expected parameter 'buffer_size' to be type int, but found %s" % type(buffer_size))
    return io.BufferedWriter(HadoopWriter(path), buffer_size=buffer_size)


@handle_py4j
@typecheck(src=strlike,
           dest=strlike)
def hadoop_copy(src, dest):
    """Copy a file through the Hadoop filesystem API.
    Supports distributed file systems like hdfs, gs, and s3.
    
    **Examples**
    
    >>> hadoop_copy('gs://hail-common/LCR.interval_list', 'file:///mnt/data/LCR.interval_list') # doctest: +SKIP
    
    **Notes**
    
    The provided source and destination file paths must be URIs
    (uniform resource identifiers).    
    
    :param str src: Source file URI. 
    :param str dest: Destination file URI.
    """
    Env.jutils().copyFile(src, dest, Env.hc()._jhc)


class HadoopReader(io.RawIOBase):
    def __init__(self, path):
        self._jfile = Env.jutils().readFile(path, Env.hc()._jhc)
        super(HadoopReader, self).__init__()

    def close(self):
        self._jfile.close()

    def readable(self):
        return True

    def readinto(self, b):
        b_from_java = self._jfile.read(len(b)).encode('iso-8859-1')
        n_read = len(b_from_java)
        b[:n_read] = b_from_java
        return n_read


class HadoopWriter(io.RawIOBase):
    def __init__(self, path):
        self._jfile = Env.jutils().writeFile(path, Env.hc()._jhc)
        super(HadoopWriter, self).__init__()

    def writable(self):
        return True

    def close(self):
        self._jfile.close()

    def flush(self):
        self._jfile.flush()

    def write(self, b):
        self._jfile.write(bytearray(b))
        return len(b)


def wrap_to_list(s):
    if isinstance(s, list):
        return s
    else:
        return [s]
