from hail.java import Env, handle_py4j


class TextTableConfig(object):
    """Configuration for delimited (text table) files.

    :param bool noheader: File has no header and columns the N columns are named ``_1``, ``_2``, ... ``_N`` (0-indexed)
    :param bool impute: Impute column types from the file
    :param comment: Skip lines beginning with the given pattern
    :type comment: str or None
    :param str delimiter: Field delimiter regex
    :param str missing: Specify identifier to be treated as missing
    :param types: Define types of fields in annotations files   
    :type types: str or None

    :ivar bool noheader: File has no header and columns the N columns are named ``_1``, ``_2``, ... ``_N`` (0-indexed)
    :ivar bool impute: Impute column types from the file
    :ivar comment: Skip lines beginning with the given pattern
    :vartype comment: str or None
    :ivar str delimiter: Field delimiter regex
    :ivar str missing: Specify identifier to be treated as missing
    :ivar types: Define types of fields in annotations files
    :vartype types: str or None
    """

    def __init__(self, noheader=False, impute=False,
                 comment=None, delimiter="\\t", missing="NA", types=None):
        self.noheader = noheader
        self.impute = impute
        self.comment = comment
        self.delimiter = delimiter
        self.missing = missing
        self.types = types

    def __str__(self):
        return self._to_java().toString()

    @handle_py4j
    def _to_java(self):
        """Convert to Java TextTableConfiguration object."""
        return Env.hail().utils.TextTableConfiguration.apply(self.types, self.comment,
                                                             self.delimiter, self.missing,
                                                             self.noheader, self.impute)


class Summary(object):
    """Class holding summary statistics about a dataset.
    
    :ivar int samples: Number of samples.
    
    :ivar int variants: Number of variants.
    
    :ivar float call_rate: Fraction of all genotypes called.
    
    :ivar contigs: Unique contigs found in dataset.
    :vartype contigs: list of str 
    
    :ivar int multiallelics: Number of multiallelic variants.
    
    :ivar int snps: Number of SNP alternate alleles.

    :ivar int indels: Number of insertion / deletion alternate alleles.
    
    :ivar int complex: Number of complex (star, MNP, etc) alternate alleles.
    
    :ivar int most_alleles: Highest number of alleles at any variant.
    """

    def __init__(self, samples, variants, call_rate, contigs, multiallelics, snps, indels, complex, most_alleles):
        self.samples = samples
        self.variants = variants
        self.call_rate = call_rate
        self.contigs = contigs
        self.multiallelics = multiallelics
        self.snps = snps
        self.indels = indels
        self.complex = complex
        self.most_alleles = most_alleles

    def __repr__(self):
        return 'Summary(samples=%d, variants=%d, call_rate=%f, contigs=%s, multiallelics=%d, snps=%d, indels=%d, ' \
               'complex=%d, most_alleles=%d)' % (
                   self.samples, self.variants, self.call_rate,
                   self.contigs, self.multiallelics, self.snps,
                   self.indels, self.complex, self.most_alleles)

    def __str__(self):
        return repr(self)

    def report(self):
        """Print the summary information."""
        print('')  # clear out pesky progress bar if necessary
        print('%14s: %d' % ('Samples', self.samples))
        print('%14s: %d' % ('Variants', self.variants))
        print('%14s: %f' % ('Call rate', self.call_rate))
        print('%14s: %s' % ('Contigs', self.contigs))
        print('%14s: %d' % ('Multiallelics', self.multiallelics))
        print('%14s: %d' % ('SNPs', self.snps))
        print('%14s: %d' % ('Indels', self.indels))
        print('%14s: %d' % ('Complex', self.complex))
        print('%14s: %d' % ('Most alleles', self.most_alleles))


class FunctionDocumentation(object):
    @handle_py4j
    def types_rst(self, file_name):
        Env.hail().utils.FunctionDocumentation.makeTypesDocs(file_name)

    @handle_py4j
    def functions_rst(self, file_name):
        Env.hail().utils.FunctionDocumentation.makeFunctionsDocs(file_name)
