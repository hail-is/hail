from hail.utils.java import *

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

    def __init__(self):
        raise NotImplementedError()

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