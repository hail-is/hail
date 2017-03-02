.. _sec-objects:

============
Hail Objects
============

Hail represents many concepts in genetics as objects with methods for use in the `Expression Language <https://hail.is/expr_lang.html>`_.

.. _variant:

-------
Variant
-------

**Variable Name:** ``v``

- **v.contig** (*String*) -- String representation of contig, exactly as imported.  *NB: Hail stores contigs as strings.  Use double-quotes when checking contig equality*
- **v.start** (*Int*) -- SNP position or start of an indel
- **v.ref** (*String*) -- Reference allele sequence
- **v.isBiallelic** (*Boolean*) -- True if `v` has one alternate allele
- **v.nAlleles** (*Int*) -- Number of alleles
- **v.nAltAlleles** (*Int*) -- Number of alternate alleles, equal to ``nAlleles - 1``
- **v.nGenotypes** (*Int*) -- Number of genotypes
- **v.altAlleles** (*Array[AltAllele]*) -- The :ref:`alternate alleles <altallele>`
- **v.altAllele** (*AltAllele*) -- The :ref:`alternate allele <altallele>`.  **Assumes biallelic.**
- **v.alt** (*String*) -- Alternate allele sequence.  **Assumes biallelic.**
- **v.locus** (*Locus*) -- Chromosomal locus (chr, pos) of this variant
- **v.isAutosomal** (*Boolean*) -- True if chromosome is not X, not Y, and not MT
- **v.inXPar** (*Boolean*) -- True if chromosome is X and start is in pseudoautosomal region of X
- **v.inYPar** (*Boolean*) -- True if chromosome is Y and start is in pseudoautosomal region of Y.
- **v.inXNonPar** (*Boolean*) -- True if chromosome is X and start is not in pseudoautosomal region of X
- **v.inYNonPar** (*Boolean*) -- True if chromosome is Y and start is not in pseudoautosomal region of Y

The `pseudoautosomal region <https://en.wikipedia.org/wiki/Pseudoautosomal_region>`_ (PAR) is currently defined with respect to reference `GRCh37 <http://www.ncbi.nlm.nih.gov/projects/genome/assembly/grc/human/>`_:

- X: 60001 - 2699520, 154931044 - 155260560
- Y: 10001 - 2649520, 59034050 - 59363566

Most callers assign variants in PAR to X.

.. _altallele:

---------
AltAllele
---------

**Variable Name:** ``v.altAlleles[idx]`` or ``v.altAllele`` (biallelic)

- **<altAllele>.ref** (*String*) -- reference allele base sequence
- **<altAllele>.alt** (*String*)  -- alternate allele base sequence
- **<altAllele>.isSNP** (*Boolean*) -- true if ``v.ref`` and ``v.alt`` are the same length and differ in one position
- **<altAllele>.isMNP** (*Boolean*) -- true if ``v.ref`` and ``v.alt`` are the same length and differ in more than one position
- **<altAllele>.isIndel** (*Boolean*) -- true if an insertion or a deletion
- **<altAllele>.isInsertion** (*Boolean*) -- true if ``v.alt`` begins with and is longer than ``v.ref``
- **<altAllele>.isDeletion** (*Boolean*) -- true if ``v.ref`` begins with and is longer than ``v.alt``
- **<altAllele>.isComplex** (*Boolean*) -- true if not a SNP, MNP, or indel
- **<altAllele>.isTransition** (*Boolean*) -- true if a purine-purine or pyrimidine-pyrimidine SNP
- **<altAllele>.isTransversion** (*Boolean*) -- true if a purine-pyrimidine SNP

.. _locus:

-----
Locus
-----

**Variable Name:** ``v.locus`` or ``Locus(chr, pos)``

- **<locus>.contig** (*String*) -- String representation of contig
- **<locus>.position** (*Int*) -- Chromosomal position

.. _interval:

--------
Interval
--------

**Variable Name:** ``Interval(locus1, locus2)``

- **<interval>.start** (*Locus*) -- :ref:`locus` at the start of the interval (inclusive)
- **<interval>.end** (*Locus*) -- :ref:`locus` at the end of the interval (exclusive)

.. _sample:

------
Sample
------

**Variable Name:** ``s``

- **s.id** (*String*) -- The ID of this sample, as read at import-time

.. _genotype:

--------
Genotype
--------

**Variable Name:** ``g``

- **g.gt** (*Int*) -- the call, ``gt = k\*(k+1)/2+j`` for call ``j/k``
- **g.ad** (*Array[Int]*) -- allelic depth for each allele
- **g.dp** (*Int*) -- the total number of informative reads
- **g.od** (*Int*) -- ``od = dp - ad.sum``
- **g.gq** (*Int*) -- the difference between the two smallest PL entries
- **g.pl** (*Array[Int]*) -- phred-scaled normalized genotype likelihood values
- **g.dosage** (*Array[Double]*) -- the linear-scaled probabilities
- **g.isHomRef** (*Boolean*) -- true if this call is ``0/0``
- **g.isHet** (*Boolean*) -- true if this call is heterozygous
- **g.isHetRef** (*Boolean*) -- true if this call is ``0/k`` with ``k>0``
- **g.isHetNonRef** (*Boolean*) -- true if this call is ``j/k`` with ``j>0``
- **g.isHomVar** (*Boolean*) -- true if this call is ``j/j`` with ``j>0``
- **g.isCalledNonRef** (*Boolean*) -- true if either ``g.isHet`` or ``g.isHomVar`` is true
- **g.isCalled** (*Boolean*) -- true if the genotype is not ``./.``
- **g.isNotCalled** (*Boolean*) -- true if the genotype is ``./.``
- **g.nNonRefAlleles** (*Int*) -- the number of called alternate alleles
- **g.pAB** (*Double*)  -- p-value for pulling the given allelic depth from a binomial distribution with mean 0.5.  Missing if the call is not heterozygous.
- **g.fractionReadsRef** (*Double*) -- the ratio of ref reads to the sum of all *informative* reads
- **g.fakeRef** (*Boolean*) -- true if this genotype was downcoded in :py:meth:`~hail.VariantDataset.split_multi`.  This can happen if a ``1/2`` call is split to ``0/1``, ``0/1``
- **g.isDosage** (*Boolean*) -- true if the data was imported from :py:meth:`~hail.HailContext.import_gen` or :py:meth:`~hail.HailContext.import_bgen`
- **g.oneHotAlleles(Variant)** (*Array[Int]*) -- Produces an array of called counts for each allele in the variant (including reference).  For example, calling this function with a biallelic variant on hom-ref, het, and hom-var genotypes will produce ``[2, 0]``, ``[1, 1]``, and ``[0, 2]`` respectively.
- **g.oneHotGenotype(Variant)** (*Array[Int]*) -- Produces an array with one element for each possible genotype in the variant, where the called genotype is 1 and all else 0.  For example, calling this function with a biallelic variant on hom-ref, het, and hom-var genotypes will produce ``[1, 0, 0]``, ``[0, 1, 0]``, and ``[0, 0, 1]`` respectively.
- **g.gtj** (*Int*) -- the index of allele ``j`` for call ``j/k`` (0 = ref, 1 = first alt allele, etc.)
- **g.gtk** (*Int*) -- the index of allele ``k`` for call ``j/k`` (0 = ref, 1 = first alt allele, etc.)


The conversion between ``g.pl`` (Phred-scaled likelihoods) and ``g.dosage`` (linear-scaled probabilities) assumes a uniform prior.
