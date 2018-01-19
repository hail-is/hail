from hail.genetics.pedigree import Pedigree
from hail.typecheck import *
from hail.expr import functions
from hail.expr.types import TDict, TStruct, TInt32, TArray
from hail.utils.java import handle_py4j
from hail.utils import Struct
from hail.api2 import MatrixTable, Table
from .misc import require_biallelic


@handle_py4j
@typecheck(dataset=MatrixTable,
           pedigree=Pedigree,
           complete_trios=bool)
def trio_matrix(dataset, pedigree, complete_trios=False):
    """Builds and returns a matrix where columns correspond to trios and entries contain genotypes for the trio.

    **Examples**

    Create a trio matrix:

    .. testsetup::

        dataset = vds.annotate_samples_expr('sa = drop(sa, qc)').to_hail2()
        from hail.methods import trio_matrix

    >>> pedigree = Pedigree.read('data/case_control_study.fam')
    >>> trio_dataset = trio_matrix(dataset, pedigree, complete_trios=True)

    **Notes**

    This method builds a new matrix table with one column per trio. If ``complete_trios`` is true,
    then only trios that satisfy :meth:`~hail.representation.Trio.is_complete`
    are included. In this new dataset, the column identifiers
    are the sample IDs of the trio probands. The column fields and
    entries of the matrix are changed in the following ways:

    The new column fields are:

     - **proband.id** (*String*) - Proband sample ID, same as trio column key.
     - **proband.fields** (*Struct*) - Fields on the proband.
     - **father.id** (*String*) - Father sample ID.
     - **father.fields** (*Struct*) - Fields on the father.
     - **mother.id** (*String*) - Mother sample ID.
     - **mother.fields** (*Struct*) - Fields on the mother.
     - **is_female** (*Boolean*) - Proband is female. True for female, false for male, missing if unknown.
     - **fam_id** (*String*) - Family identifier.

    The new entry fields are:

    - **proband_entry** (*Struct*) - Proband entry fields.
    - **father_entry** (*Struct*) - Father entry fields.
    - **mother_entry** (*Struct*) - Mother entry fields.

    :param pedigree: Collection of trios.
    :type pedigree: :class:`.hail.representation.Pedigree`

    :rtype: :class:`.VariantDataset`
    """
    return MatrixTable(dataset._jvds.trioMatrix(pedigree._jrep, complete_trios)
                       .annotateGenotypesExpr('g = {proband_entry: g.proband, father_entry: g.father, mother_entry: g.mother}'))

@handle_py4j
@typecheck(dataset=MatrixTable,
           pedigree=Pedigree)
def mendel_errors(dataset, pedigree):
    """Find Mendel errors; count per variant, individual and nuclear family.

    .. include:: ../_templates/req_tvariant.rst
    .. include:: ../_templates/req_tstring.rst
    .. include:: ../_templates/req_biallelic.rst

    Examples
    --------

    Find all violations of Mendelian inheritance in each (dad, mom, kid) trio in
    a pedigree and return four tables (all errors, errors by family, errors by
    individual, errors by variant):

    >>> ped = Pedigree.read('data/trios.fam')
    >>> all, per_fam, per_sample, per_variant = methods.mendel_errors(dataset, ped)

    Export all mendel errors to a text file:

    >>> all.export('output/all_mendel_errors.tsv')

    Annotate columns with the number of Mendel errors:

    >>> annotated_samples = dataset.annotate_cols(mendel=per_sample[dataset.id])

    Annotate rows with the number of Mendel errors:

    >>> annotated_variants = dataset.annotate_rows(mendel=per_variant[dataset.v])

    Notes
    -----

    The example above returns four tables, which contain Mendelian violations
    grouped in various ways. These tables are modeled after the `PLINK mendel
    formats <https://www.cog-genomics.org/plink2/formats#mendel>`_, resembling
    the ``.mendel``, ``.fmendel``, ``.imendel``, and ``.lmendel`` formats, respectively.

    **First table:** all Mendel errors. This table contains one row per Mendel
    error, keyed by the variant and proband id.

    Columns:

        - **fam_id** (*String*) -- Family ID.
        - **id** (*String*) -- Proband ID. (key column)
        - **v** (*Variant*) -- Variant in which the error was found. (key column)
        - **code** (*Int32*) -- Mendel error code, see below.
        - **error** (*String*) -- Readable representation of Mendel error.

    **Second table:** errors per nuclear family. This table contains one row
    per nuclear family, keyed by the parents.

    Columns:

        - **fam_id** (*String*) -- Family ID.
        - **pat_id** (*String*) -- Paternal ID. (key column)
        - **mat_id** (*String*) -- Maternal ID. (key column)
        - **children** (*Int32*) -- Number of children in this nuclear family.
        - **errors** (*Int32*) -- Number of Mendel errors in this nuclear family.
        - **snp_errors** (*Int32*) -- Number of Mendel errors at SNPs in this nuclear
          family.

    **Third table:** errors per individual. This table contains one row per
    individual. Each error is counted toward the proband, father, and mother
    according to the `Implicated` in the table below.

    Columns:

        - **id** (*String*) -- Sample ID (key column).
        - **fam_id** (*String*) -- Family ID.
        - **errors** (*Int64*) -- Number of Mendel errors involving this
          individual.
        - **snp_errors** (*Int64*) -- Number of Mendel errors involving this
          individual at SNPs.

    **Fourth table:** errors per variant.

    Columns:

        - **v** (*Variant*) -- Variant (key column).
        - **errors** (*Int32*) -- Number of Mendel errors in this variant.

    The code of each Mendel error is determined by the table below, extending
    the `Plink classification <https://www.cog-genomics.org/plink2/basic_stats#mendel>`__.

    The copy state of a locus with respect to a trio is defined as follows,
    where PAR is the `pseudoautosomal region
    <https://en.wikipedia.org/wiki/Pseudoautosomal_region>`__ (PAR) defined by
    the reference genome.

    - HemiX -- in non-PAR of X, male child
    - HemiY -- in non-PAR of Y, male child
    - Auto -- otherwise (in autosome or PAR, female child)

    `Any` refers to the set \{ HomRef, Het, HomVar, NoCall \} and `~`
    denotes complement in this set.
    
    +------+---------+---------+--------+----------------------------+
    | Code | Dad     | Mom     | Kid    | Copy State | Implicated    |
    +======+=========+=========+========+============+===============+
    |    1 | HomVar  | HomVar  | Het    | Auto       | Dad, Mom, Kid |
    +------+---------+---------+--------+------------+---------------+
    |    2 | HomRef  | HomRef  | Het    | Auto       | Dad, Mom, Kid |
    +------+---------+---------+--------+------------+---------------+
    |    3 | HomRef  | ~HomRef | HomVar | Auto       | Dad, Kid      |
    +------+---------+---------+--------+------------+---------------+
    |    4 | ~HomRef | HomRef  | HomVar | Auto       | Mom, Kid      |
    +------+---------+---------+--------+------------+---------------+
    |    5 | HomRef  | HomRef  | HomVar | Auto       | Kid           |
    +------+---------+---------+--------+------------+---------------+
    |    6 | HomVar  | ~HomVar | HomRef | Auto       | Dad, Kid      |
    +------+---------+---------+--------+------------+---------------+
    |    7 | ~HomVar | HomVar  | HomRef | Auto       | Mom, Kid      |
    +------+---------+---------+--------+------------+---------------+
    |    8 | HomVar  | HomVar  | HomRef | Auto       | Kid           |
    +------+---------+---------+--------+------------+---------------+
    |    9 | Any     | HomVar  | HomRef | HemiX      | Mom, Kid      |
    +------+---------+---------+--------+------------+---------------+
    |   10 | Any     | HomRef  | HomVar | HemiX      | Mom, Kid      |
    +------+---------+---------+--------+------------+---------------+
    |   11 | HomVar  | Any     | HomRef | HemiY      | Dad, Kid      |
    +------+---------+---------+--------+------------+---------------+
    |   12 | HomRef  | Any     | HomVar | HemiY      | Dad, Kid      |
    +------+---------+---------+--------+------------+---------------+

    This meod only considers children with two parents and a defined sex.

    This method assumes all contigs apart from those defined as
    :meth:`~hail.representation.GenomeReference.x_contigs` or
    :meth:`~hail.representation.GenomeReference.y_contigs` by the reference
    genome are fully autosomal. Mitochondria, decoys, etc. are not given
    special treatment.

    Parameters
    ----------
    dataset : :class:`.MatrixTable`
        Dataset.
    pedigree : :class:`.Pedigree`
        Sample pedigree.

    Returns
    -------
    (:class:`.Table`, :class:`.Table`, :class:`.Table`, :class:`.Table`)
        Four tables as detailed in notes with Mendel error statistics.
    """

    kts = require_biallelic(dataset, 'mendel_errors')._jvds.mendelErrors(pedigree._jrep)
    return Table(kts._1()), Table(kts._2()), \
           Table(kts._3()), Table(kts._4())

@handle_py4j
@typecheck(dataset=MatrixTable,
           pedigree=Pedigree)
def tdt(dataset, pedigree):
    """Performs the transmission disequilibrium test on trios.

    .. include:: ../_templates/req_tvariant.rst

    .. include:: ../_templates/req_biallelic.rst

    Examples
    --------
    Compute TDT association results and export to a file:

    >>> pedigree = Pedigree.read('data/trios.fam')
    >>> tdt_table = methods.tdt(dataset, pedigree)
    >>> tdt_table.export('output/tdt_results.tsv')

    Export only variants with p-values below 0.001:

    >>> tdt_table = tdt_table.filter('p < 0.001')
    >>> tdt_table.export("output/tdt_results.tsv")

    Notes
    -----
    The
    `transmission disequilibrium test <https://en.wikipedia.org/wiki/Transmission_disequilibrium_test#The_case_of_trios:_one_affected_child_per_family>`__
    tracks the number of times the alternate allele is transmitted (t) or not
    transmitted (u) from a heterozgyous parent to an affected child under the
    null that the rate of such transmissions is 0.5. For variants where
    transmission is guaranteed (i.e., the Y chromosome, mitochondria, and
    paternal chromosome X variants outside of the PAR), the test is invalid. 

    The TDT statistic is given by

    .. math::

        (t-u)^2 \over (t+u)

    and asymptotically follows a chi-squared distribution with 1 d.o.f.
    under the null hypothesis.

    The number of transmissions and untransmissions for each possible set of
    genotypes is determined from the table below. The copy state of a locus with
    respect to a trio is defined as follows, where PAR is the pseudoautosomal
    region (PAR) as specified in the reference genome. 

    - HemiX -- in non-PAR of X and child is male
    - Auto -- otherwise (in autosome or PAR, or child is female)

    +--------+--------+--------+------------+---+---+
    |  Kid   | Dad    | Mom    | Copy State | T | U |
    +========+========+========+============+===+===+
    | HomRef | Het    | Het    | Auto       | 0 | 2 |
    +--------+--------+--------+------------+---+---+
    | HomRef | HomRef | Het    | Auto       | 0 | 1 |
    +--------+--------+--------+------------+---+---+
    | HomRef | Het    | HomRef | Auto       | 0 | 1 |
    +--------+--------+--------+------------+---+---+
    | Het    | Het    | Het    | Auto       | 1 | 1 |
    +--------+--------+--------+------------+---+---+
    | Het    | HomRef | Het    | Auto       | 1 | 0 |
    +--------+--------+--------+------------+---+---+
    | Het    | Het    | HomRef | Auto       | 1 | 0 |
    +--------+--------+--------+------------+---+---+
    | Het    | HomVar | Het    | Auto       | 0 | 1 |
    +--------+--------+--------+------------+---+---+
    | Het    | Het    | HomVar | Auto       | 0 | 1 |
    +--------+--------+--------+------------+---+---+
    | HomVar | Het    | Het    | Auto       | 2 | 0 |
    +--------+--------+--------+------------+---+---+
    | HomVar | Het    | HomVar | Auto       | 1 | 0 |
    +--------+--------+--------+------------+---+---+
    | HomVar | HomVar | Het    | Auto       | 1 | 0 |
    +--------+--------+--------+------------+---+---+
    | HomRef | HomRef | Het    | HemiX      | 0 | 1 |
    +--------+--------+--------+------------+---+---+
    | HomRef | HomVar | Het    | HemiX      | 0 | 1 |
    +--------+--------+--------+------------+---+---+
    | HomVar | HomRef | Het    | HemiX      | 1 | 0 |
    +--------+--------+--------+------------+---+---+
    | HomVar | HomVar | Het    | HemiX      | 1 | 0 |
    +--------+--------+--------+------------+---+---+

    :func:`tdt` only considers complete trios (two parents and a proband with
    defined sex).

    :func:`tdt` produces a table with the following columns:

     - **v** (*Variant*) -- Variant tested.

     - **transmitted** (*Int32*) -- Number of transmitted alternate alleles.

     - **untransmitted** (*Int32*) -- Number of untransmitted alternate
       alleles.

     - **chi2** (*Float64*) -- TDT statistic.

     - **pval** (*Float64*) -- p-value.

    Parameters
    ----------
    dataset : :class:`.MatrixTable`
        Dataset.
    pedigree : :class:`~hail.genetics.Pedigree`
        Sample pedigree.

    Returns
    -------
    :class:`.Table`
        Table of TDT results.
    """

    hom_ref = 0
    het = 1
    hom_var = 2

    auto = 2
    hemi_X = 1

    t = TDict(TStruct(['kid', 'dad', 'mom', 'copy_state'], [TInt32(), TInt32(), TInt32(), TInt32()]), TArray(TInt32()))

    l = [(hom_ref,     het,     het,   auto, 0, 2),
         (hom_ref, hom_ref,     het,   auto, 0, 1),
         (hom_ref,     het, hom_ref,   auto, 0, 1),
         (    het,     het,     het,   auto, 1, 1),
         (    het, hom_ref,     het,   auto, 1, 0),
         (    het,     het, hom_ref,   auto, 1, 0),
         (    het, hom_var,     het,   auto, 0, 1),
         (    het,     het, hom_var,   auto, 0, 1),
         (hom_var,     het,     het,   auto, 2, 0),
         (hom_var,     het, hom_var,   auto, 1, 0),
         (hom_var, hom_var,     het,   auto, 1, 0),
         (hom_ref, hom_ref,     het, hemi_X, 0, 1),
         (hom_ref, hom_var,     het, hemi_X, 0, 1),
         (hom_var, hom_ref,     het, hemi_X, 1, 0),
         (hom_var, hom_var,     het, hemi_X, 1, 0)]

    mapping = {Struct(**{'kid': v[0], 'dad': v[1], 'mom': v[2], 'copy_state': v[3]}): [v[4], v[5]] for v in l}

    tri = trio_matrix(dataset, pedigree, complete_trios=True)
    tri = tri.filter_samples(functions.is_defined(tri.is_female))
    tri = tri.annotate_global(mapping = mapping)

    tri = tri.annotate_variants(category =
                                functions.cond((tri.v.isAutosomal() | tri.v.inXPar() | tri.v.inYPar()),
                                                0,
                                                functions.cond(tri.v.inXNonPar(),
                                                               1,
                                                               -1)))

    tri = tri.annotate_variants(
        agg.filter(tri.category != 1 | !tri.father.GT.isHet(), tri)
        
    s = '''
        va.{name} = gs
            .filter(g => va.category != 1 || !g.father.GT.isHet())
            .map(g =>
                let ploidy =
                    if (tri.category == 0) 2
                    else if (tri.category == -1) -1
                    else if (tri.is_female) 2
                    else 1 in
                    tri.mapping.get(
                        {{kid: g.proband.GT.nNonRefAlleles(),
                        dad: g.father.GT.nNonRefAlleles(),
                        mom: g.mother.GT.nNonRefAlleles(),
                        copy_state: ploidy}}
                    )[{index}])
            .sum()'''

    tdt_table = (tri
            .annotate_variants([s.format(name='t', index=0), s.format(name='u', index=1)])
            .variants_table())
    
    tdt_table = tdt_table
            .annotate(transmitted = tdt_table.t, untransmitted = tdt_table.u)
            .select(['v', 'transmitted', 'untransmitted'])
            .annotate(chi2 = functions.cond(transmitted + untransmitted > 0,
                                            ((transmitted - untransmitted) ** 2) / (transmitted + untransmitted) '
                      'else 0.0')
            .annotate('p = pchisqtail(chi2, 1.0)')
    )
    return tdt_results