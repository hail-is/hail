from hail.genetics.pedigree import Pedigree
from hail.typecheck import *
from hail.utils.java import handle_py4j
from hail.api2 import MatrixTable


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

    >>> pedigree = Pedigree.read('data/myStudy.fam')
    >>> trio_dataset = trio_matrix(dataset, pedigree, complete_trios=True)

    **Notes**

    This method builds a new matrix table with one column per trio. If ``complete_trios`` is true,
    then only trios that satisfy :py:meth:`~hail.representation.Trio.is_complete`
    are included. In this new dataset, the column identifiers
    are the sample IDs of the trio probands. The column annotations and
    entries of the matrix are changed in the following ways:

    The new column fields are:

     - **proband.id** (*String*) - Proband sample ID, same as trio column key.
     - **proband.annotations** (*Struct*) - Annotations on the proband.
     - **father.id** (*String*) - Father sample ID.
     - **father.annotations** (*Struct*) - Annotations on the father.
     - **mother.id** (*String*) - Mother sample ID.
     - **mother.annotations** (*Struct*) - Annotations on the mother.
     - **isFemale** (*Boolean*) - Proband is female. True for female, False for male, missing if unknown.
     - **famID** (*String*) - Family identifier.

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
