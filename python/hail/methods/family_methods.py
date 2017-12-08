from hail.genetics.pedigree import Pedigree
from hail.typecheck import *
from hail.utils.java import handle_py4j
from hail.api2 import MatrixTable

@handle_py4j
@typecheck_method(dataset=MatrixTable, # fix this
                  pedigree=Pedigree,
                  complete_trios=bool)
def trio_matrix(dataset, pedigree, complete_trios=False):
    """Builds and returns a matrix where columns correspond to trios and entries contain genotypes for the trio.

    **Examples**

    Create a trio matrix:

    .. testsetup ::
        dataset = vds1.to_hail2()
        from hail2.genetics import sample_qc

    >>> pedigree = Pedigree.read('data/myStudy.fam')
    >>> trio_matrix = trio_matrix(dataset, pedigree, complete_trios=True)

    **Notes**

    This method builds a new matrix table with one column per trio. If ``complete_trios`` is true,
    then only trios that satisfy :py:meth:`~hail.representation.Trio.is_complete`
    are included. In this new dataset, the column identifiers
    are the sample IDs of the trio probands. The column annotations and
    entries of the matrix are changed in the following ways:

    The new column annotation schema is a ``Struct`` with five children
    (structs ``proband``, ``father``, and ``mother``, boolean ``isFemale``, and string ``famID``).
    The schema of each ``annotations`` field is the column annotation schema of the input dataset.

     - **sa.proband.id** (*String*) - Proband sample ID, same as trio column key.
     - **sa.proband.annotations** (*Struct*) - Annotations on the proband.
     - **sa.father.id** (*String*) - Father sample ID.
     - **sa.father.annotations** (*Struct*) - Annotations on the father.
     - **sa.mother.id** (*String*) - Mother sample ID.
     - **sa.mother.annotations** (*Struct*) - Annotations on the mother.
     - **sa.isFemale** (*Boolean*) - Proband is female. True for female, False for male, missing if unknown.
     - **sa.famID** (*String*) - Family identifier.

    The new entry schema is a ``Struct`` with ``proband``, ``father``, and ``mother``
    fields, where the schema of each field is the entry schema of the input dataset.

    - **g.proband** (*Struct*) - Proband entry field.
    - **g.father** (*Struct*) - Father entry field.
    - **g.mother** (*Struct*) - Mother entry field.

    :param pedigree: Collection of trios.
    :type pedigree: :class:`.hail.representation.Pedigree`

    :rtype: :class:`.VariantDataset`
    """
    return MatrixTable(dataset._, dataset._jvds.trioMatrix(pedigree._jrep, complete_trios))