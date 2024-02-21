from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class SaigePhenotype(Enum):
    CONTINUOUS = 'continuous'
    BINARY = 'binary'


class SaigeTestType(Enum):
    QUANTITATIVE = 'quantitative'
    BINARY = 'binary'


saige_phenotype_to_test_type: Dict[SaigePhenotype, SaigeTestType] = {
    SaigePhenotype.CONTINUOUS: SaigeTestType.QUANTITATIVE,
    SaigePhenotype.BINARY: SaigeTestType.BINARY,
}


@dataclass
class Phenotype:
    """Class used for specifying phenotypes to analyze.

    Notes
    -----

    A phenotype can either be a phenotype to test or a covariate. There are
    two attributes. The name of the phenotype must match the name of a column
    header in a corresponding phenotypes file.

    Examples
    --------

    Specify a binary phenotype ``has_schizophrenia``:

    >>> has_schizophrenia = Phenotype('has_schizophrenia', SaigePhenotype.BINARY)

    Specify a continuous phenotype ``height``:

    >>> height = Phenotype('height', SaigePhenotype.CONTINUOUS)
    """

    name: str
    """The name of the phenotype. This should match a column header name in an input phenotypes file."""

    phenotype_type: SaigePhenotype
    """The type of the phenotype. Used to determine what type of SAIGE test to run for either binary or
    quantitative phenotypes."""


@dataclass
class PhenotypeConfig:
    """Class used for configuring phenotype information.

    Examples
    --------

    Given a phenotypes file with the given format located at ``gs://my-bucket/my-phenotypes.txt``:

    .. code-block:: text

        y_quantitative y_binary x1 x2 IID a1 a2 a3 a4 a5 a6 a7 a8 a9 a10
        2.0046544617651 0 1.51178116845085 1 1a1 0 0 0 0 0 0 0 0 1 0
        0.104213400269085 0 0.389843236411431 1 1a2 0 0 0 0 0 0 0 0 1 1
        -0.397498354133647 0 -0.621240580541804 1 1a3 0 0 0 0 0 0 0 0 0 1
        -0.333177899030597 0 -2.2146998871775 1 1a4 0 0 0 0 0 0 0 0 1 1
        1.21333962248852 0 1.12493091814311 1 1a5 0 0 0 0 0 0 0 0 1 0
        -0.275411643032321 0 -0.0449336090152309 1 1a6 0 0 0 0 0 0 0 0 1 0
        0.438532936074923 0 -0.0161902630989461 0 1a7 0 0 0 0 0 0 0 0 0 0
        0.0162938047248591 0 0.943836210685299 0 1a8 0 0 0 0 0 0 0 0 1 1
        0.147167262428064 0 0.821221195098089 1 1a9 0 0 0 0 0 0 0 0 1 0

    Define a phenotype configuration:

    >>> phenotype_config = PhenotypeConfig("gs://my-bucket/my-phenotypes.txt",
    ...                                    sample_id_col="IID",
    ...                                    phenotypes=[Phenotype("y_binary", SaigePhenotype.BINARY),
    ...                                                Phenotype("y_quantiative", SaigePhenotype.CONTINUOUS)],
    ...                                    covariates=[Phenotype("x1", SaigePhenotype.CONTINUOUS),
    ...                                                Phenotype("x2", SaigePhenotype.BINARY)])

    The helper function saige.extract_phenotypes can be used to convert a Hail Matrix Table into a
    :class:`.PhenotypeConfig`.
    """
    phenotypes_file: str
    """Path to cloud storage containing a phenotypes file. The file should contain a header with the names
    of each column in the file."""

    sample_id_col: str
    """Name of the column in the ``phenotypes_file`` to treat as the sample ID when running association tests."""

    phenotypes: List[Phenotype]
    """List of phenotype definitions to use as phenotypes for analyses."""

    covariates: List[Phenotype]
    """List of covariate definitions to use as covariates for analyses."""

    sex_col: Optional[str] = None
    """The name of a column that defines sex for use in analyses such as accounting for pseudo-autosomal regions."""

    female_code: Optional[str] = None
    """If the ``sex_col`` is defined, use this code to define females."""

    male_code: Optional[str] = None
    """If the ``sex_col`` is defined, use this code to define males."""
