from dataclasses import dataclass
from typing import List

from .constants import SaigePhenotype


@dataclass
class Phenotype:
    name: str
    phenotype_type: SaigePhenotype


class PhenotypeInformation:
    def __init__(self, phenotypes: List[Phenotype], covariates: List[Phenotype], sample_id_col: str):
        self.phenotypes = phenotypes
        self.covariates = covariates
        self.sample_id_col = sample_id_col
