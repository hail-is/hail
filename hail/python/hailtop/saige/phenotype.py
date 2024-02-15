from dataclasses import dataclass
from typing import List, Optional

from .constants import SaigePhenotype


@dataclass
class Phenotype:
    name: str
    phenotype_type: SaigePhenotype


class PhenotypeConfig:
    def __init__(self,
                 phenotypes_file: str,
                 phenotypes: List[Phenotype],
                 covariates: List[Phenotype],
                 sample_id_col: str,
                 sex_col: Optional[str],
                 female_code: Optional[str],
                 male_code: Optional[str]):
        self.phenotypes_file = phenotypes_file
        self.phenotypes = phenotypes
        self.covariates = covariates
        self.sample_id_col = sample_id_col
        self.sex_col = sex_col
        self.female_code = female_code
        self.male_code = male_code
