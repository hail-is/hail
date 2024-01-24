from dataclasses import dataclass

from .constants import SaigePhenotype


@dataclass
class Phenotype:
    name: str
    phenotype_type: SaigePhenotype
    group: str
