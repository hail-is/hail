from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import hail as hl
from hailtop.utils import grouped

from .constants import SaigePhenotype


@dataclass
class Phenotype:
    name: str
    phenotype_type: SaigePhenotype
    group: str


class Phenotypes:
    @staticmethod
    def from_grouped_str_phenotype_type_mapping(phenotypes: List[Dict[str, SaigePhenotype]]) -> 'Phenotypes':
        new_phenotypes = []
        for idx, group in enumerate(phenotypes):
            group_name = f'group_{idx+1}'
            for phenotype, typ in group.items():
                new_phenotypes.append(Phenotype(phenotype, typ, group_name))
        return Phenotypes(new_phenotypes)

    # @staticmethod
    # def from_matrix_table(mt: hl.MatrixTable, phenotypes: List[str]) -> 'Phenotypes':
    #     new_phenotypes = []
    #     for phenotype in phenotypes:
    #         col = mt[phenotype]
    #         if col.dtype == hl.tbool:
    #             phenotype_type = SaigePhenotype.CATEGORICAL
    #         else:
    #             assert hl.is_numeric(col.dtype)
    #             phenotype_type = SaigePhenotype.CONTINUOUS
    #         p = Phenotype(phenotype, phenotype_type, phenotype)
    #         new_phenotypes.append(p)
    #     return Phenotypes(new_phenotypes)
    #
    # @staticmethod
    # def from_phenotypes_file(
    #     file: str,
    #     *,
    #     max_phenotypes_per_group: int = 20,
    #     default_phenotype_type: SaigePhenotype = SaigePhenotype.CATEGORICAL,
    #     name_col: str = 'name',
    #     phenotype_type_col: Optional[str] = None,
    #     group_col: Optional[str] = None,
    #
    # ) -> 'Phenotypes':
    #     ht = hl.read_table(file, impute=True)
    #     rows = ht.collect()
    #
    #     phenotypes = []
    #     for group_idx, group in enumerate(grouped(max_phenotypes_per_group, rows)):
    #         for row in group:
    #             if phenotype_type_col in row:
    #                 phenotype_type = SaigePhenotype(row[phenotype_type_col])
    #             else:
    #                 phenotype_type = default_phenotype_type
    #
    #             if group_col in row:
    #                 group = row[group_col]
    #             else:
    #                 group = f'group{group_idx}'
    #
    #             phenotypes.append(Phenotype(row[name_col], phenotype_type, str(group)))
    #     return Phenotypes(phenotypes)

    def __init__(self, phenotypes: List[Phenotype]):
        self.phenotypes = phenotypes

        self._grouped_phenotypes = defaultdict(list)
        for phenotype in phenotypes:
            assert phenotype.group is not None
            self._grouped_phenotypes[phenotype.group].append(phenotype)

    @property
    def phenotype_names(self):
        return [p.name for p in self.phenotypes]

    def __iter__(self):
        for group in self._grouped_phenotypes:
            yield self._grouped_phenotypes[group]
