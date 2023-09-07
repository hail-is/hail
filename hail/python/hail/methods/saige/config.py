import copy

from typing import Dict, Optional

from dataclasses import dataclass

from .constants import SaigeAnalysisType, SaigePhenotype


@dataclass
class BaseConfig:
    use_checkpoints: bool = False
    default_cpu: Optional[int] = None
    default_memory: Optional[str] = None
    default_spot: bool = True
    default_storage: Optional[str] = None
    checkpoint: bool = False

    @staticmethod
    def from_yaml_file(file: str) -> 'BaseConfig':
        pass


class SparseGRMConfig(BaseConfig):
    name: str = 'sparse-grm'
    relatedness_cutoff: float
    num_markers: int
    cpu = 1
    memory = 'standard'
    spot

    def __post_init__(self):
        self.cpu = self.cpu or self.default_cpu
        self.memory = self.memory or self.default_memory
        self.spot = self.spot or self.default_spot


class NullGlmmModelConfig(BaseConfig):
    name = 'null-model'
    inv_normalize: bool = True
    skip_model_fitting: bool = True
    min_covariate_count: int = 5
    user_id_col: str = ...

    def name_with_pheno(self, pheno: str) -> str:
        return f'{self.name}-{pheno}'

    def attributes_with_pheno(self, pheno: str, analysis_type: SaigeAnalysisType, trait_type: SaigePhenotype) -> Dict[str, str]:
        attributes = self.attributes or {}
        attributes = copy.deepcopy(attributes)
        attributes.update({'pheno': pheno})
        return attributes


class SaigeConfig(BaseConfig):
    min_mac: int = 1
    min_maf: float = 0
    max_maf: float = 0.5
    mkl_off: bool = False
