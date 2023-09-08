import abc
import jinja2
import yaml

from typing import Dict, Optional, Union

from dataclasses import dataclass, field


class JobConfig(abc.ABC):
    base_name: Optional[str] = None
    base_attrs: Optional[Dict[str, str]] = None
    cpu: Optional[Union[str, int]] = None
    memory: Optional[str] = None
    storage: Optional[str] = None
    spot: Optional[bool] = None
    image: Optional[str] = 'wzhou88/saige:0.45'
    use_checkpoints: Optional[bool] = None
    checkpoint_output: Optional[bool] = None

    def name(self, *, name=None, **kwargs):
        if name is not None:
            return name
        if self.base_name is not None:
            template = jinja2.Environment(loader=jinja2.BaseLoader()).from_string(self.base_name)
            return template.render(**kwargs)
        return None

    def attributes(self, **kwargs):
        attrs = self.base_attrs
        attrs.update(kwargs)
        return attrs


@dataclass
class SparseGRMConfig(JobConfig):
    base_name = 'sparse-grm'
    relatedness_cutoff: float = ...
    num_markers: int = ...
    cpu: Union[str, float, int] = 1
    memory: str = 'highmem'
    storage: str = '10Gi'


@dataclass
class NullGlmmModelConfig(JobConfig):
    user_id_col: str
    base_name: Optional[str] = 'null-model-{{ phenotype }}'
    inv_normalize: bool = True
    skip_model_fitting: bool = True
    min_covariate_count: int = 5
    cpu: Union[str, float, int] = 1
    memory: str = 'highmem'
    storage: str = '10Gi'


@dataclass
class RunSaigeConfig:
    config: 'SaigeConfig'
    min_mac: int = 1
    min_maf: float = 0
    max_maf: float = 0.5
    mkl_off: bool = False
    cpu: Union[str, float, int] = 1
    memory: str = 'standard'
    storage: str = '10Gi'


@dataclass
class SaigeConfig:
    sparse_grm_config: Optional[SparseGRMConfig] = None
    null_glmm_config: NullGlmmModelConfig = field(default_factory=NullGlmmModelConfig, kw_only=True)
    run_saige_config: RunSaigeConfig = field(default_factory=RunSaigeConfig, kw_only=True)

    def __post_init__(self):
        if isinstance(self.sparse_grm_config, dict):
            self.sparse_grm_config = SparseGRMConfig(**self.sparse_grm_config)
        if isinstance(self.null_glmm_config, dict):
            self.null_glmm_config = NullGlmmModelConfig(**self.null_glmm_config)
        if isinstance(self.run_saige_config, dict):
            self.run_saige_config = RunSaigeConfig(**self.run_saige_config)

    @staticmethod
    def from_yaml_file(file: str) -> 'SaigeConfig':
        with open(file, 'r') as f:
            config = yaml.safe_load(f)
        return SaigeConfig(**config)
