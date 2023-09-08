import abc
from dataclasses import dataclass, field
import jinja2
from typing import Dict, List, Optional, Union
import yaml

import hail as hl

from .constants import SaigeInputDataType


def rectify_defaults(job_config: 'JobConfig', default_config: 'BaseConfig'):
    if job_config.cpu is None:
        job_config.cpu = default_config.cpu
    if job_config.memory is None:
        job_config.memory = default_config.memory
    if job_config.storage is None:
        job_config.storage = default_config.storage
    if job_config.spot is None:
        job_config.spot = default_config.spot
    if job_config.use_checkpoints is None:
        job_config.use_checkpoints = default_config.use_checkpoints
    if job_config.checkpoint_output is None:
        job_config.checkpoint_output = default_config.checkpoint_output


class BaseConfig(abc.ABC):
    image: Optional[str] = 'wzhou88/saige:0.45'
    cpu: Optional[Union[str, int]] = None
    memory: Optional[str] = None
    storage: Optional[str] = None
    spot: Optional[bool] = None
    use_checkpoints: Optional[bool] = None
    checkpoint_output: Optional[bool] = None
    overwrite: Optional[bool] = None


class JobConfig(abc.ABC):
    pass


class JobMetadataMixin(abc.ABC):
    base_name: Optional[str] = None
    base_attrs: Optional[Dict[str, str]] = None

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
class SparseGRMConfig(JobConfig, BaseConfig, JobMetadataMixin):
    base_name = 'sparse-grm'
    relatedness_cutoff: float = ...
    num_markers: int = ...
    cpu: Union[str, float, int] = 1
    memory: str = 'highmem'
    storage: str = '10Gi'


@dataclass
class NullGlmmModelConfig(JobConfig, BaseConfig, JobMetadataMixin):
    user_id_col: str
    base_name: str = '{{ output_dir }}/null-model-{{ phenotype }}'
    inv_normalize: bool = True
    skip_model_fitting: bool = True
    min_covariate_count: int = 5
    base_output_root: str = 'null-model-{{ phenotype }}'

    def output_root(self, output_dir: str, phenotype: str):
        template = jinja2.Environment(loader=jinja2.BaseLoader()).from_string(self.base_output_root)
        return template.render(output_dir=output_dir, phenotype=phenotype)


@dataclass
class RunSaigeConfig(JobConfig, BaseConfig, JobMetadataMixin):
    base_name: Optional[str] = 'run-saige-{{ phenotype }}-{{ chunk }}'
    base_output_root: str = '{{ output_dir }}/run-saige-{{ phenotype }}-{{ chunk }}'
    mkl_off: bool = False
    input_data_type: SaigeInputDataType = field(default_factory=SaigeInputDataType)
    drop_missing_dosages: bool = ...
    min_mac: float = 0.5
    min_maf: float = 0
    max_maf_for_group_test: float = 0.5
    min_info: float = 0
    num_lines_output: int = 10000
    is_sparse: bool = True
    spa_cutoff: float = 2.0
    output_af_in_case_control: bool = False
    output_n_in_case_control: bool = False
    output_het_hom_counts: bool = False
    kernel: Optional[str] = ...
    method: Optional[str] = ...
    weights_beta_rare: Optional[float] = ...
    weights_beta_common: Optional[float] = ...
    weight_maf_cutoff: Optional[float] = ...
    r_corr: Optional[float] = ...
    single_variant_in_group_test: bool = False
    output_maf_in_case_control_in_group_test: bool = False
    cate_var_ratio_min_mac_vec_exclude: Optional[List[float]] = None
    cate_var_ratio_max_mac_vec_include: Optional[List[float]] = None
    dosage_zerod_cutoff: float = 0.2
    output_pvalue_na_in_group_test_for_binary: bool = False
    account_for_case_control_imbalance_in_group_test: bool = True
    weights_include_in_group_file: bool = False  # fixme with weight
    weights_for_g2_cond: Optional[List[int]] = None
    output_beta_se_in_burden_test: bool = False
    output_logp_for_single: bool = False
    x_par_region: Optional[List[str]] = None
    rewrite_x_nonpar_for_males: bool = False
    method_to_collapse_ultra_rare: str = 'absence_or_presence'
    mac_cutoff_to_collapse_ultra_rare: float = 10
    dosage_cutoff_for_ultra_rare_presence: float = 0.5

    def output_root(self, output_dir: str, phenotype: str, chunk: str):
        template = jinja2.Environment(loader=jinja2.BaseLoader()).from_string(self.base_output_root)
        return template.render(output_dir=output_dir, phenotype=phenotype, chunk=chunk)


@dataclass
class SaigeConfig(BaseConfig):
    output_dir: str
    sparse_grm_config: Optional[SparseGRMConfig] = None
    null_glmm_config: NullGlmmModelConfig = field(default_factory=NullGlmmModelConfig, kw_only=True)
    run_saige_config: RunSaigeConfig = field(default_factory=RunSaigeConfig, kw_only=True)
    _phenotype_file: Optional[str] = None
    _input_plink_data: Optional[str] = None
    _grouped_mt: Optional[str] = None
    _sparse_grm: Optional[str] = None
    _null_model_dir: Optional[str] = None
    _ann_genomic_data: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.sparse_grm_config, dict):
            self.sparse_grm_config = SparseGRMConfig(**self.sparse_grm_config)
            rectify_defaults(self.sparse_grm_config, self)
        if isinstance(self.null_glmm_config, dict):
            self.null_glmm_config = NullGlmmModelConfig(**self.null_glmm_config)
            rectify_defaults(self.null_glmm_config, self)
        if isinstance(self.run_saige_config, dict):
            self.run_saige_config = RunSaigeConfig(**self.run_saige_config)
            rectify_defaults(self.run_saige_config, self)

        self._temp_pheno_file = hl.TemporaryFilename(suffix='.pheno')
        self._temp_input_plink_data = hl.TemporaryFilename()
        self._temp_grouped_mt = hl.TemporaryDirectory()
        self._temp_sparse_grm = hl.TemporaryDirectory()
        self._temp_null_model_dir = hl.TemporaryDirectory()
        self._temp_ann_genomic_data = hl.TemporaryDirectory()

    @staticmethod
    def from_yaml_file(file: str) -> 'SaigeConfig':
        with open(file, 'r') as f:
            config = yaml.safe_load(f)
        return SaigeConfig(**config)

    @property
    def pheno_file(self):
        return self._phenotype_file or self._temp_pheno_file.name

    @property
    def input_plink_data(self):
        return self._input_plink_data or self._temp_input_plink_data.name

    @property
    def sparse_grm(self):
        return self._sparse_grm or self._temp_sparse_grm.name

    @property
    def grouped_mt(self):
        return self._grouped_mt or self._temp_grouped_mt

    @property
    def null_model_dir(self):
        return self._null_model_dir or self._temp_null_model_dir

    @property
    def ann_genomic_data(self):
        return self._ann_genomic_data or self._temp_ann_genomic_data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self._temp_pheno_file.__exit__(exc_type, exc_val, exc_tb)
        finally:
            try:
                self._temp_input_plink_data.__exit__(exc_type, exc_val, exc_tb)
            finally:
                try:
                    self._temp_grouped_mt.__exit__(exc_type, exc_val, exc_tb)
                finally:
                    try:
                        self._temp_sparse_grm.__exit__(exc_type, exc_val, exc_tb)
                    finally:
                        try:
                            self._temp_null_model_dir.__exit__(exc_type, exc_val, exc_tb)
                        finally:
                            self._temp_ann_genomic_data.__exit__(exc_type, exc_val, exc_tb)
