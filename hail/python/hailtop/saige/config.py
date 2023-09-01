import abc
from dataclasses import dataclass
from typing import Dict, Optional
import yaml


@dataclass
class SaigeQueryOnBatchConfig(abc.ABC):
    worker_memory: str = 'standard'
    worker_cores: int = 1
    driver_memory: str = 'highmem'
    driver_cores: int = 8


@dataclass
class SaigeMatrixTableDataExtractorConfig(abc.ABC):
    output_root: str
    docker_image: str
    module: str
    gene: Optional[str] = None
    interval: str = None
    groups: Optional[str] = None
    gene_map_ht_path: str = None
    set_missing_to_hom_ref: bool = False
    callrate_filter: float = 0.0
    adj: bool = True
    input_dosage: bool = False
    reference: str = 'GRCh38'
    gene_ht_interval: str = None
    n_threads: int = 8
    storage: str = '500Mi'
    additional_args: str = ''
    memory: str = ''
    attributes: Optional[Dict[str, str]] = None
    name: Optional[str] = None

    @property
    def groups_set(self):
        if self.groups is not None:
            return self.groups.split(',')
        return {'pLoF', 'missense|LC', 'synonymous'}


class SaigeMatrixTableToVcfConfig(SaigeMatrixTableDataExtractorConfig):
    name: str = 'extract_vcf'


class SaigeMatrixTableToBgenConfig(SaigeMatrixTableDataExtractorConfig):
    name: str = 'extract_bgen'


@dataclass
class SaigeNullModelConfig:
    docker_image: str = ...
    pheno_col: str = 'value'
    user_id_col: str = 'userId'
    inv_normalize: bool = False
    skip_model_fitting: bool = False
    min_covariate_count: int = 10
    n_threads: int = 16
    storage: str = '10Gi'
    memory: str = '60G'
    spot: bool = True
    name: str = 'fit_null_model'
    attributes: Dict[str, str] = dict


@dataclass
class SaigeSparseGrmConfig:
    docker_image: str
    relatedness_cutoff: str = '0.125'
    num_markers: int = 2000
    n_threads: int = 8
    storage: str = '1500Mi'
    name: str = 'create_sparse_grm'
    attributes: Dict[str, str] = dict


@dataclass
class SaigeConfig:
    @staticmethod
    def from_file(path: str):
        with open(path, 'r') as f:
            config = yaml.safe_load(f)

        qob_config = None
        sparse_grm_config = None
        null_model_config = None

        name = config.get('name')
        attributes = config.get('attributes')

        if 'qob_config' in config:
            qob_config = SaigeQueryOnBatchConfig(**config['qob_config'])
        if 'sparse_grm_config' in config:
            sparse_grm_config = SaigeSparseGrmConfig(**config['sparse_grm_config'])
        if 'null_model_config' in config:
            null_model_config = SaigeNullModelConfig(**config['null_model_config'])

        return SaigeConfig(
            name=name,
            attributes=attributes,
            qob_config=qob_config,
            sparse_grm_config=sparse_grm_config,
            null_model_config=null_model_config
        )

    def __init__(self,
                 *,
                 name: Optional[str] = None,
                 attributes: Optional[Dict[str, str]] = None,
                 qob_config: Optional[SaigeQueryOnBatchConfig] = None,
                 sparse_grm_config: Optional[SaigeSparseGrmConfig] = None,
                 null_model_config: Optional[SaigeNullModelConfig] = None):
        self.name = name
        self.attributes = attributes
        self.qob_config = qob_config
        self.sparse_grm_config = sparse_grm_config
        self.null_model_config = null_model_config
