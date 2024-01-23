from typing import Mapping, List, Union, TypeVar, Tuple, Dict, Optional, Any, AbstractSet
import abc
from enum import Enum
from dataclasses import dataclass
import warnings
import orjson
import pkg_resources
import zipfile

from hailtop.config.user_config import unchecked_configuration_of
from hailtop.fs.fs import FS

from ..builtin_references import BUILTIN_REFERENCE_RESOURCE_PATHS
from ..expr import Expression
from ..expr.table_type import ttable
from ..expr.matrix_type import tmatrix
from ..expr.blockmatrix_type import tblockmatrix
from ..expr.types import HailType, dtype, tvoid
from ..ir import BaseIR, finalize_randomness
from ..ir.renderer import CSERenderer
from ..linalg.blockmatrix import BlockMatrix
from ..matrixtable import MatrixTable
from ..table import Table
from ..utils.java import FatalError


Dataset = TypeVar('Dataset', Table, MatrixTable, BlockMatrix)


def fatal_error_from_java_error_triplet(short_message, expanded_message, error_id):
    from .. import __version__

    if error_id != -1:
        return FatalError(f'Error summary: {short_message}', error_id)
    return FatalError(
        f"""{short_message}

Java stack trace:
{expanded_message}
Hail version: {__version__}
Error summary: {short_message}""",
        error_id,
    )


class LocalJarInformation:
    def __init__(self, development_mode: bool, local_jar_path: str, extra_classpath: List[str]):
        self.development_mode = development_mode
        self.path = local_jar_path
        self.extra_classpath = extra_classpath


def resource_exists(f: str) -> bool:
    return pkg_resources.resource_exists(__name__, f)


def resource_filename(f: str) -> str:
    return pkg_resources.resource_filename(__name__, f)


def local_jar_information() -> LocalJarInformation:
    if resource_exists('hail.jar'):
        warnings.warn('!!! THIS IS A DEVELOPMENT VERSION OF HAIL !!!')
        with open(pkg_resources.resource_filename(__name__, 'extra_classpath')) as fobj:
            return LocalJarInformation(
                True,
                resource_filename('hail.jar'),
                [fobj.read()],
            )
    elif resource_exists('hail-all-spark.jar'):
        return LocalJarInformation(
            False,
            resource_filename("hail-all-spark.jar"),
            [],
        )
    else:
        raise ValueError(
            f'Hail requires either {resource_filename("hail.jar")} or {resource_filename("hail-all-spark.jar")}.'
        )


class ActionTag(Enum):
    LOAD_REFERENCES_FROM_DATASET = 1
    VALUE_TYPE = 2
    TABLE_TYPE = 3
    MATRIX_TABLE_TYPE = 4
    BLOCK_MATRIX_TYPE = 5
    EXECUTE = 6
    PARSE_VCF_METADATA = 7
    IMPORT_FAM = 8
    FROM_FASTA_FILE = 9


@dataclass
class ActionPayload:
    pass


@dataclass
class IRTypePayload(ActionPayload):
    ir: str


@dataclass
class ExecutePayload(ActionPayload):
    ir: str
    stream_codec: str
    timed: bool


@dataclass
class LoadReferencesFromDatasetPayload(ActionPayload):
    path: str


@dataclass
class ParseVCFMetadataPayload(ActionPayload):
    path: str


@dataclass
class ImportFamPayload(ActionPayload):
    path: str
    quant_pheno: bool
    delimiter: str
    missing: str


@dataclass
class FromFASTAFilePayload(ActionPayload):
    name: str
    fasta_file: str
    index_file: str
    x_contigs: List[str]
    y_contigs: List[str]
    mt_contigs: List[str]
    par: List[str]


class Backend(abc.ABC):
    # Must match knownFlags in HailFeatureFlags.scala
    _flags_env_vars_and_defaults: Dict[str, Tuple[str, Optional[str]]] = {
        "no_whole_stage_codegen": ("HAIL_DEV_NO_WHOLE_STAGE_CODEGEN", None),
        "no_ir_logging": ("HAIL_DEV_NO_IR_LOG", None),
        "lower": ("HAIL_DEV_LOWER", None),
        "lower_only": ("HAIL_DEV_LOWER_ONLY", None),
        "lower_bm": ("HAIL_DEV_LOWER_BM", None),
        "print_ir_on_worker": ("HAIL_DEV_PRINT_IR_ON_WORKER", None),
        "print_inputs_on_worker": ("HAIL_DEV_PRINT_INPUTS_ON_WORKER", None),
        "max_leader_scans": ("HAIL_DEV_MAX_LEADER_SCANS", "1000"),
        "distributed_scan_comb_op": ("HAIL_DEV_DISTRIBUTED_SCAN_COMB_OP", None),
        "jvm_bytecode_dump": ("HAIL_DEV_JVM_BYTECODE_DUMP", None),
        "write_ir_files": ("HAIL_WRITE_IR_FILES", None),
        "method_split_ir_limit": ("HAIL_DEV_METHOD_SPLIT_LIMIT", "16"),
        "use_new_shuffle": ("HAIL_USE_NEW_SHUFFLE", None),
        "shuffle_max_branch_factor": ("HAIL_SHUFFLE_MAX_BRANCH", "64"),
        "shuffle_cutoff_to_local_sort": ("HAIL_SHUFFLE_CUTOFF", "512000000"),  # This is in bytes
        "grouped_aggregate_buffer_size": ("HAIL_GROUPED_AGGREGATE_BUFFER_SIZE", "50"),
        "use_ssa_logs": ("HAIL_USE_SSA_LOGS", "1"),
        "gcs_requester_pays_project": ("HAIL_GCS_REQUESTER_PAYS_PROJECT", None),
        "gcs_requester_pays_buckets": ("HAIL_GCS_REQUESTER_PAYS_BUCKETS", None),
        "index_branching_factor": ("HAIL_INDEX_BRANCHING_FACTOR", None),
        "rng_nonce": ("HAIL_RNG_NONCE", "0x0"),
        "profile": ("HAIL_PROFILE", None),
        "use_fast_restarts": ("HAIL_USE_FAST_RESTARTS", None),
        "cachedir": ("HAIL_CACHE_DIR", None),
    }

    def _valid_flags(self) -> AbstractSet[str]:
        return self._flags_env_vars_and_defaults.keys()

    @abc.abstractmethod
    def __init__(self):
        self._persisted_locations = dict()
        self._references = {}

    @abc.abstractmethod
    def validate_file(self, uri: str):
        raise NotImplementedError

    @abc.abstractmethod
    def stop(self):
        pass

    def execute(self, ir: BaseIR, timed: bool = False) -> Any:
        payload = ExecutePayload(self._render_ir(ir), '{"name":"StreamBufferSpec"}', timed)
        try:
            result, timings = self._rpc(ActionTag.EXECUTE, payload)
        except FatalError as e:
            raise e.maybe_user_error(ir) from None
        if ir.typ == tvoid:
            value = None
        else:
            value = ir.typ._from_encoding(result)
        return (value, timings) if timed else value

    @abc.abstractmethod
    def _rpc(self, action: ActionTag, payload: ActionPayload) -> Tuple[bytes, str]:
        pass

    def _render_ir(self, ir):
        r = CSERenderer()
        return r(finalize_randomness(ir))

    def value_type(self, ir):
        payload = IRTypePayload(ir=self._render_ir(ir))
        type_bytes, _ = self._rpc(ActionTag.VALUE_TYPE, payload)
        return dtype(type_bytes.decode('utf-8'))

    def table_type(self, tir):
        payload = IRTypePayload(ir=self._render_ir(tir))
        type_bytes, _ = self._rpc(ActionTag.TABLE_TYPE, payload)
        return ttable._from_json(orjson.loads(type_bytes))

    def matrix_type(self, mir):
        payload = IRTypePayload(ir=self._render_ir(mir))
        type_bytes, _ = self._rpc(ActionTag.MATRIX_TABLE_TYPE, payload)
        return tmatrix._from_json(orjson.loads(type_bytes))

    def blockmatrix_type(self, bmir):
        payload = IRTypePayload(ir=self._render_ir(bmir))
        type_bytes, _ = self._rpc(ActionTag.BLOCK_MATRIX_TYPE, payload)
        return tblockmatrix._from_json(orjson.loads(type_bytes))

    def load_references_from_dataset(self, path):
        payload = LoadReferencesFromDatasetPayload(path=path)
        references_json_bytes, _ = self._rpc(ActionTag.LOAD_REFERENCES_FROM_DATASET, payload)
        return orjson.loads(references_json_bytes)

    def from_fasta_file(self, name, fasta_file, index_file, x_contigs, y_contigs, mt_contigs, par):
        payload = FromFASTAFilePayload(name, fasta_file, index_file, x_contigs, y_contigs, mt_contigs, par)
        rg_json_bytes, _ = self._rpc(ActionTag.FROM_FASTA_FILE, payload)
        return orjson.loads(rg_json_bytes)

    def add_reference(self, rg):
        self._references[rg.name] = rg
        self._add_reference_to_scala_backend(rg)

    def _add_reference_to_scala_backend(self, rg):
        pass

    def get_reference(self, name):
        return self._references[name]

    def initialize_references(self):
        from hail.genetics.reference_genome import ReferenceGenome

        jar_path = local_jar_information().path
        for path_in_jar in BUILTIN_REFERENCE_RESOURCE_PATHS.values():
            rg_config = orjson.loads(zipfile.ZipFile(jar_path).open(path_in_jar).read())
            rg = ReferenceGenome._from_config(rg_config, _builtin=True)
            self._references[rg.name] = rg

    def remove_reference(self, name):
        del self._references[name]
        self._remove_reference_from_scala_backend(name)

    def _remove_reference_from_scala_backend(self, name):
        pass

    @abc.abstractmethod
    def add_sequence(self, name, fasta_file, index_file):
        pass

    @abc.abstractmethod
    def remove_sequence(self, name):
        pass

    @abc.abstractmethod
    def add_liftover(self, name, chain_file, dest_reference_genome):
        pass

    @abc.abstractmethod
    def remove_liftover(self, name, dest_reference_genome):
        pass

    def parse_vcf_metadata(self, path):
        payload = ParseVCFMetadataPayload(path)
        metadata_json_bytes, _ = self._rpc(ActionTag.PARSE_VCF_METADATA, payload)
        return orjson.loads(metadata_json_bytes)

    @property
    @abc.abstractmethod
    def logger(self):
        pass

    @property
    @abc.abstractmethod
    def fs(self) -> FS:
        pass

    def import_fam(self, path: str, quant_pheno: bool, delimiter: str, missing: str):
        payload = ImportFamPayload(path, quant_pheno, delimiter, missing)
        fam_json_bytes, _ = self._rpc(ActionTag.IMPORT_FAM, payload)
        return orjson.loads(fam_json_bytes)

    def persist(self, dataset: Dataset) -> Dataset:
        from hail.context import TemporaryFilename

        tempfile = TemporaryFilename(prefix=f'persist_{type(dataset).__name__}')
        persisted = dataset.checkpoint(tempfile.__enter__())
        self._persisted_locations[persisted] = (tempfile, dataset)
        return persisted

    def unpersist(self, dataset: Dataset) -> Dataset:
        tempfile, unpersisted = self._persisted_locations.pop(dataset, (None, None))
        if tempfile is None:
            return dataset
        tempfile.__exit__(None, None, None)
        return unpersisted

    @abc.abstractmethod
    def register_ir_function(
        self,
        name: str,
        type_parameters: Union[Tuple[HailType, ...], List[HailType]],
        value_parameter_names: Union[Tuple[str, ...], List[str]],
        value_parameter_types: Union[Tuple[HailType, ...], List[HailType]],
        return_type: HailType,
        body: Expression,
    ):
        pass

    @abc.abstractmethod
    def _is_registered_ir_function_name(self, name: str) -> bool:
        pass

    @abc.abstractmethod
    def persist_expression(self, expr: Expression) -> Expression:
        pass

    def _initialize_flags(self, initial_flags: Dict[str, str]) -> None:
        self.set_flags(
            **{
                k: unchecked_configuration_of('query', k, None, default, deprecated_envvar=deprecated_envvar)
                for k, (deprecated_envvar, default) in Backend._flags_env_vars_and_defaults.items()
                if k not in initial_flags
            },
            **initial_flags,
        )

    @abc.abstractmethod
    def set_flags(self, **flags: str):
        """Set Hail flags."""
        pass

    @abc.abstractmethod
    def get_flags(self, *flags) -> Mapping[str, str]:
        """Mapping of Hail flags."""
        pass

    @property
    @abc.abstractmethod
    def requires_lowering(self):
        pass
