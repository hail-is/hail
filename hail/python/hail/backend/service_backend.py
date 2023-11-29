from typing import Dict, Optional, Awaitable, Mapping, Any, List, Union, Tuple, TypeVar, Set
import abc
import asyncio
from dataclasses import dataclass
import math
import struct
from hail.expr.expressions.base_expression import Expression
import orjson
import logging
import warnings

from hail.context import TemporaryDirectory, TemporaryFilename, tmp_dir, revision, version
from hail.utils import FatalError
from hail.expr.types import HailType
from hail.experimental import read_expression, write_expression
from hail.ir import finalize_randomness
from hail.ir.renderer import CSERenderer

from hailtop import yamlx
from hailtop.config import ConfigVariable, configuration_of, get_remote_tmpdir
from hailtop.utils import async_to_blocking, Timings, am_i_interactive, retry_transient_errors
from hailtop.utils.rich_progress_bar import BatchProgressBar
from hailtop.batch_client import client as hb
from hailtop.batch_client import aioclient as aiohb
from hailtop.batch.utils import needs_tokens_mounted
from hailtop.aiotools.router_fs import RouterAsyncFS
from hailtop.aiocloud.aiogoogle import GCSRequesterPaysConfiguration, get_gcs_requester_pays_configuration
import hailtop.aiotools.fs as afs
from hailtop.fs.fs import FS
from hailtop.fs.router_fs import RouterFS
from hailtop.aiotools.fs.exceptions import UnexpectedEOFError

from .backend import Backend, fatal_error_from_java_error_triplet, ActionTag, ActionPayload, ExecutePayload
from ..builtin_references import BUILTIN_REFERENCES
from ..utils import ANY_REGION
from hailtop.aiotools.validators import validate_file


ReferenceGenomeConfig = Dict[str, Any]


T = TypeVar("T")


log = logging.getLogger('backend.service_backend')


async def read_byte(strm: afs.ReadableStream) -> int:
    return (await strm.readexactly(1))[0]


async def read_bool(strm: afs.ReadableStream) -> bool:
    return (await read_byte(strm)) != 0


async def read_int(strm: afs.ReadableStream) -> int:
    b = await strm.readexactly(4)
    return struct.unpack('<i', b)[0]


async def read_bytes(strm: afs.ReadableStream) -> bytes:
    n = await read_int(strm)
    return await strm.readexactly(n)


async def read_str(strm: afs.ReadableStream) -> str:
    b = await read_bytes(strm)
    return b.decode('utf-8')


class JarSpec(abc.ABC):
    @abc.abstractmethod
    def to_dict(self) -> Dict[str, str]:
        raise NotImplementedError


class JarUrl(JarSpec):
    def __init__(self, url):
        self.url = url

    def to_dict(self) -> Dict[str, str]:
        return {'type': 'jar_url', 'value': self.url}

    def __repr__(self):
        return f'JarUrl({self.url})'


class GitRevision(JarSpec):
    def __init__(self, revision):
        self.revision = revision

    def to_dict(self) -> Dict[str, str]:
        return {'type': 'git_revision', 'value': self.revision}

    def __repr__(self):
        return f'GitRevision({self.revision})'


@dataclass
class SerializedIRFunction:
    name: str
    type_parameters: List[str]
    value_parameter_names: List[str]
    value_parameter_types: List[str]
    return_type: str
    rendered_body: str


class IRFunction:
    def __init__(self,
                 name: str,
                 type_parameters: Union[Tuple[HailType, ...], List[HailType]],
                 value_parameter_names: Union[Tuple[str, ...], List[str]],
                 value_parameter_types: Union[Tuple[HailType, ...], List[HailType]],
                 return_type: HailType,
                 body: Expression):
        assert len(value_parameter_names) == len(value_parameter_types)
        render = CSERenderer()
        self._name = name
        self._type_parameters = type_parameters
        self._value_parameter_names = value_parameter_names
        self._value_parameter_types = value_parameter_types
        self._return_type = return_type
        self._rendered_body = render(finalize_randomness(body._ir))

    def to_dataclass(self):
        return SerializedIRFunction(
            name=self._name,
            type_parameters=[tp._parsable_string() for tp in self._type_parameters],
            value_parameter_names=list(self._value_parameter_names),
            value_parameter_types=[vpt._parsable_string() for vpt in self._value_parameter_types],
            return_type=self._return_type._parsable_string(),
            rendered_body=self._rendered_body,
        )


@dataclass
class ServiceBackendExecutePayload(ActionPayload):
    functions: List[SerializedIRFunction]
    idempotency_token: str
    payload: ExecutePayload


@dataclass
class CloudfuseConfig:
    bucket: str
    mount_path: str
    read_only: bool


@dataclass
class SequenceConfig:
    fasta: str
    index: str


@dataclass
class ServiceBackendRPCConfig:
    tmp_dir: str
    remote_tmpdir: str
    billing_project: str
    worker_cores: str
    worker_memory: str
    storage: str
    cloudfuse_configs: List[CloudfuseConfig]
    regions: List[str]
    flags: Dict[str, str]
    custom_references: List[str]
    liftovers: Dict[str, Dict[str, str]]
    sequences: Dict[str, SequenceConfig]


class ServiceBackend(Backend):
    # is.hail.backend.service.Main protocol
    WORKER = "worker"
    DRIVER = "driver"

    # is.hail.backend.service.ServiceBackendSocketAPI2 protocol
    LOAD_REFERENCES_FROM_DATASET = 1
    VALUE_TYPE = 2
    TABLE_TYPE = 3
    MATRIX_TABLE_TYPE = 4
    BLOCK_MATRIX_TYPE = 5
    EXECUTE = 6
    PARSE_VCF_METADATA = 7
    IMPORT_FAM = 8
    FROM_FASTA_FILE = 9

    @staticmethod
    async def create(*,
                     billing_project: Optional[str] = None,
                     batch_client: Optional[aiohb.BatchClient] = None,
                     disable_progress_bar: Optional[bool] = None,
                     remote_tmpdir: Optional[str] = None,
                     flags: Optional[Dict[str, str]] = None,
                     jar_url: Optional[str] = None,
                     driver_cores: Optional[Union[int, str]] = None,
                     driver_memory: Optional[str] = None,
                     worker_cores: Optional[Union[int, str]] = None,
                     worker_memory: Optional[str] = None,
                     name_prefix: Optional[str] = None,
                     credentials_token: Optional[str] = None,
                     regions: Optional[List[str]] = None,
                     gcs_requester_pays_configuration: Optional[GCSRequesterPaysConfiguration] = None,
                     gcs_bucket_allow_list: Optional[List[str]] = None):
        billing_project = configuration_of(ConfigVariable.BATCH_BILLING_PROJECT, billing_project, None)
        if billing_project is None:
            raise ValueError(
                "No billing project.  Call 'init_batch' with the billing "
                "project or run 'hailctl config set batch/billing_project "
                "MY_BILLING_PROJECT'"
            )
        gcs_requester_pays_configuration = get_gcs_requester_pays_configuration(
            gcs_requester_pays_configuration=gcs_requester_pays_configuration,
        )
        async_fs = RouterAsyncFS(
            gcs_kwargs={'gcs_requester_pays_configuration': gcs_requester_pays_configuration},
            gcs_bucket_allow_list=gcs_bucket_allow_list
        )
        sync_fs = RouterFS(async_fs)
        if batch_client is None:
            batch_client = await aiohb.BatchClient.create(billing_project, _token=credentials_token)
        bc = hb.BatchClient.from_async(batch_client)
        remote_tmpdir = get_remote_tmpdir('ServiceBackend', remote_tmpdir=remote_tmpdir)

        jar_url = configuration_of(ConfigVariable.QUERY_JAR_URL, jar_url, None)
        jar_spec = GitRevision(revision()) if jar_url is None else JarUrl(jar_url)

        name_prefix = configuration_of(ConfigVariable.QUERY_NAME_PREFIX, name_prefix, '')
        batch_attributes: Dict[str, str] = {
            'hail-version': version(),
        }
        if name_prefix:
            batch_attributes['name'] = name_prefix

        driver_cores = configuration_of(ConfigVariable.QUERY_BATCH_DRIVER_CORES, driver_cores, None)
        driver_memory = configuration_of(ConfigVariable.QUERY_BATCH_DRIVER_MEMORY, driver_memory, None)
        worker_cores = configuration_of(ConfigVariable.QUERY_BATCH_WORKER_CORES, worker_cores, None)
        worker_memory = configuration_of(ConfigVariable.QUERY_BATCH_WORKER_MEMORY, worker_memory, None)

        if regions is None:
            regions_from_conf = configuration_of(ConfigVariable.BATCH_REGIONS, regions, None)
            if regions_from_conf is not None:
                assert isinstance(regions_from_conf, str)
                regions = regions_from_conf.split(',')

        if regions is None or regions == ANY_REGION:
            regions = bc.supported_regions()

        assert len(regions) > 0, regions

        if disable_progress_bar is None:
            disable_progress_bar_str = configuration_of(ConfigVariable.QUERY_DISABLE_PROGRESS_BAR, None, None)
            if disable_progress_bar_str is None:
                disable_progress_bar = not am_i_interactive()
            else:
                disable_progress_bar = len(disable_progress_bar_str) > 0

        flags = flags or {}
        if 'gcs_requester_pays_project' in flags or 'gcs_requester_pays_buckets' in flags:
            raise ValueError(
                'Specify neither gcs_requester_pays_project nor gcs_requester_'
                'pays_buckets in the flags argument to ServiceBackend.create'
            )
        if gcs_requester_pays_configuration is not None:
            if isinstance(gcs_requester_pays_configuration, str):
                flags['gcs_requester_pays_project'] = gcs_requester_pays_configuration
            else:
                assert isinstance(gcs_requester_pays_configuration, tuple)
                flags['gcs_requester_pays_project'] = gcs_requester_pays_configuration[0]
                flags['gcs_requester_pays_buckets'] = ','.join(gcs_requester_pays_configuration[1])

        sb = ServiceBackend(
            billing_project=billing_project,
            sync_fs=sync_fs,
            async_fs=async_fs,
            bc=bc,
            disable_progress_bar=disable_progress_bar,
            batch_attributes=batch_attributes,
            remote_tmpdir=remote_tmpdir,
            jar_spec=jar_spec,
            driver_cores=driver_cores,
            driver_memory=driver_memory,
            worker_cores=worker_cores,
            worker_memory=worker_memory,
            regions=regions
        )
        sb._initialize_flags(flags)
        return sb

    def __init__(self,
                 *,
                 billing_project: str,
                 sync_fs: FS,
                 async_fs: RouterAsyncFS,
                 bc: hb.BatchClient,
                 disable_progress_bar: bool,
                 batch_attributes: Dict[str, str],
                 remote_tmpdir: str,
                 jar_spec: JarSpec,
                 driver_cores: Optional[Union[int, str]],
                 driver_memory: Optional[str],
                 worker_cores: Optional[Union[int, str]],
                 worker_memory: Optional[str],
                 regions: List[str]):
        super(ServiceBackend, self).__init__()
        self.billing_project = billing_project
        self._sync_fs = sync_fs
        self._async_fs = async_fs
        self.bc = bc
        self.async_bc = self.bc._async_client
        self._batch_was_submitted: bool = False
        self.disable_progress_bar = disable_progress_bar
        self.batch_attributes = batch_attributes
        self.remote_tmpdir = remote_tmpdir
        self.flags: Dict[str, str] = {}
        self.jar_spec = jar_spec
        self.functions: List[IRFunction] = []
        self._registered_ir_function_names: Set[str] = set()
        self.driver_cores = driver_cores
        self.driver_memory = driver_memory
        self.worker_cores = worker_cores
        self.worker_memory = worker_memory
        self.regions = regions

        self._batch: aiohb.Batch = self._create_batch()

    def _create_batch(self) -> aiohb.Batch:
        return self.async_bc.create_batch(attributes=self.batch_attributes)

    def validate_file(self, uri: str) -> None:
        validate_file(uri, self._async_fs, validate_scheme=True)

    def debug_info(self) -> Dict[str, Any]:
        return {
            'jar_spec': str(self.jar_spec),
            'billing_project': self.billing_project,
            'batch_attributes': self.batch_attributes,
            'remote_tmpdir': self.remote_tmpdir,
            'flags': self.flags,
            'driver_cores': self.driver_cores,
            'driver_memory': self.driver_memory,
            'worker_cores': self.worker_cores,
            'worker_memory': self.worker_memory,
            'regions': self.regions,
        }

    @property
    def fs(self) -> FS:
        return self._sync_fs

    @property
    def logger(self):
        return log

    def stop(self):
        async_to_blocking(self._async_fs.close())
        async_to_blocking(self.async_bc.close())
        self.functions = []
        self._registered_ir_function_names = set()

    async def _run_on_batch(
        self,
        name: str,
        service_backend_config: ServiceBackendRPCConfig,
        action: ActionTag,
        payload: ActionPayload,
        *,
        progress: Optional[BatchProgressBar] = None,
        driver_cores: Optional[Union[int, str]] = None,
        driver_memory: Optional[str] = None,
    ) -> Tuple[bytes, str]:
        timings = Timings()
        with TemporaryDirectory(ensure_exists=False) as iodir:
            with timings.step("write input"):
                async with await self._async_fs.create(iodir + '/in') as infile:
                    await infile.write(orjson.dumps({
                        'config': service_backend_config,
                        'action': action.value,
                        'payload': payload,
                    }))

            with timings.step("submit batch"):
                resources: Dict[str, Union[str, bool]] = {'preemptible': False}
                if driver_cores is not None:
                    resources['cpu'] = str(driver_cores)
                elif self.driver_cores is not None:
                    resources['cpu'] = str(self.driver_cores)

                if driver_memory is not None:
                    resources['memory'] = str(driver_memory)
                elif self.driver_memory is not None:
                    resources['memory'] = str(self.driver_memory)

                if service_backend_config.storage != '0Gi':
                    resources['storage'] = service_backend_config.storage

                j = self._batch.create_jvm_job(
                    jar_spec=self.jar_spec.to_dict(),
                    argv=[
                        ServiceBackend.DRIVER,
                        name,
                        iodir + '/in',
                        iodir + '/out',
                    ],
                    mount_tokens=needs_tokens_mounted(),
                    resources=resources,
                    attributes={'name': name + '_driver'},
                    regions=self.regions,
                    cloudfuse=[(c.bucket, c.mount_path, c.read_only) for c in service_backend_config.cloudfuse_configs],
                    profile=self.flags['profile'] is not None,
                )
                await self._batch.submit(disable_progress_bar=True)
                self._batch_was_submitted = True

            with timings.step("wait driver"):
                try:
                    await asyncio.sleep(0.6)  # it is not possible for the batch to be finished in less than 600ms
                    await self._batch.wait(
                        description=name,
                        disable_progress_bar=self.disable_progress_bar,
                        progress=progress,
                        starting_job=j.job_id,
                    )
                except KeyboardInterrupt:
                    raise
                except Exception:
                    await self._batch.cancel()
                    self._batch = self._create_batch()
                    self._batch_was_submitted = False
                    raise

            with timings.step("read output"):
                result_bytes = await retry_transient_errors(self._read_output, iodir + '/out', iodir + '/in')
                return result_bytes, str(timings.to_dict())

    async def _read_output(self, output_uri: str, input_uri: str) -> bytes:
        try:
            driver_output = await self._async_fs.open(output_uri)
        except FileNotFoundError as exc:
            raise FatalError('Hail internal error. Please contact the Hail team and provide the following information.\n\n' + yamlx.dump({
                'service_backend_debug_info': self.debug_info(),
                'batch_debug_info': await self._batch.debug_info(
                    _jobs_query_string='bad',
                    _max_jobs=10
                ),
                'input_uri': await self._async_fs.read(input_uri),
            })) from exc

        try:
            async with driver_output as outfile:
                success = await read_bool(outfile)
                if success:
                    return await read_bytes(outfile)

                short_message = await read_str(outfile)
                expanded_message = await read_str(outfile)
                error_id = await read_int(outfile)

                raise fatal_error_from_java_error_triplet(short_message, expanded_message, error_id)
        except UnexpectedEOFError as exc:
            raise FatalError('Hail internal error. Please contact the Hail team and provide the following information.\n\n' + yamlx.dump({
                'service_backend_debug_info': self.debug_info(),
                'batch_debug_info': await self._batch.debug_info(
                    _jobs_query_string='bad',
                    _max_jobs=10
                ),
                'in': await self._async_fs.read(input_uri),
                'out': await self._async_fs.read(output_uri),
            })) from exc

    def _cancel_on_ctrl_c(self, coro: Awaitable[T]) -> T:
        try:
            return async_to_blocking(coro)
        except KeyboardInterrupt:
            if self._batch_was_submitted:
                print("Received a keyboard interrupt, cancelling the batch...")
                async_to_blocking(self._batch.cancel())
                self._batch = self._create_batch()
                self._batch_was_submitted = False
            raise

    def _rpc(self, action: ActionTag, payload: ActionPayload) -> Tuple[bytes, str]:
        return self._cancel_on_ctrl_c(self._async_rpc(action, payload))

    async def _async_rpc(self, action: ActionTag, payload: ActionPayload):
        if isinstance(payload, ExecutePayload):
            payload = ServiceBackendExecutePayload([f.to_dataclass() for f in self.functions], self._batch.token, payload)

        storage_requirement_bytes = 0
        readonly_fuse_buckets: Set[str] = set()

        added_sequences = {rg.name: rg._sequence_files for rg in self._references.values() if rg._sequence_files is not None}
        sequence_file_mounts = {}
        for rg_name, (fasta_file, index_file) in added_sequences.items():
            fasta_bucket, fasta_path = self._get_bucket_and_path(fasta_file)
            index_bucket, index_path = self._get_bucket_and_path(index_file)
            for bucket, blob in [(fasta_bucket, fasta_file), (index_bucket, index_file)]:
                readonly_fuse_buckets.add(bucket)
                storage_requirement_bytes += await (await self._async_fs.statfile(blob)).size()
            sequence_file_mounts[rg_name] = SequenceConfig(f'/cloudfuse/{fasta_bucket}/{fasta_path}', f'/cloudfuse/{index_bucket}/{index_path}')

        storage_gib_str = f'{math.ceil(storage_requirement_bytes / 1024 / 1024 / 1024)}Gi'
        qob_config = ServiceBackendRPCConfig(
            tmp_dir=tmp_dir(),
            remote_tmpdir=self.remote_tmpdir,
            billing_project=self.billing_project,
            worker_cores=str(self.worker_cores),
            worker_memory=str(self.worker_memory),
            storage=storage_gib_str,
            cloudfuse_configs=[CloudfuseConfig(bucket, f'/cloudfuse/{bucket}', True) for bucket in readonly_fuse_buckets],
            regions=self.regions,
            flags=self.flags,
            custom_references=[orjson.dumps(rg._config).decode('utf-8') for rg in self._references.values() if rg.name not in BUILTIN_REFERENCES],
            liftovers={rg.name: rg._liftovers for rg in self._references.values() if len(rg._liftovers) > 0},
            sequences=sequence_file_mounts,
        )
        return await self._run_on_batch(f'{action.name.lower()}(...)', qob_config, action, payload)

    # Sequence and liftover information is stored on the ReferenceGenome
    # and there is no persistent backend to keep in sync.
    # Sequence and liftover information are passed on RPC
    def add_sequence(self, name, fasta_file, index_file):  # pylint: disable=unused-argument
        # FIXME Not only should this be in the cloud, it should be in the *right* cloud
        for blob in (fasta_file, index_file):
            self.validate_file(blob)

    def remove_sequence(self, name):  # pylint: disable=unused-argument
        pass

    def _get_bucket_and_path(self, blob_uri):
        url = self._async_fs.parse_url(blob_uri)
        return '/'.join(url.bucket_parts), url.path

    def add_liftover(self, name: str, chain_file: str, dest_reference_genome: str):  # pylint: disable=unused-argument
        pass

    def remove_liftover(self, name, dest_reference_genome):  # pylint: disable=unused-argument
        pass

    def register_ir_function(self,
                             name: str,
                             type_parameters: Union[Tuple[HailType, ...], List[HailType]],
                             value_parameter_names: Union[Tuple[str, ...], List[str]],
                             value_parameter_types: Union[Tuple[HailType, ...], List[HailType]],
                             return_type: HailType,
                             body: Expression):
        self._registered_ir_function_names.add(name)
        self.functions.append(IRFunction(
            name,
            type_parameters,
            value_parameter_names,
            value_parameter_types,
            return_type,
            body
        ))

    def _is_registered_ir_function_name(self, name: str) -> bool:
        return name in self._registered_ir_function_names

    def persist_expression(self, expr):
        # FIXME: should use context manager to clean up persisted resources
        fname = TemporaryFilename(prefix='persist_expression').name
        write_expression(expr, fname)
        return read_expression(fname, _assert_type=expr.dtype)

    def set_flags(self, **flags: str):
        unknown_flags = set(flags) - self._valid_flags()
        if unknown_flags:
            raise ValueError(f'unknown flags: {", ".join(unknown_flags)}')
        if 'gcs_requester_pays_project' in flags or 'gcs_requester_pays_buckets' in flags:
            warnings.warn(
                'Modifying the requester pays project or buckets at runtime '
                'using flags is deprecated. Expect this behavior to become '
                'unsupported soon.'
            )
        self.flags.update(flags)

    def get_flags(self, *flags: str) -> Mapping[str, str]:
        unknown_flags = set(flags) - self._valid_flags()
        if unknown_flags:
            raise ValueError(f'unknown flags: {", ".join(unknown_flags)}')
        if 'gcs_requester_pays_project' in flags or 'gcs_requester_pays_buckets' in flags:
            warnings.warn(
                'Retrieving the requester pays project or buckets at runtime '
                'using flags is deprecated. Expect this behavior to become '
                'unsupported soon.'
            )
        return {flag: self.flags[flag] for flag in flags if flag in self.flags}

    @property
    def requires_lowering(self):
        return True
