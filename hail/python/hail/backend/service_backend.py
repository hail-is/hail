import asyncio
import logging
import math
import struct
import warnings
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Awaitable, Dict, List, Mapping, NoReturn, Optional, Set, Tuple, TypeVar, Union

import orjson

import hailtop.aiotools.fs as afs
from hail.context import TemporaryDirectory, TemporaryFilename
from hail.experimental import read_expression, write_expression
from hail.utils import FatalError
from hail.version import __revision__, __version__
from hailtop import yamlx
from hailtop.aiocloud.aiogoogle import GCSRequesterPaysConfiguration, get_gcs_requester_pays_configuration
from hailtop.aiotools.fs.exceptions import UnexpectedEOFError
from hailtop.aiotools.router_fs import RouterAsyncFS
from hailtop.aiotools.validators import validate_file
from hailtop.batch_client.aioclient import Batch, BatchClient, JobGroup
from hailtop.config import ConfigVariable, configuration_of, get_remote_tmpdir
from hailtop.fs.fs import FS
from hailtop.fs.router_fs import RouterFS
from hailtop.hail_event_loop import hail_event_loop
from hailtop.utils import Timings, am_i_interactive, async_to_blocking, retry_transient_errors
from hailtop.utils.rich_progress_bar import BatchProgressBar

from ..builtin_references import BUILTIN_REFERENCES
from ..utils import ANY_REGION
from .backend import ActionPayload, ActionTag, Backend, fatal_error_from_java_error_triplet

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
    flags: Dict[str, str]
    custom_references: List[str]
    liftovers: Dict[str, Dict[str, str]]
    sequences: Dict[str, SequenceConfig]


@dataclass
class BatchJobConfig:
    worker_cores: str
    worker_memory: str
    storage: str
    cloudfuse_configs: List[CloudfuseConfig]
    regions: List[str]


class ServiceBackend(Backend):
    # is.hail.backend.service.Main protocol
    WORKER = "worker"
    DRIVER = "driver"

    @staticmethod
    async def create(
        *,
        billing_project: Optional[str] = None,
        batch_client: Optional[BatchClient] = None,
        disable_progress_bar: Optional[bool] = None,
        remote_tmpdir: Optional[str] = None,
        flags: Optional[Dict[str, str]] = None,
        driver_cores: Optional[Union[int, str]] = None,
        driver_memory: Optional[str] = None,
        worker_cores: Optional[Union[int, str]] = None,
        worker_memory: Optional[str] = None,
        batch_id: Optional[int] = None,
        name_prefix: Optional[str] = None,
        credentials_token: Optional[str] = None,
        regions: Optional[List[str]] = None,
        gcs_requester_pays_configuration: Optional[GCSRequesterPaysConfiguration] = None,
        gcs_bucket_allow_list: Optional[List[str]] = None,
        branching_factor: Optional[int] = None,
    ):
        async_exit_stack = AsyncExitStack()
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
            gcs_bucket_allow_list=gcs_bucket_allow_list,
        )
        async_exit_stack.push_async_callback(async_fs.close)
        sync_fs = RouterFS(async_fs)
        if batch_client is None:
            batch_client = await BatchClient.create(billing_project, _token=credentials_token)
            async_exit_stack.push_async_callback(batch_client.close)

        remote_tmpdir = get_remote_tmpdir('ServiceBackend', remote_tmpdir=remote_tmpdir)

        name_prefix = configuration_of(ConfigVariable.QUERY_NAME_PREFIX, name_prefix, '')
        batch_attributes: Dict[str, str] = {
            'hail-version': __version__,
        }
        if name_prefix:
            batch_attributes['name'] = name_prefix

        driver_cores = configuration_of(ConfigVariable.QUERY_BATCH_DRIVER_CORES, driver_cores, None)
        driver_memory = configuration_of(ConfigVariable.QUERY_BATCH_DRIVER_MEMORY, driver_memory, None)
        worker_cores = configuration_of(ConfigVariable.QUERY_BATCH_WORKER_CORES, worker_cores, None)
        worker_memory = configuration_of(ConfigVariable.QUERY_BATCH_WORKER_MEMORY, worker_memory, None)

        if regions == ANY_REGION:
            regions = await batch_client.supported_regions()
        elif regions is None:
            fallback = await batch_client.default_region()
            regions_from_conf = configuration_of(
                ConfigVariable.BATCH_REGIONS, explicit_argument=None, fallback=fallback
            )
            regions = regions_from_conf.split(',')

        assert len(regions) > 0, regions

        if disable_progress_bar is None:
            disable_progress_bar_str = configuration_of(ConfigVariable.QUERY_DISABLE_PROGRESS_BAR, None, None)
            if disable_progress_bar_str is None:
                disable_progress_bar = not am_i_interactive()
            else:
                disable_progress_bar = len(disable_progress_bar_str) > 0

        flags = flags or {}
        if branching_factor is not None:
            flags['branching_factor'] = str(branching_factor)

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
            batch_client=batch_client,
            batch=(
                (await batch_client.get_batch(batch_id))
                if batch_id is not None
                else batch_client.create_batch(attributes=batch_attributes)
            ),
            disable_progress_bar=disable_progress_bar,
            remote_tmpdir=remote_tmpdir,
            driver_cores=driver_cores,
            driver_memory=driver_memory,
            worker_cores=worker_cores,
            worker_memory=worker_memory,
            regions=regions,
            async_exit_stack=async_exit_stack,
        )
        sb._initialize_flags(flags)
        return sb

    def __init__(
        self,
        *,
        billing_project: str,
        sync_fs: FS,
        async_fs: RouterAsyncFS,
        batch_client: BatchClient,
        batch: Batch,
        disable_progress_bar: bool,
        remote_tmpdir: str,
        driver_cores: Optional[Union[int, str]],
        driver_memory: Optional[str],
        worker_cores: Optional[Union[int, str]],
        worker_memory: Optional[str],
        regions: List[str],
        async_exit_stack: AsyncExitStack,
    ):
        super(ServiceBackend, self).__init__()
        self.billing_project = billing_project
        self._sync_fs = sync_fs
        self._async_fs = async_fs
        self._batch_client = batch_client
        self._batch = batch
        self._job_group_was_submitted: bool = False
        self.disable_progress_bar = disable_progress_bar
        self._remote_tmpdir = remote_tmpdir
        self.flags: Dict[str, str] = {}
        self._registered_ir_function_names: Set[str] = set()
        self.driver_cores = driver_cores
        self.driver_memory = driver_memory
        self.worker_cores = worker_cores
        self.worker_memory = worker_memory
        self.regions = regions
        self._job_group: Optional[JobGroup] = None
        self._async_exit_stack = async_exit_stack

    def validate_file(self, uri: str) -> None:
        async_to_blocking(validate_file(uri, self._async_fs))

    def debug_info(self) -> Dict[str, Any]:
        return {
            'jar_spec': self.jar_spec,
            'billing_project': self.billing_project,
            'batch_attributes': self._batch.attributes,
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
    def jar_spec(self) -> dict:
        return {'type': 'git_revision', 'value': __revision__}

    @property
    def logger(self):
        return log

    def stop(self):
        hail_event_loop().run_until_complete(self._stop())
        super().stop()

    async def _stop(self):
        await self._async_exit_stack.aclose()

    async def _run_on_batch(
        self,
        name: str,
        service_backend_config: ServiceBackendRPCConfig,
        job_config: BatchJobConfig,
        action: ActionTag,
        payload: ActionPayload,
        *,
        progress: Optional[BatchProgressBar] = None,
        driver_cores: Optional[Union[int, str]] = None,
        driver_memory: Optional[str] = None,
    ) -> Tuple[bytes, Optional[dict]]:
        timings = Timings()
        async with TemporaryDirectory(ensure_exists=False) as iodir:
            with timings.step("write input"):
                async with await self._async_fs.create(iodir + '/in') as infile:
                    await infile.write(
                        orjson.dumps({
                            'rpc_config': service_backend_config,
                            'job_config': job_config,
                            'action': action.value,
                            'payload': payload,
                        })
                    )

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

                if job_config.storage != '0Gi':
                    resources['storage'] = job_config.storage

                self._job_group = self._batch.create_job_group(attributes={'name': name})
                self._batch.create_jvm_job(
                    jar_spec=self.jar_spec,
                    argv=[
                        ServiceBackend.DRIVER,
                        name,
                        iodir + '/in',
                        iodir + '/out',
                    ],
                    job_group=self._job_group,
                    resources=resources,
                    attributes={'name': name + '_driver'},
                    regions=self.regions,
                    cloudfuse=[(c.bucket, c.mount_path, c.read_only) for c in job_config.cloudfuse_configs],
                    profile=self.flags['profile'] is not None,
                )
                await self._batch.submit(disable_progress_bar=True)
                self._job_group_was_submitted = True

            with timings.step("wait driver"):
                try:
                    await asyncio.sleep(0.6)  # it is not possible for the batch to be finished in less than 600ms
                    await self._job_group.wait(
                        description=name,
                        disable_progress_bar=self.disable_progress_bar,
                        progress=progress,
                    )
                except KeyboardInterrupt:
                    raise
                except Exception:
                    await self._job_group.cancel()
                    self._job_group_was_submitted = False
                    raise

            with timings.step("read output"):
                result_bytes = await retry_transient_errors(self._read_output, iodir + '/out', iodir + '/in')
                return result_bytes, timings.to_dict()

    async def _read_output(self, output_uri: str, input_uri: str) -> bytes:
        try:
            driver_output = await self._async_fs.open(output_uri)
        except FileNotFoundError as exc:
            raise FatalError(
                'Hail internal error. Please contact the Hail team and provide the following information.\n\n'
                + yamlx.dump({
                    'service_backend_debug_info': self.debug_info(),
                    'batch_debug_info': await self._batch.debug_info(_jobs_query_string='bad', _max_jobs=10),
                    'input_uri': await self._async_fs.read(input_uri),
                })
            ) from exc

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
            raise FatalError(
                'Hail internal error. Please contact the Hail team and provide the following information.\n\n'
                + yamlx.dump({
                    'service_backend_debug_info': self.debug_info(),
                    'batch_debug_info': await self._batch.debug_info(_jobs_query_string='bad', _max_jobs=10),
                    'in': await self._async_fs.read(input_uri),
                    'out': await self._async_fs.read(output_uri),
                })
            ) from exc

    def _cancel_on_ctrl_c(self, coro: Awaitable[T]) -> T:
        try:
            return async_to_blocking(coro)
        except KeyboardInterrupt:
            if self._job_group_was_submitted:
                print("Received a keyboard interrupt, cancelling the batch...")
                async_to_blocking(self._job_group.cancel())
                self._job_group_was_submitted = False
            raise

    def _rpc(self, action: ActionTag, payload: ActionPayload) -> Tuple[bytes, Optional[dict]]:
        return self._cancel_on_ctrl_c(self._async_rpc(action, payload))

    async def _async_rpc(self, action: ActionTag, payload: ActionPayload):
        storage_requirement_bytes = 0
        readonly_fuse_buckets: Set[str] = set()

        added_sequences = {
            rg.name: rg._sequence_files for rg in self._references.values() if rg._sequence_files is not None
        }
        sequence_file_mounts = {}
        for rg_name, (fasta_file, index_file) in added_sequences.items():
            fasta_bucket, fasta_path = self._get_bucket_and_path(fasta_file)
            index_bucket, index_path = self._get_bucket_and_path(index_file)
            for bucket, blob in [(fasta_bucket, fasta_file), (index_bucket, index_file)]:
                readonly_fuse_buckets.add(bucket)
                storage_requirement_bytes += await (await self._async_fs.statfile(blob)).size()
            sequence_file_mounts[rg_name] = SequenceConfig(
                f'/cloudfuse/{fasta_bucket}/{fasta_path}',
                f'/cloudfuse/{index_bucket}/{index_path}',
            )

        return await self._run_on_batch(
            name=f'{action.name.lower()}(...)',
            service_backend_config=ServiceBackendRPCConfig(
                tmp_dir=self.remote_tmpdir,
                flags=self.flags,
                custom_references=[
                    orjson.dumps(rg._config).decode('utf-8')
                    for rg in self._references.values()
                    if rg.name not in BUILTIN_REFERENCES
                ],
                liftovers={rg.name: rg._liftovers for rg in self._references.values() if len(rg._liftovers) > 0},
                sequences=sequence_file_mounts,
            ),
            job_config=BatchJobConfig(
                worker_cores=str(self.worker_cores),
                worker_memory=str(self.worker_memory),
                storage=f'{math.ceil(storage_requirement_bytes / 1024 / 1024 / 1024)}Gi',
                cloudfuse_configs=[
                    CloudfuseConfig(bucket, f'/cloudfuse/{bucket}', True) for bucket in readonly_fuse_buckets
                ],
                regions=self.regions,
            ),
            action=action,
            payload=payload,
        )

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

    @property
    def local_tmpdir(self) -> NoReturn:
        raise AttributeError('local tmp folders are not supported on the batch backend')

    @local_tmpdir.setter
    def local_tmpdir(self, tmpdir: str) -> NoReturn:
        raise AttributeError('local tmp folders are not supported on the batch backend')

    @property
    def remote_tmpdir(self) -> str:
        return self._remote_tmpdir

    @remote_tmpdir.setter
    def remote_tmpdir(self, tmpdir: str) -> None:
        self._remote_tmpdir = tmpdir
