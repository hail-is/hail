from typing import Dict, Optional, Callable, Awaitable, Mapping, Any, List, Union, Tuple
import abc
import asyncio
import struct
import os
from hail.expr.expressions.base_expression import Expression
import orjson
import logging
import re
import yaml
from pathlib import Path

from hail.context import TemporaryDirectory, tmp_dir, TemporaryFilename, revision, _TemporaryFilenameManager
from hail.utils import FatalError
from hail.expr.types import HailType, dtype, ttuple, tvoid
from hail.expr.table_type import ttable
from hail.expr.matrix_type import tmatrix
from hail.expr.blockmatrix_type import tblockmatrix
from hail.experimental import write_expression, read_expression
from hail.ir import finalize_randomness
from hail.ir.renderer import CSERenderer

from hailtop.config import (configuration_of, get_user_local_cache_dir, get_remote_tmpdir, get_deploy_config)
from hailtop.utils import async_to_blocking, secret_alnum_string, TransientError, Timings
from hailtop.utils.rich_progress_bar import BatchProgressBar
from hailtop.batch_client import client as hb
from hailtop.batch_client import aioclient as aiohb
from hailtop.aiotools.fs import AsyncFS
from hailtop.aiotools.router_fs import RouterAsyncFS
import hailtop.aiotools.fs as afs

from .backend import Backend, fatal_error_from_java_error_triplet
from ..builtin_references import BUILTIN_REFERENCE_DOWNLOAD_LOCKS
from ..fs.fs import FS
from ..fs.router_fs import RouterFS
from ..ir import BaseIR
from ..context import version
from ..utils import frozendict


log = logging.getLogger('backend.service_backend')


async def write_bool(strm: afs.WritableStream, v: bool):
    if v:
        await strm.write(b'\x01')
    else:
        await strm.write(b'\x00')


async def write_int(strm: afs.WritableStream, v: int):
    await strm.write(struct.pack('<i', v))


async def write_long(strm: afs.WritableStream, v: int):
    await strm.write(struct.pack('<q', v))


async def write_bytes(strm: afs.WritableStream, b: bytes):
    n = len(b)
    await write_int(strm, n)
    await strm.write(b)


async def write_str(strm: afs.WritableStream, s: str):
    await write_bytes(strm, s.encode('utf-8'))


class EndOfStream(TransientError):
    pass


async def read_byte(strm: afs.ReadableStream) -> int:
    return (await strm.readexactly(1))[0]


async def read_bool(strm: afs.ReadableStream) -> bool:
    return (await read_byte(strm)) != 0


async def read_int(strm: afs.ReadableStream) -> int:
    b = await strm.readexactly(4)
    return struct.unpack('<i', b)[0]


async def read_long(strm: afs.ReadableStream) -> int:
    b = await strm.readexactly(8)
    return struct.unpack('<q', b)[0]


async def read_bytes(strm: afs.ReadableStream) -> bytes:
    n = await read_int(strm)
    return await strm.readexactly(n)


async def read_str(strm: afs.ReadableStream) -> str:
    b = await read_bytes(strm)
    return b.decode('utf-8')


class yaml_literally_shown_str(str):
    pass


def yaml_literally_shown_str_representer(dumper, data):
    return dumper.represent_scalar(u'tag:yaml.org,2002:str', data, style='|')


yaml.add_representer(yaml_literally_shown_str, yaml_literally_shown_str_representer)


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


class IRFunction:
    def __init__(self,
                 name: str,
                 type_parameters: Union[Tuple[HailType, ...], List[HailType]],
                 value_parameter_names: Union[Tuple[str, ...], List[str]],
                 value_parameter_types: Union[Tuple[HailType, ...], List[HailType]],
                 return_type: HailType,
                 body: Expression):
        assert len(value_parameter_names) == len(value_parameter_types)
        render = CSERenderer(stop_at_jir=True)
        self._name = name
        self._type_parameters = type_parameters
        self._value_parameter_names = value_parameter_names
        self._value_parameter_types = value_parameter_types
        self._return_type = return_type
        self._rendered_body = render(finalize_randomness(body._ir))

    async def serialize(self, writer: afs.WritableStream):
        await write_str(writer, self._name)

        await write_int(writer, len(self._type_parameters))
        for type_parameter in self._type_parameters:
            await write_str(writer, type_parameter._parsable_string())

        await write_int(writer, len(self._value_parameter_names))
        for value_parameter_name in self._value_parameter_names:
            await write_str(writer, value_parameter_name)

        await write_int(writer, len(self._value_parameter_types))
        for value_parameter_type in self._value_parameter_types:
            await write_str(writer, value_parameter_type._parsable_string())

        await write_str(writer, self._return_type._parsable_string())
        await write_str(writer, self._rendered_body)


class ServiceBackend(Backend):
    HAIL_BATCH_FAILURE_EXCEPTION_MESSAGE_RE = re.compile("is.hail.backend.service.HailBatchFailure: ([0-9]+)\n")

    # is.hail.backend.service.Main protocol
    WORKER = "worker"
    DRIVER = "driver"

    # is.hail.backend.service.ServiceBackendSocketAPI2 protocol
    LOAD_REFERENCES_FROM_DATASET = 1
    VALUE_TYPE = 2
    TABLE_TYPE = 3
    MATRIX_TABLE_TYPE = 4
    BLOCK_MATRIX_TYPE = 5
    REFERENCE_GENOME = 6
    EXECUTE = 7
    PARSE_VCF_METADATA = 8
    INDEX_BGEN = 9
    IMPORT_FAM = 10

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
                     token: Optional[str] = None):
        billing_project = configuration_of('batch', 'billing_project', billing_project, None)
        if billing_project is None:
            raise ValueError(
                "No billing project.  Call 'init_batch' with the billing "
                "project or run 'hailctl config set batch/billing_project "
                "MY_BILLING_PROJECT'"
            )

        async_fs = RouterAsyncFS('file')
        sync_fs = RouterFS(async_fs)
        if batch_client is None:
            batch_client = await aiohb.BatchClient.create(billing_project, _token=token)
        bc = hb.BatchClient.from_async(batch_client)
        batch_attributes: Dict[str, str] = dict()
        user_local_reference_cache_dir = Path(get_user_local_cache_dir(), 'references', version())
        os.makedirs(user_local_reference_cache_dir, exist_ok=True)
        remote_tmpdir = get_remote_tmpdir('ServiceBackend', remote_tmpdir=remote_tmpdir)

        jar_url = configuration_of('query', 'jar_url', jar_url, None)
        jar_spec = GitRevision(revision()) if jar_url is None else JarUrl(jar_url)

        driver_cores = configuration_of('query', 'batch_driver_cores', driver_cores, None)
        driver_memory = configuration_of('query', 'batch_driver_memory', driver_memory, None)
        worker_cores = configuration_of('query', 'batch_worker_cores', worker_cores, None)
        worker_memory = configuration_of('query', 'batch_worker_memory', worker_memory, None)
        name_prefix = configuration_of('query', 'name_prefix', name_prefix, '')

        if disable_progress_bar is None:
            disable_progress_bar_str = configuration_of('query', 'disable_progress_bar', None, '1')
            disable_progress_bar = len(disable_progress_bar_str) > 0

        flags = {"use_new_shuffle": "1", **(flags or {})}

        return ServiceBackend(
            billing_project=billing_project,
            sync_fs=sync_fs,
            async_fs=async_fs,
            bc=bc,
            disable_progress_bar=disable_progress_bar,
            batch_attributes=batch_attributes,
            user_local_reference_cache_dir=user_local_reference_cache_dir,
            remote_tmpdir=remote_tmpdir,
            flags=flags,
            jar_spec=jar_spec,
            driver_cores=driver_cores,
            driver_memory=driver_memory,
            worker_cores=worker_cores,
            worker_memory=worker_memory,
            name_prefix=name_prefix or '',
        )

    def __init__(self,
                 *,
                 billing_project: str,
                 sync_fs: FS,
                 async_fs: AsyncFS,
                 bc: hb.BatchClient,
                 disable_progress_bar: bool,
                 batch_attributes: Dict[str, str],
                 user_local_reference_cache_dir: Path,
                 remote_tmpdir: str,
                 flags: Dict[str, str],
                 jar_spec: JarSpec,
                 driver_cores: Optional[Union[int, str]],
                 driver_memory: Optional[str],
                 worker_cores: Optional[Union[int, str]],
                 worker_memory: Optional[str],
                 name_prefix: str):
        self.billing_project = billing_project
        self._sync_fs = sync_fs
        self._async_fs = async_fs
        self.bc = bc
        self.async_bc = self.bc._async_client
        self.disable_progress_bar = disable_progress_bar
        self.batch_attributes = batch_attributes
        self.user_local_reference_cache_dir = user_local_reference_cache_dir
        self.remote_tmpdir = remote_tmpdir
        self.flags = flags
        self.jar_spec = jar_spec
        self.functions: List[IRFunction] = []
        self.driver_cores = driver_cores
        self.driver_memory = driver_memory
        self.worker_cores = worker_cores
        self.worker_memory = worker_memory
        self.name_prefix = name_prefix
        self._persisted_locations: Dict[Any, _TemporaryFilenameManager] = dict()

    def debug_info(self) -> Dict[str, Any]:
        return {
            'jar_spec': str(self.jar_spec),
            'billing_project': self.billing_project,
            'batch_attributes': self.batch_attributes,
            'user_local_reference_cache_dir': str(self.user_local_reference_cache_dir),
            'remote_tmpdir': self.remote_tmpdir,
            'flags': self.flags,
            'driver_cores': self.driver_cores,
            'driver_memory': self.driver_memory,
            'worker_cores': self.worker_cores,
            'worker_memory': self.worker_memory,
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

    def render(self, ir):
        r = CSERenderer()
        assert len(r.jirs) == 0
        return r(finalize_randomness(ir))

    async def _rpc(self,
                   name: str,
                   inputs: Callable[[afs.WritableStream, str], Awaitable[None]],
                   *,
                   ir: Optional[BaseIR] = None,
                   progress: Optional[BatchProgressBar] = None):
        timings = Timings()
        token = secret_alnum_string()
        with TemporaryDirectory(ensure_exists=False) as iodir:
            with timings.step("write input"):
                async with await self._async_fs.create(iodir + '/in') as infile:
                    nonnull_flag_count = sum(v is not None for v in self.flags.values())
                    await write_int(infile, nonnull_flag_count)
                    for k, v in self.flags.items():
                        if v is not None:
                            await write_str(infile, k)
                            await write_str(infile, v)
                    await write_str(infile, str(self.worker_cores))
                    await write_str(infile, str(self.worker_memory))
                    await inputs(infile, token)

            with timings.step("submit batch"):
                batch_attributes = self.batch_attributes
                if 'name' not in batch_attributes:
                    batch_attributes = {**batch_attributes, 'name': self.name_prefix + name}
                bb = self.async_bc.create_batch(token=token, attributes=batch_attributes)

                resources: Dict[str, Union[str, bool]] = {'preemptible': False}
                if self.driver_cores is not None:
                    resources['cpu'] = str(self.driver_cores)
                if self.driver_memory is not None:
                    resources['memory'] = str(self.driver_memory)

                j = bb.create_jvm_job(
                    jar_spec=self.jar_spec.to_dict(),
                    argv=[
                        ServiceBackend.DRIVER,
                        batch_attributes['name'],
                        iodir + '/in',
                        iodir + '/out',
                    ],
                    mount_tokens=True,
                    resources=resources,
                    attributes={'name': 'driver'},
                )
                b = await bb.submit(disable_progress_bar=True)

            with timings.step("wait batch"):
                try:
                    if self.disable_progress_bar is not True:
                        deploy_config = get_deploy_config()
                        url = deploy_config.external_url('batch', f'/batches/{b.id}/jobs/1')
                        print(f'Submitted batch {b.id}, see {url}')

                    status = await b.wait(description=name,
                                          disable_progress_bar=self.disable_progress_bar,
                                          progress=progress)
                except Exception:
                    await b.cancel()
                    raise

            with timings.step("parse status"):
                if status['n_succeeded'] != status['n_jobs']:
                    job_status = await j.status()
                    if 'status' in job_status:
                        if 'error' in job_status['status']:
                            job_status['status']['error'] = yaml_literally_shown_str(job_status['status']['error'].strip())
                    logs = await j.log()
                    for k in logs:
                        logs[k] = yaml_literally_shown_str(logs[k].strip())
                    message = {'service_backend_debug_info': self.debug_info(),
                               'batch_status': status,
                               'job_status': job_status,
                               'log': logs}
                    log.error(yaml.dump(message))
                    raise FatalError(message)

            with timings.step("read output"):
                async with await self._async_fs.open(iodir + '/out') as outfile:
                    success = await read_bool(outfile)
                    if success:
                        result_bytes = await read_bytes(outfile)
                        try:
                            return token, result_bytes, timings
                        except orjson.JSONDecodeError as err:
                            raise FatalError(f'batch id was {b.id}\ncould not decode {result_bytes}') from err
                    else:
                        short_message = await read_str(outfile)
                        expanded_message = await read_str(outfile)
                        error_id = await read_int(outfile)
                        if error_id == -1:
                            error_id = None
                        maybe_batch_id = ServiceBackend.HAIL_BATCH_FAILURE_EXCEPTION_MESSAGE_RE.match(expanded_message)
                        if error_id is not None:
                            assert maybe_batch_id is None, str((short_message, expanded_message, error_id))
                            assert ir is not None
                            self._handle_fatal_error_from_backend(
                                fatal_error_from_java_error_triplet(short_message, expanded_message, error_id),
                                ir)
                        if maybe_batch_id is not None:
                            assert error_id is None, str((short_message, expanded_message, error_id))
                            batch_id = maybe_batch_id.groups()[0]
                            b2 = await self.async_bc.get_batch(batch_id)
                            b2_status = await b2.status()
                            assert b2_status['state'] != 'success'
                            failed_jobs = []
                            async for j in b2.jobs():
                                if j['state'] != 'Success':
                                    logs, job = await asyncio.gather(
                                        self.async_bc.get_job_log(j['batch_id'], j['job_id']),
                                        self.async_bc.get_job(j['batch_id'], j['job_id']),
                                    )
                                    full_status = job._status
                                    if 'status' in full_status:
                                        if 'error' in full_status['status']:
                                            full_status['status']['error'] = yaml_literally_shown_str(full_status['status']['error'].strip())
                                    main_log = logs.get('main', '')
                                    failed_jobs.append({
                                        'partial_status': j,
                                        'status': full_status,
                                        'log': yaml_literally_shown_str(main_log.strip()),
                                    })
                            message = {
                                'id': b.id,
                                'service_backend_debug_info': self.debug_info(),
                                'short_message': yaml_literally_shown_str(short_message.strip()),
                                'expanded_message': yaml_literally_shown_str(expanded_message.strip()),
                                'cause': {'id': batch_id, 'batch_status': b2_status, 'failed_jobs': failed_jobs}}
                            log.error(yaml.dump(message))
                            raise FatalError(orjson.dumps(message).decode('utf-8'))
                        raise FatalError(f'batch id was {b.id}\n' + short_message + '\n' + expanded_message)

    def execute(self, ir: BaseIR, timed: bool = False):
        return async_to_blocking(self._async_execute(ir, timed=timed))

    async def _async_execute(self,
                             ir: BaseIR,
                             *,
                             timed: bool = False,
                             progress: Optional[BatchProgressBar] = None):
        async def inputs(infile, token):
            await write_int(infile, ServiceBackend.EXECUTE)
            await write_str(infile, tmp_dir())
            await write_str(infile, self.billing_project)
            await write_str(infile, self.remote_tmpdir)
            await write_str(infile, self.render(ir))
            await write_str(infile, token)
            await write_int(infile, len(self.functions))
            for fun in self.functions:
                await fun.serialize(infile)
            await write_str(infile, '{"name":"StreamBufferSpec"}')

        _, resp, timings = await self._rpc('execute(...)', inputs, ir=ir, progress=progress)
        typ: HailType = ir.typ
        if typ == tvoid:
            assert resp == b'', (typ, resp)
            converted_value = None
        else:
            converted_value = ttuple(typ)._from_encoding(resp)[0]
        if timed:
            return converted_value, timings
        return converted_value

    def execute_many(self, *irs, timed=False):
        return async_to_blocking(self._async_execute_many(*irs, timed=timed))

    async def _async_execute_many(self,
                                  *irs,
                                  timed=False,
                                  progress: Optional[BatchProgressBar] = None):
        if progress is None:
            with BatchProgressBar() as progress:
                return await asyncio.gather(*[self._async_execute(ir, timed=timed, progress=progress)
                                              for ir in irs])
        else:
            return await asyncio.gather(*[self._async_execute(ir, timed=timed, progress=progress)
                                          for ir in irs])

    def value_type(self, ir):
        return async_to_blocking(self._async_value_type(ir))

    async def _async_value_type(self, ir, *, progress: Optional[BatchProgressBar] = None):
        async def inputs(infile, _):
            await write_int(infile, ServiceBackend.VALUE_TYPE)
            await write_str(infile, tmp_dir())
            await write_str(infile, self.billing_project)
            await write_str(infile, self.remote_tmpdir)
            await write_str(infile, self.render(ir))
        _, resp, _ = await self._rpc('value_type(...)', inputs, progress=progress)
        return dtype(orjson.loads(resp))

    def table_type(self, tir):
        return async_to_blocking(self._async_table_type(tir))

    async def _async_table_type(self, tir, *, progress: Optional[BatchProgressBar] = None):
        async def inputs(infile, _):
            await write_int(infile, ServiceBackend.TABLE_TYPE)
            await write_str(infile, tmp_dir())
            await write_str(infile, self.billing_project)
            await write_str(infile, self.remote_tmpdir)
            await write_str(infile, self.render(tir))
        _, resp, _ = await self._rpc('table_type(...)', inputs, progress=progress)
        return ttable._from_json(orjson.loads(resp))

    def matrix_type(self, mir):
        return async_to_blocking(self._async_matrix_type(mir))

    async def _async_matrix_type(self, mir, *, progress: Optional[BatchProgressBar] = None):
        async def inputs(infile, _):
            await write_int(infile, ServiceBackend.MATRIX_TABLE_TYPE)
            await write_str(infile, tmp_dir())
            await write_str(infile, self.billing_project)
            await write_str(infile, self.remote_tmpdir)
            await write_str(infile, self.render(mir))
        _, resp, _ = await self._rpc('matrix_type(...)', inputs, progress=progress)
        return tmatrix._from_json(orjson.loads(resp))

    def blockmatrix_type(self, bmir):
        return async_to_blocking(self._async_blockmatrix_type(bmir))

    async def _async_blockmatrix_type(self, bmir, *, progress: Optional[BatchProgressBar] = None):
        async def inputs(infile, _):
            await write_int(infile, ServiceBackend.BLOCK_MATRIX_TYPE)
            await write_str(infile, tmp_dir())
            await write_str(infile, self.billing_project)
            await write_str(infile, self.remote_tmpdir)
            await write_str(infile, self.render(bmir))
        _, resp, _ = await self._rpc('blockmatrix_type(...)', inputs, progress=progress)
        return tblockmatrix._from_json(orjson.loads(resp))

    def add_reference(self, config):
        raise NotImplementedError("ServiceBackend does not support 'add_reference'")

    def from_fasta_file(self, name, fasta_file, index_file, x_contigs, y_contigs, mt_contigs, par):
        raise NotImplementedError("ServiceBackend does not support 'from_fasta_file'")

    def remove_reference(self, name):
        raise NotImplementedError("ServiceBackend does not support 'remove_reference'")

    def get_reference(self, name):
        return async_to_blocking(self._async_get_reference(name))

    async def _async_get_reference(self, name, *, progress: Optional[BatchProgressBar] = None):
        async def inputs(infile, _):
            await write_int(infile, ServiceBackend.REFERENCE_GENOME)
            await write_str(infile, tmp_dir())
            await write_str(infile, self.billing_project)
            await write_str(infile, self.remote_tmpdir)
            await write_str(infile, name)

        if name in BUILTIN_REFERENCE_DOWNLOAD_LOCKS:
            with BUILTIN_REFERENCE_DOWNLOAD_LOCKS[name]:
                try:
                    with open(Path(self.user_local_reference_cache_dir, name)) as f:
                        return orjson.loads(f.read())
                except FileNotFoundError:
                    _, resp, _ = await self._rpc('get_reference(...)', inputs, progress=progress)
                    with open(Path(self.user_local_reference_cache_dir, name), 'wb') as f:
                        f.write(resp)
        else:
            _, resp, _ = await self._rpc('get_reference(...)', inputs)

        return orjson.loads(resp)

    def get_references(self, names):
        return async_to_blocking(self._async_get_references(names))

    async def _async_get_references(self, names, *, progress: Optional[BatchProgressBar] = None):
        if progress is None:
            with BatchProgressBar() as progress:
                return await asyncio.gather(*[self._async_get_reference(name, progress=progress)
                                              for name in names])
        else:
            return await asyncio.gather(*[self._async_get_reference(name, progress=progress)
                                          for name in names])

    def load_references_from_dataset(self, path):
        return async_to_blocking(self._async_load_references_from_dataset(path))

    async def _async_load_references_from_dataset(self, path, *, progress: Optional[BatchProgressBar] = None):
        async def inputs(infile, _):
            await write_int(infile, ServiceBackend.LOAD_REFERENCES_FROM_DATASET)
            await write_str(infile, tmp_dir())
            await write_str(infile, self.billing_project)
            await write_str(infile, self.remote_tmpdir)
            await write_str(infile, path)
        _, resp, _ = await self._rpc('load_references_from_dataset(...)', inputs, progress=progress)
        return orjson.loads(resp)

    def add_sequence(self, name, fasta_file, index_file):
        raise NotImplementedError("ServiceBackend does not support 'add_sequence'")

    def remove_sequence(self, name):
        raise NotImplementedError("ServiceBackend does not support 'remove_sequence'")

    def add_liftover(self, name, chain_file, dest_reference_genome):
        raise NotImplementedError("ServiceBackend does not support 'add_liftover'")

    def remove_liftover(self, name, dest_reference_genome):
        raise NotImplementedError("ServiceBackend does not support 'remove_liftover'")

    def parse_vcf_metadata(self, path):
        return async_to_blocking(self._async_parse_vcf_metadata(path))

    async def _async_parse_vcf_metadata(self, path, *, progress: Optional[BatchProgressBar] = None):
        async def inputs(infile, _):
            await write_int(infile, ServiceBackend.PARSE_VCF_METADATA)
            await write_str(infile, tmp_dir())
            await write_str(infile, self.billing_project)
            await write_str(infile, self.remote_tmpdir)
            await write_str(infile, path)
        _, resp, _ = await self._rpc('parse_vcf_metadata(...)', inputs, progress=progress)
        return orjson.loads(resp)

    def index_bgen(self,
                   files: List[str],
                   index_file_map: Dict[str, str],
                   referenceGenomeName: Optional[str],
                   contig_recoding: Dict[str, str],
                   skip_invalid_loci: bool):
        return async_to_blocking(self._async_index_bgen(
            files,
            index_file_map,
            referenceGenomeName,
            contig_recoding,
            skip_invalid_loci
        ))

    async def _async_index_bgen(self,
                                files: List[str],
                                index_file_map: Dict[str, str],
                                referenceGenomeName: Optional[str],
                                contig_recoding: Dict[str, str],
                                skip_invalid_loci: bool,
                                *,
                                progress: Optional[BatchProgressBar] = None):
        async def inputs(infile, _):
            await write_int(infile, ServiceBackend.INDEX_BGEN)
            await write_str(infile, tmp_dir())
            await write_str(infile, self.billing_project)
            await write_str(infile, self.remote_tmpdir)
            await write_int(infile, len(files))
            for fname in files:
                await write_str(infile, fname)
            await write_int(infile, len(index_file_map))
            for k, v in index_file_map.items():
                await write_str(infile, k)
                await write_str(infile, v)
            if referenceGenomeName is None:
                await write_bool(infile, False)
            else:
                await write_bool(infile, True)
                await write_str(infile, referenceGenomeName)
            await write_int(infile, len(contig_recoding))
            for k, v in contig_recoding.items():
                await write_str(infile, k)
                await write_str(infile, v)
            await write_bool(infile, skip_invalid_loci)

        _, resp, _ = await self._rpc('index_bgen(...)', inputs, progress=progress)
        assert resp == b'null'
        return None

    def import_fam(self, path: str, quant_pheno: bool, delimiter: str, missing: str):
        return async_to_blocking(self._async_import_fam(path, quant_pheno, delimiter, missing))

    async def _async_import_fam(self,
                                path: str,
                                quant_pheno: bool,
                                delimiter: str,
                                missing: str,
                                *,
                                progress: Optional[BatchProgressBar] = None):
        async def inputs(infile, _):
            await write_int(infile, ServiceBackend.IMPORT_FAM)
            await write_str(infile, tmp_dir())
            await write_str(infile, self.billing_project)
            await write_str(infile, self.remote_tmpdir)
            await write_str(infile, path)
            await write_bool(infile, quant_pheno)
            await write_str(infile, delimiter)
            await write_str(infile, missing)
        _, resp, _ = await self._rpc('import_fam(...)', inputs, progress=progress)
        return orjson.loads(resp)

    def register_ir_function(self,
                             name: str,
                             type_parameters: Union[Tuple[HailType, ...], List[HailType]],
                             value_parameter_names: Union[Tuple[str, ...], List[str]],
                             value_parameter_types: Union[Tuple[HailType, ...], List[HailType]],
                             return_type: HailType,
                             body: Expression):
        self.functions.append(IRFunction(
            name,
            type_parameters,
            value_parameter_names,
            value_parameter_types,
            return_type,
            body
        ))

    def persist_expression(self, expr):
        # FIXME: should use context manager to clean up persisted resources
        fname = TemporaryFilename(prefix='persist_expression').name
        write_expression(expr, fname)
        return read_expression(fname, _assert_type=expr.dtype)

    def persist_table(self, t, storage_level):
        tf = TemporaryFilename(prefix='persist_table')
        self._persisted_locations[t] = tf
        return t.checkpoint(tf.__enter__())

    def unpersist_table(self, t):
        try:
            self._persisted_locations[t].__exit__(None, None, None)
        except KeyError as err:
            raise ValueError(f'{t} is not persisted') from err

    def persist_matrix_table(self, mt, storage_level):
        tf = TemporaryFilename(prefix='persist_matrix_table')
        self._persisted_locations[mt] = tf
        return mt.checkpoint(tf.__enter__())

    def unpersist_matrix_table(self, mt):
        try:
            self._persisted_locations[mt].__exit__(None, None, None)
        except KeyError as err:
            raise ValueError(f'{mt} is not persisted') from err

    def set_flags(self, **flags: str):
        self.flags.update(flags)

    def get_flags(self, *flags) -> Mapping[str, str]:
        return frozendict(self.flags)

    @property
    def requires_lowering(self):
        return True
