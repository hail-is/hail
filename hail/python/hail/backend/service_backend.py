from typing import Dict, Optional, Callable, Awaitable, Mapping, Any, List, Union, Tuple, TypeVar
import abc
import math
import struct
from hail.expr.expressions.base_expression import Expression
import orjson
import logging

from hail.context import TemporaryDirectory, tmp_dir, TemporaryFilename, revision
from hail.utils import FatalError
from hail.expr.types import HailType, dtype, ttuple, tvoid
from hail.expr.table_type import ttable
from hail.expr.matrix_type import tmatrix
from hail.expr.blockmatrix_type import tblockmatrix
from hail.experimental import write_expression, read_expression
from hail.ir import finalize_randomness
from hail.ir.renderer import CSERenderer

from hailtop import yamlx
from hailtop.config import (configuration_of, get_remote_tmpdir)
from hailtop.utils import async_to_blocking, secret_alnum_string, TransientError, Timings, am_i_interactive
from hailtop.utils.rich_progress_bar import BatchProgressBar
from hailtop.batch_client import client as hb
from hailtop.batch_client import aioclient as aiohb
from hailtop.aiotools.fs import AsyncFS
from hailtop.aiotools.router_fs import RouterAsyncFS
import hailtop.aiotools.fs as afs

from .backend import Backend, fatal_error_from_java_error_triplet
from ..builtin_references import BUILTIN_REFERENCES
from ..fs.fs import FS
from ..fs.router_fs import RouterFS
from ..ir import BaseIR
from ..utils import ANY_REGION


ReferenceGenomeConfig = Dict[str, Any]


T = TypeVar("T")


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


async def write_str_array(strm: afs.WritableStream, los: List[str]):
    await write_int(strm, len(los))
    for s in los:
        await write_str(strm, s)


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
                     token: Optional[str] = None,
                     regions: Optional[List[str]] = None):
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
        remote_tmpdir = get_remote_tmpdir('ServiceBackend', remote_tmpdir=remote_tmpdir)

        jar_url = configuration_of('query', 'jar_url', jar_url, None)
        jar_spec = GitRevision(revision()) if jar_url is None else JarUrl(jar_url)

        driver_cores = configuration_of('query', 'batch_driver_cores', driver_cores, None)
        driver_memory = configuration_of('query', 'batch_driver_memory', driver_memory, None)
        worker_cores = configuration_of('query', 'batch_worker_cores', worker_cores, None)
        worker_memory = configuration_of('query', 'batch_worker_memory', worker_memory, None)
        name_prefix = configuration_of('query', 'name_prefix', name_prefix, '')

        if regions is None:
            regions_from_conf = configuration_of('batch', 'regions', regions, None)
            if regions_from_conf is not None:
                assert isinstance(regions_from_conf, str)
                regions = regions_from_conf.split(',')

        if regions is None or regions == ANY_REGION:
            regions = bc.supported_regions()

        assert len(regions) > 0, regions

        if disable_progress_bar is None:
            disable_progress_bar_str = configuration_of('query', 'disable_progress_bar', None, None)
            if disable_progress_bar_str is None:
                disable_progress_bar = not am_i_interactive()
            else:
                disable_progress_bar = len(disable_progress_bar_str) > 0

        sb = ServiceBackend(
            billing_project=billing_project,
            sync_fs=sync_fs,
            async_fs=async_fs,
            bc=bc,
            disable_progress_bar=disable_progress_bar,
            batch_attributes=batch_attributes,
            remote_tmpdir=remote_tmpdir,
            flags=flags or {},
            jar_spec=jar_spec,
            driver_cores=driver_cores,
            driver_memory=driver_memory,
            worker_cores=worker_cores,
            worker_memory=worker_memory,
            name_prefix=name_prefix or '',
            regions=regions,
        )
        sb._initialize_flags()
        return sb

    def __init__(self,
                 *,
                 billing_project: str,
                 sync_fs: FS,
                 async_fs: AsyncFS,
                 bc: hb.BatchClient,
                 disable_progress_bar: bool,
                 batch_attributes: Dict[str, str],
                 remote_tmpdir: str,
                 flags: Dict[str, str],
                 jar_spec: JarSpec,
                 driver_cores: Optional[Union[int, str]],
                 driver_memory: Optional[str],
                 worker_cores: Optional[Union[int, str]],
                 worker_memory: Optional[str],
                 name_prefix: str,
                 regions: List[str]):
        super(ServiceBackend, self).__init__()
        self.billing_project = billing_project
        self._sync_fs = sync_fs
        self._async_fs = async_fs
        self.bc = bc
        self.async_bc = self.bc._async_client
        self._batch: Optional[aiohb.Batch] = None
        self.disable_progress_bar = disable_progress_bar
        self.batch_attributes = batch_attributes
        self.remote_tmpdir = remote_tmpdir
        self.flags = flags
        self.jar_spec = jar_spec
        self.functions: List[IRFunction] = []
        self.driver_cores = driver_cores
        self.driver_memory = driver_memory
        self.worker_cores = worker_cores
        self.worker_memory = worker_memory
        self.name_prefix = name_prefix
        self.regions = regions

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

    def validate_file_scheme(self, url):
        assert isinstance(self._async_fs, RouterAsyncFS)
        if self._async_fs.get_scheme(url) == 'file':
            raise ValueError(
                f'Found local filepath {url} when using Query on Batch. Specify a remote filepath instead.')

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
            readonly_fuse_buckets = set()
            storage_requirement_bytes = 0

            with timings.step("write input"):
                async with await self._async_fs.create(iodir + '/in') as infile:
                    nonnull_flag_count = sum(v is not None for v in self.flags.values())
                    await write_int(infile, nonnull_flag_count)
                    for k, v in self.flags.items():
                        if v is not None:
                            await write_str(infile, k)
                            await write_str(infile, v)
                    custom_references = [rg for rg in self._references.values() if rg.name not in BUILTIN_REFERENCES]
                    await write_int(infile, len(custom_references))
                    for reference_config in custom_references:
                        await write_str(infile, orjson.dumps(reference_config._config).decode('utf-8'))
                    non_empty_liftovers = {rg.name: rg._liftovers for rg in self._references.values() if len(rg._liftovers) > 0}
                    await write_int(infile, len(non_empty_liftovers))
                    for source_genome_name, liftovers in non_empty_liftovers.items():
                        await write_str(infile, source_genome_name)
                        await write_int(infile, len(liftovers))
                        for dest_reference_genome, chain_file in liftovers.items():
                            await write_str(infile, dest_reference_genome)
                            await write_str(infile, chain_file)
                    added_sequences = {rg.name: rg._sequence_files for rg in self._references.values() if rg._sequence_files is not None}
                    await write_int(infile, len(added_sequences))
                    for rg_name, (fasta_file, index_file) in added_sequences.items():
                        await write_str(infile, rg_name)
                        for blob in (fasta_file, index_file):
                            bucket, path = self._get_bucket_and_path(blob)
                            readonly_fuse_buckets.add(bucket)
                            storage_requirement_bytes += await (await self._async_fs.statfile(blob)).size()
                            await write_str(infile, f'/cloudfuse/{bucket}/{path}')
                    await write_int(infile, storage_requirement_bytes)
                    await write_str(infile, str(self.worker_cores))
                    await write_str(infile, str(self.worker_memory))
                    await write_int(infile, len(self.regions))
                    for region in self.regions:
                        await write_str(infile, region)
                    await inputs(infile, token)

            with timings.step("submit batch"):
                batch_attributes = self.batch_attributes
                if 'name' not in batch_attributes:
                    batch_attributes = {**batch_attributes, 'name': self.name_prefix}
                if self._batch is None:
                    bb = self.async_bc.create_batch(token=token, attributes=batch_attributes)
                else:
                    bb = await self.async_bc.update_batch(self._batch)

                resources: Dict[str, Union[str, bool]] = {'preemptible': False}
                if self.driver_cores is not None:
                    resources['cpu'] = str(self.driver_cores)
                if self.driver_memory is not None:
                    resources['memory'] = str(self.driver_memory)
                if storage_requirement_bytes != 0:
                    storage_gib = math.ceil(storage_requirement_bytes / 1024 / 1024 / 1024)
                    resources['storage'] = f'{storage_gib}Gi'

                j = bb.create_jvm_job(
                    jar_spec=self.jar_spec.to_dict(),
                    argv=[
                        ServiceBackend.DRIVER,
                        name,
                        iodir + '/in',
                        iodir + '/out',
                    ],
                    mount_tokens=True,
                    resources=resources,
                    attributes={'name': name + '_driver'},
                    regions=self.regions,
                    cloudfuse=[(bucket, f'/cloudfuse/{bucket}', True) for bucket in readonly_fuse_buckets]
                )
                self._batch = await bb.submit(disable_progress_bar=True)

            with timings.step("wait driver"):
                try:
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
                    self._batch = None
                    raise

            with timings.step("read output"):
                try:
                    driver_output = await self._async_fs.open(iodir + '/out')
                except FileNotFoundError as exc:
                    raise FatalError('Hail internal error. Please contact the Hail team and provide the following information.\n\n' + yamlx.dump({
                        'service_backend_debug_info': self.debug_info(),
                        'batch_debug_info': await self._batch.debug_info()
                    })) from exc

                async with driver_output as outfile:
                    success = await read_bool(outfile)
                    if success:
                        result_bytes = await read_bytes(outfile)
                        try:
                            return token, result_bytes, timings
                        except orjson.JSONDecodeError as err:
                            raise FatalError('Hail internal error. Please contact the Hail team and provide the following information.\n\n' + yamlx.dump({
                                'service_backend_debug_info': self.debug_info(),
                                'batch_debug_info': await self._batch.debug_info()
                            })) from err

                    short_message = await read_str(outfile)
                    expanded_message = await read_str(outfile)
                    error_id = await read_int(outfile)

                    reconstructed_error = fatal_error_from_java_error_triplet(short_message, expanded_message, error_id)
                    if ir is None:
                        raise reconstructed_error
                    raise reconstructed_error.maybe_user_error(ir)

    def _cancel_on_ctrl_c(self, coro: Awaitable[T]) -> T:
        try:
            return async_to_blocking(coro)
        except KeyboardInterrupt:
            if self._batch is not None:
                print("Received a keyboard interrupt, cancelling the batch...")
                async_to_blocking(self._batch.cancel())
            raise

    def execute(self, ir: BaseIR, timed: bool = False):
        return self._cancel_on_ctrl_c(self._async_execute(ir, timed=timed))

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

    def value_type(self, ir):
        return self._cancel_on_ctrl_c(self._async_value_type(ir))

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
        return self._cancel_on_ctrl_c(self._async_table_type(tir))

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
        return self._cancel_on_ctrl_c(self._async_matrix_type(mir))

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
        return self._cancel_on_ctrl_c(self._async_blockmatrix_type(bmir))

    async def _async_blockmatrix_type(self, bmir, *, progress: Optional[BatchProgressBar] = None):
        async def inputs(infile, _):
            await write_int(infile, ServiceBackend.BLOCK_MATRIX_TYPE)
            await write_str(infile, tmp_dir())
            await write_str(infile, self.billing_project)
            await write_str(infile, self.remote_tmpdir)
            await write_str(infile, self.render(bmir))
        _, resp, _ = await self._rpc('blockmatrix_type(...)', inputs, progress=progress)
        return tblockmatrix._from_json(orjson.loads(resp))

    def from_fasta_file(self, name, fasta_file, index_file, x_contigs, y_contigs, mt_contigs, par):
        return async_to_blocking(self._from_fasta_file(name, fasta_file, index_file, x_contigs, y_contigs, mt_contigs, par))

    async def _from_fasta_file(self, name, fasta_file, index_file, x_contigs, y_contigs, mt_contigs, par, *, progress: Optional[BatchProgressBar] = None):
        async def inputs(infile, _):
            await write_int(infile, ServiceBackend.FROM_FASTA_FILE)
            await write_str(infile, tmp_dir())
            await write_str(infile, self.billing_project)
            await write_str(infile, self.remote_tmpdir)
            await write_str(infile, name)
            await write_str(infile, fasta_file)
            await write_str(infile, index_file)
            await write_str_array(infile, x_contigs)
            await write_str_array(infile, y_contigs)
            await write_str_array(infile, mt_contigs)
            await write_str_array(infile, par)
        _, resp, _ = await self._rpc('from_fasta_file(...)', inputs, progress=progress)
        return orjson.loads(resp)

    def load_references_from_dataset(self, path):
        return self._cancel_on_ctrl_c(self._async_load_references_from_dataset(path))

    async def _async_load_references_from_dataset(self, path, *, progress: Optional[BatchProgressBar] = None):
        async def inputs(infile, _):
            await write_int(infile, ServiceBackend.LOAD_REFERENCES_FROM_DATASET)
            await write_str(infile, tmp_dir())
            await write_str(infile, self.billing_project)
            await write_str(infile, self.remote_tmpdir)
            await write_str(infile, path)
        _, resp, _ = await self._rpc('load_references_from_dataset(...)', inputs, progress=progress)
        return orjson.loads(resp)

    # Sequence and liftover information is stored on the ReferenceGenome
    # and there is no persistent backend to keep in sync.
    # Sequence and liftover information are passed on RPC
    def add_sequence(self, name, fasta_file, index_file):  # pylint: disable=unused-argument
        # FIXME Not only should this be in the cloud, it should be in the *right* cloud
        for blob in (fasta_file, index_file):
            self.validate_file_scheme(blob)

    def remove_sequence(self, name):  # pylint: disable=unused-argument
        pass

    def _get_bucket_and_path(self, blob_uri):
        url = self._async_fs.parse_url(blob_uri)
        return '/'.join(url.bucket_parts), url.path

    def add_liftover(self, name: str, chain_file: str, dest_reference_genome: str):  # pylint: disable=unused-argument
        pass

    def remove_liftover(self, name, dest_reference_genome):  # pylint: disable=unused-argument
        pass

    def parse_vcf_metadata(self, path):
        return self._cancel_on_ctrl_c(self._async_parse_vcf_metadata(path))

    async def _async_parse_vcf_metadata(self, path, *, progress: Optional[BatchProgressBar] = None):
        async def inputs(infile, _):
            await write_int(infile, ServiceBackend.PARSE_VCF_METADATA)
            await write_str(infile, tmp_dir())
            await write_str(infile, self.billing_project)
            await write_str(infile, self.remote_tmpdir)
            await write_str(infile, path)
        _, resp, _ = await self._rpc('parse_vcf_metadata(...)', inputs, progress=progress)
        return orjson.loads(resp)

    def import_fam(self, path: str, quant_pheno: bool, delimiter: str, missing: str):
        return self._cancel_on_ctrl_c(self._async_import_fam(path, quant_pheno, delimiter, missing))

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

    def set_flags(self, **flags: str):
        unknown_flags = set(flags) - self._valid_flags()
        if unknown_flags:
            raise ValueError(f'unknown flags: {", ".join(unknown_flags)}')
        self.flags.update(flags)

    def get_flags(self, *flags: str) -> Mapping[str, str]:
        unknown_flags = set(flags) - self._valid_flags()
        if unknown_flags:
            raise ValueError(f'unknown flags: {", ".join(unknown_flags)}')
        return {flag: self.flags[flag] for flag in flags if flag in self.flags}

    @property
    def requires_lowering(self):
        return True
