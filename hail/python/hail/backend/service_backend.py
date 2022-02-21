from typing import Dict, Optional, Callable, Awaitable, Tuple
import asyncio
import struct
import os
import orjson
import logging
import contextlib
import re
import yaml
from pathlib import Path

from hail.context import TemporaryDirectory, tmp_dir
from hail.utils import FatalError
from hail.expr.types import dtype
from hail.expr.table_type import ttable
from hail.expr.matrix_type import tmatrix
from hail.expr.blockmatrix_type import tblockmatrix
from hail.ir.renderer import CSERenderer

from hailtop.config import get_user_config, get_user_local_cache_dir, get_remote_tmpdir
from hailtop.utils import async_to_blocking, secret_alnum_string, TransientError, time_msecs
from hailtop.batch_client import client as hb
from hailtop.batch_client import aioclient as aiohb
from hailtop.aiotools.fs import AsyncFS
from hailtop.aiotools.router_fs import RouterAsyncFS
import hailtop.aiotools.fs as afs

from .backend import Backend
from ..builtin_references import BUILTIN_REFERENCES
from ..fs.fs import FS
from ..fs.router_fs import RouterFS
from ..context import version


log = logging.getLogger('backend.service_backend')


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


class Timings:
    def __init__(self):
        self.timings: Dict[str, Dict[str, int]] = dict()

    @contextlib.contextmanager
    def step(self, name: str):
        assert name not in self.timings
        d: Dict[str, int] = dict()
        self.timings[name] = d
        d['start_time'] = time_msecs()
        yield
        d['finish_time'] = time_msecs()
        d['duration'] = d['finish_time'] - d['start_time']

    def to_dict(self):
        return self.timings


class yaml_literally_shown_str(str):
    pass


def yaml_literally_shown_str_representer(dumper, data):
    return dumper.represent_scalar(u'tag:yaml.org,2002:str', data, style='|')


yaml.add_representer(yaml_literally_shown_str, yaml_literally_shown_str_representer)


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
    GOODBYE = 254

    @staticmethod
    async def create(*,
                     billing_project: Optional[str] = None,
                     batch_client: Optional[aiohb.BatchClient] = None,
                     skip_logging_configuration: Optional[bool] = None,
                     disable_progress_bar: bool = True,
                     remote_tmpdir: Optional[str] = None):
        del skip_logging_configuration

        if billing_project is None:
            billing_project = get_user_config().get('batch', 'billing_project', fallback=None)
        if billing_project is None:
            raise ValueError(
                "No billing project.  Call 'init_service' with the billing "
                "project or run 'hailctl config set batch/billing_project "
                "MY_BILLING_PROJECT'"
            )

        async_fs = RouterAsyncFS('file')
        sync_fs = RouterFS(async_fs)
        if batch_client is None:
            async_client = await aiohb.BatchClient.create(billing_project)
        bc = hb.BatchClient.from_async(async_client)
        batch_attributes: Dict[str, str] = dict()
        user_local_reference_cache_dir = Path(get_user_local_cache_dir(), 'references', version())
        os.makedirs(user_local_reference_cache_dir, exist_ok=True)
        remote_tmpdir = get_remote_tmpdir('ServiceBackend', remote_tmpdir=remote_tmpdir)

        return ServiceBackend(
            billing_project=billing_project,
            sync_fs=sync_fs,
            async_fs=async_fs,
            bc=bc,
            disable_progress_bar=disable_progress_bar,
            batch_attributes=batch_attributes,
            user_local_reference_cache_dir=user_local_reference_cache_dir,
            remote_tmpdir=remote_tmpdir,
        )

    def __init__(self,
                 billing_project: str,
                 sync_fs: FS,
                 async_fs: AsyncFS,
                 bc: hb.BatchClient,
                 disable_progress_bar: bool,
                 batch_attributes: Dict[str, str],
                 user_local_reference_cache_dir: Path,
                 remote_tmpdir: str):
        self.billing_project = billing_project
        self._sync_fs = sync_fs
        self._async_fs = async_fs
        self.bc = bc
        self.async_bc = self.bc._async_client
        self.disable_progress_bar = disable_progress_bar
        self.batch_attributes = batch_attributes
        self.user_local_reference_cache_dir = user_local_reference_cache_dir
        self.remote_tmpdir = remote_tmpdir

    @property
    def fs(self) -> FS:
        return self._sync_fs

    @property
    def logger(self):
        return log

    @property
    def stop(self):
        pass

    def render(self, ir):
        r = CSERenderer()
        assert len(r.jirs) == 0
        return r(ir)

    async def _rpc(self,
                   name: str,
                   inputs: Callable[[afs.WritableStream, str], Awaitable[Tuple[str, dict]]]):
        timings = Timings()
        token = secret_alnum_string()
        iodir = TemporaryDirectory(ensure_exists=False).name  # FIXME: actually cleanup
        with TemporaryDirectory(ensure_exists=False) as _:
            with timings.step("write input"):
                async with await self._async_fs.create(iodir + '/in') as infile:
                    await inputs(infile, token)

            with timings.step("submit batch"):
                batch_attributes = self.batch_attributes
                if 'name' not in batch_attributes:
                    batch_attributes = {**batch_attributes, 'name': name}
                bb = self.async_bc.create_batch(token=token, attributes=batch_attributes)

                j = bb.create_jvm_job([
                    ServiceBackend.DRIVER,
                    os.environ['HAIL_SHA'],
                    os.environ['HAIL_JAR_URL'],
                    batch_attributes['name'],
                    iodir + '/in',
                    iodir + '/out',
                ], mount_tokens=True)
                b = await bb.submit(disable_progress_bar=self.disable_progress_bar)

            with timings.step("wait batch"):
                try:
                    status = await b.wait(disable_progress_bar=self.disable_progress_bar)
                except Exception:
                    await b.cancel()
                    raise

            with timings.step("parse status"):
                if status['n_succeeded'] != 1:
                    job_status = await j.status()
                    if 'status' in job_status:
                        if 'error' in job_status['status']:
                            job_status['status']['error'] = yaml_literally_shown_str(job_status['status']['error'].strip())
                    logs = await j.log()
                    for k in logs:
                        logs[k] = yaml_literally_shown_str(logs[k].strip())
                    message = {'batch_status': status,
                               'job_status': job_status,
                               'log': logs}
                    log.error(yaml.dump(message))
                    raise ValueError(message)

            with timings.step("read output"):
                async with await self._async_fs.open(iodir + '/out') as outfile:
                    success = await read_bool(outfile)
                    if success:
                        b = await read_bytes(outfile)
                        try:
                            return token, orjson.loads(b), timings
                        except orjson.JSONDecodeError as err:
                            raise ValueError(f'batch id was {b.id}\ncould not decode {b}') from err
                    else:
                        jstacktrace = await read_str(outfile)
                        maybe_id = ServiceBackend.HAIL_BATCH_FAILURE_EXCEPTION_MESSAGE_RE.match(jstacktrace)
                        if maybe_id:
                            batch_id = maybe_id.groups()[0]
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
                                'stacktrace': yaml_literally_shown_str(jstacktrace.strip()),
                                'cause': {'id': batch_id, 'batch_status': b2_status, 'failed_jobs': failed_jobs}}
                            log.error(yaml.dump(message))
                            raise ValueError(orjson.dumps(message).decode('utf-8'))
                        raise FatalError(f'batch id was {b.id}\n' + jstacktrace)

    def execute(self, ir, timed=False):
        return async_to_blocking(self._async_execute(ir, timed=timed))

    async def _async_execute(self, ir, timed=False):
        async def inputs(infile, token):
            await write_int(infile, ServiceBackend.EXECUTE)
            await write_str(infile, tmp_dir())
            await write_str(infile, self.billing_project)
            await write_str(infile, self.remote_tmpdir)
            await write_str(infile, self.render(ir))
            await write_str(infile, token)
        _, resp, timings = await self._rpc('execute(...)', inputs)
        typ = dtype(resp['type'])
        converted_value = typ._convert_from_json_na(resp['value'])
        if timed:
            return converted_value, timings
        return converted_value

    def execute_many(self, *irs, timed=False):
        return async_to_blocking(self._async_execute_many(*irs, timed=timed))

    async def _async_execute_many(self, *irs, timed=False):
        return await asyncio.gather(*[self._async_execute(ir, timed=timed) for ir in irs])

    def value_type(self, ir):
        return async_to_blocking(self._async_value_type(ir))

    async def _async_value_type(self, ir):
        async def inputs(infile, _):
            await write_int(infile, ServiceBackend.VALUE_TYPE)
            await write_str(infile, tmp_dir())
            await write_str(infile, self.billing_project)
            await write_str(infile, self.remote_tmpdir)
            await write_str(infile, self.render(ir))
        _, resp, _ = await self._rpc('value_type(...)', inputs)
        return dtype(resp)

    def table_type(self, tir):
        return async_to_blocking(self._async_table_type(tir))

    async def _async_table_type(self, tir):
        async def inputs(infile, _):
            await write_int(infile, ServiceBackend.TABLE_TYPE)
            await write_str(infile, tmp_dir())
            await write_str(infile, self.billing_project)
            await write_str(infile, self.remote_tmpdir)
            await write_str(infile, self.render(tir))
        _, resp, _ = await self._rpc('table_type(...)', inputs)
        return ttable._from_json(resp)

    def matrix_type(self, mir):
        return async_to_blocking(self._async_matrix_type(mir))

    async def _async_matrix_type(self, mir):
        async def inputs(infile, _):
            await write_int(infile, ServiceBackend.MATRIX_TABLE_TYPE)
            await write_str(infile, tmp_dir())
            await write_str(infile, self.billing_project)
            await write_str(infile, self.remote_tmpdir)
            await write_str(infile, self.render(mir))
        _, resp, _ = await self._rpc('matrix_type(...)', inputs)
        return tmatrix._from_json(resp)

    def blockmatrix_type(self, bmir):
        return async_to_blocking(self._async_blockmatrix_type(bmir))

    async def _async_blockmatrix_type(self, bmir):
        async def inputs(infile, _):
            await write_int(infile, ServiceBackend.BLOCK_MATRIX_TYPE)
            await write_str(infile, tmp_dir())
            await write_str(infile, self.billing_project)
            await write_str(infile, self.remote_tmpdir)
            await write_str(infile, self.render(bmir))
        _, resp, _ = await self._rpc('blockmatrix_type(...)', inputs)
        return tblockmatrix._from_json(resp)

    def add_reference(self, config):
        raise NotImplementedError("ServiceBackend does not support 'add_reference'")

    def from_fasta_file(self, name, fasta_file, index_file, x_contigs, y_contigs, mt_contigs, par):
        raise NotImplementedError("ServiceBackend does not support 'from_fasta_file'")

    def remove_reference(self, name):
        raise NotImplementedError("ServiceBackend does not support 'remove_reference'")

    def get_reference(self, name):
        return async_to_blocking(self._async_get_reference(name))

    async def _async_get_reference(self, name):
        if name in BUILTIN_REFERENCES:
            try:
                with open(Path(self.user_local_reference_cache_dir, name)) as f:
                    return orjson.loads(f.read())
            except FileNotFoundError:
                pass

        async def inputs(infile, _):
            await write_int(infile, ServiceBackend.REFERENCE_GENOME)
            await write_str(infile, tmp_dir())
            await write_str(infile, self.billing_project)
            await write_str(infile, self.remote_tmpdir)
            await write_str(infile, name)
        _, resp, _ = await self._rpc('get_reference(...)', inputs)
        if name in BUILTIN_REFERENCES:
            with open(Path(self.user_local_reference_cache_dir, name), 'wb') as f:
                f.write(orjson.dumps(resp))
        return resp

    def get_references(self, names):
        return async_to_blocking(self._async_get_references(names))

    async def _async_get_references(self, names):
        return await asyncio.gather(*[self._async_get_reference(name) for name in names])

    def load_references_from_dataset(self, path):
        return async_to_blocking(self._async_load_references_from_dataset(path))

    async def _async_load_references_from_dataset(self, path):
        async def inputs(infile, _):
            await write_int(infile, ServiceBackend.LOAD_REFERENCES_FROM_DATASET)
            await write_str(infile, tmp_dir())
            await write_str(infile, self.billing_project)
            await write_str(infile, self.remote_tmpdir)
            await write_str(infile, path)
        _, resp, _ = await self._rpc('load_references_from_dataset(...)', inputs)
        return resp

    def add_sequence(self, name, fasta_file, index_file):
        raise NotImplementedError("ServiceBackend does not support 'add_sequence'")

    def remove_sequence(self, name):
        raise NotImplementedError("ServiceBackend does not support 'remove_sequence'")

    def add_liftover(self, name, chain_file, dest_reference_genome):
        raise NotImplementedError("ServiceBackend does not support 'add_liftover'")

    def remove_liftover(self, name, dest_reference_genome):
        raise NotImplementedError("ServiceBackend does not support 'remove_liftover'")

    def parse_vcf_metadata(self, path):
        raise NotImplementedError("ServiceBackend does not support 'parse_vcf_metadata'")

    def index_bgen(self, files, index_file_map, rg, contig_recoding, skip_invalid_loci):
        raise NotImplementedError("ServiceBackend does not support 'index_bgen'")

    def import_fam(self, path: str, quant_pheno: bool, delimiter: str, missing: str):
        raise NotImplementedError("ServiceBackend does not support 'import_fam'")

    def register_ir_function(self, name, type_parameters, argument_names, argument_types, return_type, body):
        raise NotImplementedError("ServiceBackend does not support 'register_ir_function'")

    def persist_ir(self, ir):
        raise NotImplementedError("ServiceBackend does not support 'persist_ir'")
