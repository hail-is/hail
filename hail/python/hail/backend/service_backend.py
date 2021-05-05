from typing import BinaryIO, Dict
import asyncio
import struct
import os
import json
import logging
import contextlib

from hail.context import TemporaryDirectory, tmp_dir
from hail.utils import FatalError
from hail.expr.types import dtype, tvoid
from hail.expr.table_type import ttable
from hail.expr.matrix_type import tmatrix
from hail.expr.blockmatrix_type import tblockmatrix
from hail.ir.renderer import CSERenderer

from hailtop.config import get_deploy_config, get_user_config
from hailtop.auth import get_tokens
from hailtop.utils import async_to_blocking, secret_alnum_string, TransientError, time_msecs
from hailtop.batch_client import client as hb

from .backend import Backend
from ..fs.google_fs import GoogleCloudStorageFS


log = logging.getLogger('backend.service_backend')


def write_int(io: BinaryIO, v: int):
    io.write(struct.pack('<i', v))


def write_long(io: BinaryIO, v: int):
    io.write(struct.pack('<q', v))


def write_bytes(io: BinaryIO, b: bytes):
    n = len(b)
    write_int(io, n)
    io.write(b)


def write_str(io: BinaryIO, s: str):
    write_bytes(io, s.encode('utf-8'))


class EndOfStream(TransientError):
    pass


def read(io: BinaryIO, n: int) -> bytes:
    b = bytearray()
    left = n
    while left > 0:
        t = io.read(left)
        if not t:
            log.warning(f'unexpected EOS, Java violated protocol ({b})')
            raise EndOfStream()
        left -= len(t)
        b.extend(t)
    return b


def read_byte(io: BinaryIO) -> int:
    b = read(io, 1)
    return b[0]


def read_bool(io: BinaryIO) -> bool:
    return read_byte(io) != 0


def read_int(io: BinaryIO) -> int:
    b = read(io, 4)
    return struct.unpack('<i', b)[0]


def read_long(io: BinaryIO) -> int:
    b = read(io, 8)
    return struct.unpack('<q', b)[0]


def read_bytes(io: BinaryIO) -> bytes:
    n = read_int(io)
    return read(io, n)


def read_str(io: BinaryIO) -> str:
    b = read_bytes(io)
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


class ServiceBackend(Backend):
    LOAD_REFERENCES_FROM_DATASET = 1
    VALUE_TYPE = 2
    TABLE_TYPE = 3
    MATRIX_TABLE_TYPE = 4
    BLOCK_MATRIX_TYPE = 5
    REFERENCE_GENOME = 6
    EXECUTE = 7
    FLAGS = 8
    GET_FLAG = 9
    UNSET_FLAG = 10
    SET_FLAG = 11
    ADD_USER = 12
    GOODBYE = 254

    def __init__(self,
                 billing_project: str = None,
                 bucket: str = None,
                 *,
                 deploy_config=None,
                 skip_logging_configuration=None,
                 disable_progress_bar: bool = True):
        del skip_logging_configuration

        if billing_project is None:
            billing_project = get_user_config().get('batch', 'billing_project', fallback=None)
        if billing_project is None:
            billing_project = os.environ.get('HAIL_BILLING_PROJECT')
        if billing_project is None:
            raise ValueError(
                "No billing project.  Call 'init_service' with the billing "
                "project, set the HAIL_BILLING_PROJECT environment variable, "
                "or run 'hailctl config set batch/billing_project "
                "MY_BILLING_PROJECT'"
            )

        if bucket is None:
            bucket = get_user_config().get('batch', 'bucket', fallback=None)
        if bucket is None:
            bucket = os.environ.get('HAIL_BUCKET')
        if bucket is None:
            raise ValueError(
                'the bucket parameter of ServiceBackend must be set '
                'or run `hailctl config set batch/bucket '
                'MY_BUCKET`'
            )

        self.billing_project = billing_project
        self.bucket = bucket
        self._fs = GoogleCloudStorageFS()
        deploy_config = deploy_config or get_deploy_config()
        self.bc = hb.BatchClient(self.billing_project)
        self.async_bc = self.bc._async_client
        self.disable_progress_bar = disable_progress_bar
        self.batch_attributes: Dict[str, str] = dict()

    @property
    def fs(self) -> GoogleCloudStorageFS:
        return self._fs

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

    def execute(self, ir, timed=False):
        return async_to_blocking(self._async_execute(ir, timed=timed))

    async def _async_execute(self, ir, timed=False):
        result = await self._async_execute_untimed(ir)
        if timed:
            return result, dict()
        return result

    async def _async_execute_untimed(self, ir):
        token = secret_alnum_string()
        with TemporaryDirectory(ensure_exists=False) as dir:
            async def create_inputs():
                with self.fs.open(dir + '/in', 'wb') as infile:
                    write_int(infile, ServiceBackend.EXECUTE)
                    write_str(infile, tmp_dir())
                    write_str(infile, self.billing_project)
                    write_str(infile, self.bucket)
                    write_str(infile, self.render(ir))
                    write_str(infile, token)

            async def create_batch():
                batch_attributes = self.batch_attributes
                if 'name' not in batch_attributes:
                    batch_attributes = {**batch_attributes, 'name': 'execute(...)'}
                bb = self.async_bc.create_batch(token=token, attributes=batch_attributes)

                j = bb.create_jvm_job([
                    'is.hail.backend.service.ServiceBackendSocketAPI2',
                    os.environ['HAIL_SHA'],
                    os.environ['HAIL_JAR_URL'],
                    batch_attributes['name'],
                    dir + '/in',
                    dir + '/out',
                ], mount_tokens=True)
                return (j, await bb.submit(disable_progress_bar=self.disable_progress_bar))

            _, (j, b) = await asyncio.gather(create_inputs(), create_batch())

            status = await b.wait(disable_progress_bar=self.disable_progress_bar)
            if status['n_succeeded'] != 1:
                raise ValueError(f'batch failed {status} {await j.log()}')


            with self.fs.open(dir + '/out', 'rb') as outfile:
                success = read_bool(outfile)
                if success:
                    s = read_str(outfile)
                    try:
                        resp = json.loads(s)
                    except json.decoder.JSONDecodeError as err:
                        raise ValueError(f'could not decode {s}') from err
                else:
                    jstacktrace = read_str(outfile)
                    raise FatalError(jstacktrace)

        typ = dtype(resp['type'])
        if typ == tvoid:
            x = None
        else:
            x = typ._convert_from_json_na(resp['value'])

        return x

    def execute_many(self, *irs, timed=False):
        return async_to_blocking(self._async_execute_many(*irs, timed=timed))

    async def _async_execute_many(self, *irs, timed=False):
        return await asyncio.gather(*[self._async_execute(ir, timed=timed) for ir in irs])

    def value_type(self, ir):
        token = secret_alnum_string()
        with TemporaryDirectory(ensure_exists=False) as dir:
            with self.fs.open(dir + '/in', 'wb') as infile:
                write_int(infile, ServiceBackend.VALUE_TYPE)
                write_str(infile, tmp_dir())
                write_str(infile, self.render(ir))

            batch_attributes = self.batch_attributes
            if 'name' not in batch_attributes:
                batch_attributes = {**batch_attributes, 'name': 'value_type(...)'}
            bb = self.bc.create_batch(token=token, attributes=batch_attributes)

            j = bb.create_jvm_job([
                'is.hail.backend.service.ServiceBackendSocketAPI2',
                os.environ['HAIL_SHA'],
                os.environ['HAIL_JAR_URL'],
                batch_attributes['name'],
                dir + '/in',
                dir + '/out',
            ], mount_tokens=True)
            b = bb.submit(disable_progress_bar=self.disable_progress_bar)
            status = b.wait(disable_progress_bar=self.disable_progress_bar)
            if status['n_succeeded'] != 1:
                raise ValueError(f'batch failed {status} {j.log()}')


            with self.fs.open(dir + '/out', 'rb') as outfile:
                success = read_bool(outfile)
                if success:
                    s = read_str(outfile)
                    try:
                        return dtype(json.loads(s))
                    except json.decoder.JSONDecodeError as err:
                        raise ValueError(f'could not decode {s}') from err
                else:
                    jstacktrace = read_str(outfile)
                    raise FatalError(jstacktrace)

    def table_type(self, tir):
        token = secret_alnum_string()
        with TemporaryDirectory(ensure_exists=False) as dir:
            with self.fs.open(dir + '/in', 'wb') as infile:
                write_int(infile, ServiceBackend.TABLE_TYPE)
                write_str(infile, tmp_dir())
                write_str(infile, self.render(tir))

            batch_attributes = self.batch_attributes
            if 'name' not in batch_attributes:
                batch_attributes = {**batch_attributes, 'name': 'table_type(...)'}
            bb = self.bc.create_batch(token=token, attributes=batch_attributes)

            j = bb.create_jvm_job([
                'is.hail.backend.service.ServiceBackendSocketAPI2',
                os.environ['HAIL_SHA'],
                os.environ['HAIL_JAR_URL'],
                batch_attributes['name'],
                dir + '/in',
                dir + '/out',
            ], mount_tokens=True)
            b = bb.submit(disable_progress_bar=self.disable_progress_bar)
            status = b.wait(disable_progress_bar=self.disable_progress_bar)
            if status['n_succeeded'] != 1:
                raise ValueError(f'batch failed {status} {j.log()}')


            with self.fs.open(dir + '/out', 'rb') as outfile:
                success = read_bool(outfile)
                if success:
                    s = read_str(outfile)
                    try:
                        return ttable._from_json(json.loads(s))
                    except json.decoder.JSONDecodeError as err:
                        raise ValueError(f'could not decode {s}') from err
                else:
                    jstacktrace = read_str(outfile)
                    raise FatalError(jstacktrace)

    def matrix_type(self, mir):
        token = secret_alnum_string()
        with TemporaryDirectory(ensure_exists=False) as dir:
            with self.fs.open(dir + '/in', 'wb') as infile:
                write_int(infile, ServiceBackend.MATRIX_TABLE_TYPE)
                write_str(infile, tmp_dir())
                write_str(infile, self.render(mir))

            batch_attributes = self.batch_attributes
            if 'name' not in batch_attributes:
                batch_attributes = {**batch_attributes, 'name': 'matrix_type(...)'}
            bb = self.bc.create_batch(token=token, attributes=batch_attributes)

            j = bb.create_jvm_job([
                'is.hail.backend.service.ServiceBackendSocketAPI2',
                os.environ['HAIL_SHA'],
                os.environ['HAIL_JAR_URL'],
                batch_attributes['name'],
                dir + '/in',
                dir + '/out',
            ], mount_tokens=True)
            b = bb.submit(disable_progress_bar=self.disable_progress_bar)
            status = b.wait(disable_progress_bar=self.disable_progress_bar)
            if status['n_succeeded'] != 1:
                raise ValueError(f'batch failed {status} {j.log()}')


            with self.fs.open(dir + '/out', 'rb') as outfile:
                success = read_bool(outfile)
                if success:
                    s = read_str(outfile)
                    try:
                        return tmatrix._from_json(json.loads(s))
                    except json.decoder.JSONDecodeError as err:
                        raise ValueError(f'could not decode {s}') from err
                else:
                    jstacktrace = read_str(outfile)
                    raise FatalError(jstacktrace)

    def blockmatrix_type(self, bmir):
        token = secret_alnum_string()
        with TemporaryDirectory(ensure_exists=False) as dir:
            with self.fs.open(dir + '/in', 'wb') as infile:
                write_int(infile, ServiceBackend.BLOCK_MATRIX_TYPE)
                write_str(infile, tmp_dir())
                write_str(infile, self.render(bmir))

            batch_attributes = self.batch_attributes
            if 'name' not in batch_attributes:
                batch_attributes = {**batch_attributes, 'name': 'blockmatrix_type(...)'}
            bb = self.bc.create_batch(token=token, attributes=batch_attributes)

            j = bb.create_jvm_job([
                'is.hail.backend.service.ServiceBackendSocketAPI2',
                os.environ['HAIL_SHA'],
                os.environ['HAIL_JAR_URL'],
                batch_attributes['name'],
                dir + '/in',
                dir + '/out',
            ], mount_tokens=True)
            b = bb.submit(disable_progress_bar=self.disable_progress_bar)
            status = b.wait(disable_progress_bar=self.disable_progress_bar)
            if status['n_succeeded'] != 1:
                raise ValueError(f'batch failed {status} {j.log()}')


            with self.fs.open(dir + '/out', 'rb') as outfile:
                success = read_bool(outfile)
                if success:
                    s = read_str(outfile)
                    try:
                        return tblockmatrix._from_json(json.loads(s))
                    except json.decoder.JSONDecodeError as err:
                        raise ValueError(f'could not decode {s}') from err
                else:
                    jstacktrace = read_str(outfile)
                    raise FatalError(jstacktrace)

    def add_reference(self, config):
        raise NotImplementedError("ServiceBackend does not support 'add_reference'")

    def from_fasta_file(self, name, fasta_file, index_file, x_contigs, y_contigs, mt_contigs, par):
        raise NotImplementedError("ServiceBackend does not support 'from_fasta_file'")

    def remove_reference(self, name):
        raise NotImplementedError("ServiceBackend does not support 'remove_reference'")

    def get_reference(self, name):
        return async_to_blocking(self._async_get_reference(name))

    async def _async_get_reference(self, name):
        token = secret_alnum_string()
        with TemporaryDirectory(ensure_exists=False) as dir:
            with self.fs.open(dir + '/in', 'wb') as infile:
                write_int(infile, ServiceBackend.REFERENCE_GENOME)
                write_str(infile, tmp_dir())
                write_str(infile, name)

            batch_attributes = self.batch_attributes
            if 'name' not in batch_attributes:
                batch_attributes = {**batch_attributes, 'name': f'get_reference({name})'}
            bb = self.async_bc.create_batch(token=token, attributes=batch_attributes)

            j = bb.create_jvm_job([
                'is.hail.backend.service.ServiceBackendSocketAPI2',
                os.environ['HAIL_SHA'],
                os.environ['HAIL_JAR_URL'],
                batch_attributes['name'],
                dir + '/in',
                dir + '/out',
            ], mount_tokens=True)
            b = await bb.submit(disable_progress_bar=self.disable_progress_bar)
            status = await b.wait(disable_progress_bar=self.disable_progress_bar)
            if status['n_succeeded'] != 1:
                raise ValueError(f'batch failed {status} {await j.log()}')

            with self.fs.open(dir + '/out', 'rb') as outfile:
                success = read_bool(outfile)
                if success:
                    s = read_str(outfile)
                    try:
                        # FIXME: do we not have to parse the result?
                        return json.loads(s)
                    except json.decoder.JSONDecodeError as err:
                        raise ValueError(f'could not decode {s}') from err
                else:
                    jstacktrace = read_str(outfile)
                    raise FatalError(jstacktrace)

    def get_references(self, names):
        return async_to_blocking(self._async_get_references(names))

    async def _async_get_references(self, names):
        return await asyncio.gather(*[self._async_get_reference(name) for name in names])

    def load_references_from_dataset(self, path):
        token = secret_alnum_string()
        with TemporaryDirectory(ensure_exists=False) as dir:
            with self.fs.open(dir + '/in', 'wb') as infile:
                write_int(infile, ServiceBackend.LOAD_REFERENCES_FROM_DATASET)
                write_str(infile, tmp_dir())
                write_str(infile, self.billing_project)
                write_str(infile, self.bucket)
                write_str(infile, path)

            batch_attributes = self.batch_attributes
            if 'name' not in batch_attributes:
                batch_attributes = {**batch_attributes, 'name': 'load_references_from_dataset(...)'}
            bb = self.bc.create_batch(token=token, attributes=batch_attributes)

            j = bb.create_jvm_job([
                'is.hail.backend.service.ServiceBackendSocketAPI2',
                os.environ['HAIL_SHA'],
                os.environ['HAIL_JAR_URL'],
                batch_attributes['name'],
                dir + '/in',
                dir + '/out',
            ], mount_tokens=True)
            b = bb.submit(disable_progress_bar=self.disable_progress_bar)
            status = b.wait(disable_progress_bar=self.disable_progress_bar)
            if status['n_succeeded'] != 1:
                raise ValueError(f'batch failed {status} {j.log()}')


            with self.fs.open(dir + '/out', 'rb') as outfile:
                success = read_bool(outfile)
                if success:
                    s = read_str(outfile)
                    try:
                        # FIXME: do we not have to parse the result?
                        return json.loads(s)
                    except json.decoder.JSONDecodeError as err:
                        raise ValueError(f'could not decode {s}') from err
                else:
                    jstacktrace = read_str(outfile)
                    raise FatalError(jstacktrace)

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
