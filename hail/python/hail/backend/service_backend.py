from typing import Optional
import os
import aiohttp
import json
import warnings

from hail.utils import FatalError
from hail.expr.types import dtype, tvoid
from hail.expr.table_type import ttable
from hail.expr.matrix_type import tmatrix
from hail.expr.blockmatrix_type import tblockmatrix

from hailtop.config import get_deploy_config, get_user_config, DeployConfig
from hailtop.auth import service_auth_headers
from hailtop.utils import async_to_blocking, retry_transient_errors, secret_alnum_string, TransientError
from hail.ir.renderer import CSERenderer

from .backend import Backend
from ..hail_logging import PythonOnlyLogger
from ..fs.google_fs import GoogleCloudStorageFS


class ServiceSocket:
    def __init__(self, *, deploy_config: Optional[DeployConfig] = None):
        if not deploy_config:
            deploy_config = get_deploy_config()
        self.deploy_config = deploy_config
        self.url = deploy_config.base_url('query')
        self._session: Optional[aiohttp.ClientSession] = None

    async def session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(
                headers=service_auth_headers(self.deploy_config, 'query'))
        return self._session

    def close(self):
        if self._session is not None:
            async_to_blocking(self._session.close())
            self._session = None

    def handle_response(self, resp):
        if resp.type == aiohttp.WSMsgType.CLOSE:
            raise aiohttp.ServerDisconnectedError('Socket was closed by server. (code={resp.data})')
        if resp.type == aiohttp.WSMsgType.ERROR:
            raise FatalError(f'Error raised while waiting for response from server: {resp}.')
        assert resp.type == aiohttp.WSMsgType.TEXT, resp.type
        return resp.data

    async def async_request(self, endpoint, **data):
        data['token'] = secret_alnum_string()
        session = await self.session()
        async with session.ws_connect(f'{self.url}/api/v1alpha/{endpoint}') as socket:
            await socket.send_str(json.dumps(data))
            response = await socket.receive()
            await socket.send_str('bye')
            if response.type == aiohttp.WSMsgType.ERROR:
                raise ValueError(f'bad response: {endpoint}; {data}; {response}')
            if response.type in (aiohttp.WSMsgType.CLOSE,
                                 aiohttp.WSMsgType.CLOSED):
                warnings.warn(f'retrying after losing connection {endpoint}; {data}; {response}')
                raise TransientError()
            assert response.type == aiohttp.WSMsgType.TEXT
            result = json.loads(response.data)
            if result['status'] != 200:
                raise FatalError(f'Error from server: {result["value"]}')
            return result['value']

    def request(self, endpoint, **data):
        return async_to_blocking(retry_transient_errors(self.async_request, endpoint, **data))


class ServiceBackend(Backend):
    def __init__(self, billing_project: str = None, bucket: str = None, *, deploy_config=None,
                 skip_logging_configuration: bool = False):
        if billing_project is None:
            billing_project = get_user_config().get('batch', 'billing_project', fallback=None)
        if billing_project is None:
            billing_project = os.environ.get('HAIL_BILLING_PROJECT')
        if billing_project is None:
            raise ValueError(
                "No billing project.  Call 'init_service' with the billing "
                "project, set the HAIL_BILLING_PROJECT environment variable, "
                "or run 'hailctl config set batch/billing_project "
                "MY_BILLING_PROJECT'")
        self._billing_project = billing_project

        if bucket is None:
            bucket = get_user_config().get('batch', 'bucket', fallback=None)
        if bucket is None:
            bucket = os.environ.get('HAIL_BUCKET')
        if bucket is None:
            raise ValueError(
                'the bucket parameter of ServiceBackend must be set '
                'or run `hailctl config set batch/bucket '
                'MY_BUCKET`')
        self._bucket = bucket

        self._fs = None
        self._logger = PythonOnlyLogger(skip_logging_configuration)

        self.socket = ServiceSocket(deploy_config=deploy_config)

    @property
    def logger(self):
        return self._logger

    @property
    def fs(self) -> GoogleCloudStorageFS:
        if self._fs is None:
            from hail.fs.google_fs import GoogleCloudStorageFS
            self._fs = GoogleCloudStorageFS()
        return self._fs

    def stop(self):
        self.socket.close()

    def _render(self, ir):
        r = CSERenderer()
        assert len(r.jirs) == 0
        return r(ir)

    def execute(self, ir, timed=False):
        resp = self.socket.request('execute',
                                   code=self._render(ir),
                                   billing_project=self._billing_project,
                                   bucket=self._bucket)
        typ = dtype(resp['type'])
        if typ == tvoid:
            value = None
        else:
            value = typ._convert_from_json_na(resp['value'])
        # FIXME put back timings

        return (value, None) if timed else value

    def _request_type(self, ir, kind):
        code = self._render(ir)
        return self.socket.request(f'type/{kind}', code=code)

    def value_type(self, ir):
        resp = self._request_type(ir, 'value')
        return dtype(resp)

    def table_type(self, tir):
        resp = self._request_type(tir, 'table')
        return ttable._from_json(resp)

    def matrix_type(self, mir):
        resp = self._request_type(mir, 'matrix')
        return tmatrix._from_json(resp)

    def blockmatrix_type(self, bmir):
        resp = self._request_type(bmir, 'blockmatrix')
        return tblockmatrix._from_json(resp)

    def add_reference(self, config):
        raise NotImplementedError("ServiceBackend does not support 'add_reference'")

    def from_fasta_file(self, name, fasta_file, index_file, x_contigs, y_contigs, mt_contigs, par):
        raise NotImplementedError("ServiceBackend does not support 'from_fasta_file'")

    def remove_reference(self, name):
        raise NotImplementedError("ServiceBackend does not support 'remove_reference'")

    def get_reference(self, name):
        return self.socket.request('references/get', name=name)

    def load_references_from_dataset(self, path):
        return self.socket.request('load_references_from_dataset',
                                   path=path,
                                   billing_project=self._billing_project,
                                   bucket=self._bucket)

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
