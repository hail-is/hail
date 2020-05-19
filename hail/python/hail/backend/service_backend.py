import requests

from hail.utils import FatalError
from hail.expr.types import dtype
from hail.expr.table_type import ttable
from hail.expr.matrix_type import tmatrix
from hail.expr.blockmatrix_type import tblockmatrix

from hailtop.config import get_deploy_config
from hailtop.auth import service_auth_headers
from hailtop.utils import retry_response_returning_functions
from hail.ir.renderer import CSERenderer

from .backend import Backend
from ..hail_logging import PythonOnlyLogger


class ServiceBackend(Backend):
    def __init__(self, deploy_config=None, skip_logging_configuration=False):
        if not deploy_config:
            deploy_config = get_deploy_config()
        self.url = deploy_config.base_url('query')
        self.headers = service_auth_headers(deploy_config, 'query')
        self._fs = None
        self._logger = PythonOnlyLogger(skip_logging_configuration)

    @property
    def logger(self):
        return self._logger

    @property
    def fs(self):
        if self._fs is None:
            from hail.fs.google_fs import GoogleCloudStorageFS
            self._fs = GoogleCloudStorageFS()
        return self._fs

    def stop(self):
        pass

    def _render(self, ir):
        r = CSERenderer()
        assert len(r.jirs) == 0
        return r(ir)

    def execute(self, ir, timed=False):
        code = self._render(ir)
        resp = retry_response_returning_functions(
            requests.post,
            f'{self.url}/execute', json=code, headers=self.headers)
        if resp.status_code == 400 or resp.status_code == 500:
            raise FatalError(resp.text)
        resp.raise_for_status()
        resp_json = resp.json()
        typ = dtype(resp_json['type'])
        value = typ._convert_from_json_na(resp_json['value'])
        # FIXME put back timings

        return (value, None) if timed else value

    def _request_type(self, ir, kind):
        code = self._render(ir)
        resp = retry_response_returning_functions(
            requests.post,
            f'{self.url}/type/{kind}', json=code, headers=self.headers)
        if resp.status_code == 400 or resp.status_code == 500:
            raise FatalError(resp.text)
        resp.raise_for_status()

        return resp.json()

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
        resp = retry_response_returning_functions(
            requests.post,
            f'{self.url}/references/create', json=config, headers=self.headers)
        if resp.status_code == 400 or resp.status_code == 500:
            resp_json = resp.json()
            raise FatalError(resp_json['message'])
        resp.raise_for_status()

    def from_fasta_file(self, name, fasta_file, index_file, x_contigs, y_contigs, mt_contigs, par):
        resp = retry_response_returning_functions(
            requests.post,
            f'{self.url}/references/create/fasta',
            json={
                'name': name,
                'fasta_file': fasta_file,
                'index_file': index_file,
                'x_contigs': x_contigs,
                'y_contigs': y_contigs,
                'mt_contigs': mt_contigs,
                'par': par
            }, headers=self.headers)
        if resp.status_code == 400 or resp.status_code == 500:
            resp_json = resp.json()
            raise FatalError(resp_json['message'])
        resp.raise_for_status()

    def remove_reference(self, name):
        resp = retry_response_returning_functions(
            requests.delete,
            f'{self.url}/references/delete',
            json={'name': name},
            headers=self.headers)
        if resp.status_code == 400 or resp.status_code == 500:
            resp_json = resp.json()
            raise FatalError(resp_json['message'])
        resp.raise_for_status()

    def get_reference(self, name):
        resp = retry_response_returning_functions(
            requests.get,
            f'{self.url}/references/get',
            json={'name': name},
            headers=self.headers)
        if resp.status_code == 400 or resp.status_code == 500:
            resp_json = resp.json()
            raise FatalError(resp_json['message'])
        resp.raise_for_status()
        return resp.json()

    def load_references_from_dataset(self, path):
        # FIXME
        return []

    def add_sequence(self, name, fasta_file, index_file):
        resp = retry_response_returning_functions(
            requests.post,
            f'{self.url}/references/sequence/set',
            json={'name': name, 'fasta_file': fasta_file, 'index_file': index_file},
            headers=self.headers)
        if resp.status_code == 400 or resp.status_code == 500:
            resp_json = resp.json()
            raise FatalError(resp_json['message'])
        resp.raise_for_status()

    def remove_sequence(self, name):
        resp = retry_response_returning_functions(
            requests.delete,
            f'{self.url}/references/sequence/delete',
            json={'name': name},
            headers=self.headers)
        if resp.status_code == 400 or resp.status_code == 500:
            resp_json = resp.json()
            raise FatalError(resp_json['message'])
        resp.raise_for_status()

    def add_liftover(self, name, chain_file, dest_reference_genome):
        resp = retry_response_returning_functions(
            requests.post,
            f'{self.url}/references/liftover/add',
            json={'name': name, 'chain_file': chain_file,
                  'dest_reference_genome': dest_reference_genome},
            headers=self.headers)
        if resp.status_code == 400 or resp.status_code == 500:
            resp_json = resp.json()
            raise FatalError(resp_json['message'])
        resp.raise_for_status()

    def remove_liftover(self, name, dest_reference_genome):
        resp = retry_response_returning_functions(
            requests.delete,
            f'{self.url}/references/liftover/remove',
            json={'name': name, 'dest_reference_genome': dest_reference_genome},
            headers=self.headers)
        if resp.status_code == 400 or resp.status_code == 500:
            resp_json = resp.json()
            raise FatalError(resp_json['message'])
        resp.raise_for_status()

    def parse_vcf_metadata(self, path):
        resp = retry_response_returning_functions(
            requests.post,
            f'{self.url}/parse-vcf-metadata',
            json={'path': path},
            headers=self.headers)
        if resp.status_code == 400 or resp.status_code == 500:
            resp_json = resp.json()
            raise FatalError(resp_json['message'])
        resp.raise_for_status()
        return resp.json()

    def index_bgen(self, files, index_file_map, rg, contig_recoding, skip_invalid_loci):
        resp = retry_response_returning_functions(
            requests.post,
            f'{self.url}/index-bgen',
            json={
                'files': files,
                'index_file_map': index_file_map,
                'rg': rg,
                'contig_recoding': contig_recoding,
                'skip_invalid_loci': skip_invalid_loci
            },
            headers=self.headers)
        if resp.status_code == 400 or resp.status_code == 500:
            resp_json = resp.json()
            raise FatalError(resp_json['message'])
        resp.raise_for_status()
        return resp.json()
