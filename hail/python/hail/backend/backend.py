import abc
import os

from hail.utils.java import *
from hail.expr.types import dtype
from hail.expr.table_type import *
from hail.expr.matrix_type import *
from hail.expr.blockmatrix_type import *
from hail.ir.renderer import Renderer
from hail.table import Table
from hail.matrixtable import MatrixTable
from hail.fs import HadoopFS

import requests

import pyspark


class Backend(abc.ABC):
    @abc.abstractmethod
    def execute(self, ir, timed=False):
        return

    @abc.abstractmethod
    def value_type(self, ir):
        return

    @abc.abstractmethod
    def table_type(self, tir):
        return

    @abc.abstractmethod
    def matrix_type(self, mir):
        return

    def persist_table(self, t, storage_level):
        return t

    def unpersist_table(self, t):
        return t

    def persist_matrix_table(self, mt, storage_level):
        return mt

    def unpersist_matrix_table(self, mt):
        return mt

    @abc.abstractmethod
    def add_reference(self, config):
        pass

    @abc.abstractmethod
    def from_fasta_file(self, name, fasta_file, index_file, x_contigs, y_contigs, mt_contigs, par):
        pass

    @abc.abstractmethod
    def remove_reference(self, name):
        pass

    @abc.abstractmethod
    def get_reference(self, name):
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

    @abc.abstractmethod
    def parse_vcf_metadata(self, path):
        pass

    @property
    @abc.abstractmethod
    def fs(self):
        pass


class SparkBackend(Backend):
    def __init__(self):
        self._fs = HadoopFS()

    @property
    def fs(self):
        return self._fs

    def _to_java_ir(self, ir):
        if not hasattr(ir, '_jir'):
            r = Renderer(stop_at_jir=True)
            # FIXME parse should be static
            ir._jir = ir.parse(r(ir), ir_map=r.jirs)
        return ir._jir

    def execute(self, ir, timed=False):
        result = json.loads(Env.hail().backend.spark.SparkBackend.executeJSON(self._to_java_ir(ir)))
        value = ir.typ._from_json(result['value'])
        timings = result['timings']

        return (value, timings) if timed else value

    def value_type(self, ir):
        jir = self._to_java_ir(ir)
        return dtype(jir.typ().toString())

    def table_type(self, tir):
        jir = self._to_java_ir(tir)
        return ttable._from_java(jir.typ())

    def matrix_type(self, mir):
        jir = self._to_java_ir(mir)
        return tmatrix._from_java(jir.typ())

    def persist_table(self, t, storage_level):
        return Table._from_java(self._to_java_ir(t._tir).pyPersist(storage_level))

    def unpersist_table(self, t):
        return Table._from_java(self._to_java_ir(t._tir).pyUnpersist())

    def persist_matrix_table(self, mt, storage_level):
        return MatrixTable._from_java(self._to_java_ir(mt._mir).pyPersist(storage_level))

    def unpersist_matrix_table(self, mt):
        return MatrixTable._from_java(self._to_java_ir(mt._mir).pyUnpersist())

    def blockmatrix_type(self, bmir):
        jir = self._to_java_ir(bmir)
        return tblockmatrix._from_java(jir.typ())

    def from_spark(self, df, key):
        return Table._from_java(Env.hail().table.Table.pyFromDF(df._jdf, key))

    def to_spark(self, t, flatten):
        t = t.expand_types()
        if flatten:
            t = t.flatten()
        return pyspark.sql.DataFrame(self._to_java_ir(t._tir).pyToDF(), Env.sql_context())

    def to_pandas(self, t, flatten):
        return self.to_spark(t, flatten).toPandas()

    def from_pandas(self, df, key):
        return Table.from_spark(Env.sql_context().createDataFrame(df), key)

    def add_reference(self, config):
        Env.hail().variant.ReferenceGenome.fromJSON(json.dumps(config))

    def from_fasta_file(self, name, fasta_file, index_file, x_contigs, y_contigs, mt_contigs, par):
        Env.hail().variant.ReferenceGenome.fromFASTAFile(
            Env.hc()._jhc,
            name, fasta_file, index_file, x_contigs, y_contigs, mt_contigs, par)

    def remove_reference(self, name):
        Env.hail().variant.ReferenceGenome.removeReference(name)

    def get_reference(self, name):
        return json.loads(Env.hail().variant.ReferenceGenome.getReference(name).toJSONString())

    def add_sequence(self, name, fasta_file, index_file):
        scala_object(Env.hail().variant, 'ReferenceGenome').addSequence(
            name, fasta_file, index_file)

    def remove_sequence(self, name):
        scala_object(Env.hail().variant, 'ReferenceGenome').removeSequence(name)

    def add_liftover(self, name, chain_file, dest_reference_genome):
        scala_object(Env.hail().variant, 'ReferenceGenome').referenceAddLiftover(
            name, chain_file, dest_reference_genome)

    def remove_liftover(self, name, dest_reference_genome):
        scala_object(Env.hail().variant, 'ReferenceGenome').referenceRemoveLiftover(
            name, dest_reference_genome)

    def parse_vcf_metadata(self, path):
        return json.loads(Env.hc()._jhc.pyParseVCFMetadataJSON(path))


class LocalBackend(Backend):
    def __init__(self):
        pass

    def _to_java_ir(self, ir):
        if not hasattr(ir, '_jir'):
            r = Renderer(stop_at_jir=True)
            # FIXME parse should be static
            ir._jir = ir.parse(r(ir), ir_map=r.jirs)
        return ir._jir

    def execute(self, ir, timed=False):
        result = json.loads(Env.hail().expr.ir.LocalBackend.executeJSON(self._to_java_ir(ir)))
        value = ir.typ._from_json(result['value'])
        timings = result['timings']
        return (value, timings) if timed else value


class ServiceBackend(Backend):
    def __init__(self, url, token=None, token_file=None):
        if token_file is not None and token is not None:
            raise ValueError('set only one of token_file and token')
        self.url = url
        self._fs = HadoopFS()
        if token is None:
            token_file = (token_file or
                          os.environ.get('HAIL_TOKEN_FILE') or
                          os.path.expanduser('~/.hail/token'))
            if not os.path.exists(token_file):
                raise ValueError(
                    f'cannot create a client without a token. no file was '
                    f'found at {token_file}')
            with open(token_file) as f:
                token = f.read()
        self.cookies = {'user': token}


    @property
    def fs(self):
        return self._fs

    def _render(self, ir):
        r = Renderer()
        assert len(r.jirs) == 0
        return r(ir)

    def execute(self, ir, timed=False):
        code = self._render(ir)
        resp = requests.post(f'{self.url}/execute', json=code, cookies=self.cookies)
        if resp.status_code == 400:
            resp_json = resp.json()
            raise FatalError(resp_json['message'])
        resp.raise_for_status()

        resp_json = resp.json()
        typ = dtype(resp_json['type'])
        result = json.loads(resp_json['result'])
        value = typ._from_json(result['value'])
        timings = result['timings']

        return (value, timings) if timed else value

    def _request_type(self, ir, kind):
        code = self._render(ir)
        resp = requests.post(f'{self.url}/type/{kind}', json=code, cookies=self.cookies)
        if resp.status_code == 400:
            resp_json = resp.json()
            raise FatalError(resp_json['message'])
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
        resp = requests.post(f'{self.url}/references/create', json=config, cookies=self.cookies)
        if resp.status_code == 400:
            resp_json = resp.json()
            raise FatalError(resp_json['message'])
        resp.raise_for_status()

    def from_fasta_file(self, name, fasta_file, index_file, x_contigs, y_contigs, mt_contigs, par):
        resp = requests.post(f'{self.url}/references/create/fasta', json={
            'name': name,
            'fasta_file': fasta_file,
            'index_file': index_file,
            'x_contigs': x_contigs,
            'y_contigs': y_contigs,
            'mt_contigs': mt_contigs,
            'par': par
        }, cookies=self.cookies)
        if resp.status_code == 400:
            resp_json = resp.json()
            raise FatalError(resp_json['message'])
        resp.raise_for_status()

    def remove_reference(self, name):
        resp = requests.delete(f'{self.url}/references/delete',
                               json={'name': name},
                               cookies=self.cookies)
        if resp.status_code == 400:
            resp_json = resp.json()
            raise FatalError(resp_json['message'])
        resp.raise_for_status()

    def get_reference(self, name):
        resp = requests.get(f'{self.url}/references/get',
                            json={'name': name},
                            cookies=self.cookies)
        if resp.status_code == 400:
            resp_json = resp.json()
            raise FatalError(resp_json['message'])
        resp.raise_for_status()
        return resp.json()

    def add_sequence(self, name, fasta_file, index_file):
        resp = requests.post(f'{self.url}/references/sequence/set',
                             json={'name': name, 'fasta_file': fasta_file, 'index_file': index_file},
                             cookies=self.cookies)
        if resp.status_code == 400:
            resp_json = resp.json()
            raise FatalError(resp_json['message'])
        resp.raise_for_status()

    def remove_sequence(self, name):
        resp = requests.delete(f'{self.url}/references/sequence/delete',
                               json={'name': name},
                               cookies=self.cookies)
        if resp.status_code == 400:
            resp_json = resp.json()
            raise FatalError(resp_json['message'])
        resp.raise_for_status()

    def add_liftover(self, name, chain_file, dest_reference_genome):
        resp = requests.post(f'{self.url}/references/liftover/add',
                             json={'name': name, 'chain_file': chain_file, 'dest_reference_genome': dest_reference_genome},
                             cookies=self.cookies)
        if resp.status_code == 400:
            resp_json = resp.json()
            raise FatalError(resp_json['message'])
        resp.raise_for_status()

    def remove_liftover(self, name, dest_reference_genome):
        resp = requests.delete(f'{self.url}/references/liftover/remove',
                               json={'name': name, 'dest_reference_genome': dest_reference_genome},
                               cookies=self.cookies)
        if resp.status_code == 400:
            resp_json = resp.json()
            raise FatalError(resp_json['message'])
        resp.raise_for_status()

    def parse_vcf_metadata(self, path):
        resp = requests.post(f'{self.url}/parse-vcf-metadata',
                             json={'path': path},
                             cookies=self.cookies)
        if resp.status_code == 400:
            resp_json = resp.json()
            raise FatalError(resp_json['message'])
        resp.raise_for_status()
        return resp.json()
