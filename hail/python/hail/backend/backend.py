import pkg_resources
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import abc
import os
import requests
import pyspark
from hail.utils.java import *
from hail.expr.types import dtype
from hail.expr.table_type import *
from hail.expr.matrix_type import *
from hail.expr.blockmatrix_type import *
from hail.ir.renderer import CSERenderer, Renderer
from hail.table import Table
from hail.matrixtable import MatrixTable
from hailtop.utils import sync_retry_transient_errors


class Backend(abc.ABC):
    @abc.abstractmethod
    def stop(self):
        pass

    @abc.abstractmethod
    def execute(self, ir, timed=False):
        pass

    @abc.abstractmethod
    def value_type(self, ir):
        pass

    @abc.abstractmethod
    def table_type(self, tir):
        pass

    @abc.abstractmethod
    def matrix_type(self, mir):
        pass

    @abc.abstractmethod
    def add_reference(self, config):
        pass

    @abc.abstractmethod
    def load_references_from_dataset(self, path):
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

    @abc.abstractmethod
    def index_bgen(self, files, index_file_map, rg, contig_recoding, skip_invalid_loci):
        pass

    def persist_table(self, t, storage_level):
        return t

    def unpersist_table(self, t):
        return t

    def persist_matrix_table(self, mt, storage_level):
        return mt

    def unpersist_matrix_table(self, mt):
        return mt


class SparkBackend(Backend):
    def __init__(self, idempotent, sc, spark_conf, app_name, master,
                 local, log, quiet, append, min_block_size,
                 branching_factor, tmp_dir, optimizer_iterations):
        if pkg_resources.resource_exists(__name__, "hail-all-spark.jar"):
            hail_jar_path = pkg_resources.resource_filename(__name__, "hail-all-spark.jar")
            assert os.path.exists(hail_jar_path), f'{hail_jar_path} does not exist'
            conf = SparkConf()

            base_conf = spark_conf or {}
            for k, v in base_conf.items():
                conf.set(k, v)

            jars = [hail_jar_path]

            if os.environ.get('HAIL_SPARK_MONITOR'):
                import sparkmonitor
                jars.append(os.path.join(os.path.dirname(sparkmonitor.__file__), 'listener.jar'))
                conf.set("spark.extraListeners", "sparkmonitor.listener.JupyterSparkMonitorListener")

            conf.set('spark.jars', ','.join(jars))
            conf.set('spark.driver.extraClassPath', ','.join(jars))
            conf.set('spark.executor.extraClassPath', './hail-all-spark.jar')
            if sc is None:
                SparkContext._ensure_initialized(conf=conf)
            else:
                import warnings
                warnings.warn(
                    'pip-installed Hail requires additional configuration options in Spark referring\n'
                    '  to the path to the Hail Python module directory HAIL_DIR,\n'
                    '  e.g. /path/to/python/site-packages/hail:\n'
                    '    spark.jars=HAIL_DIR/hail-all-spark.jar\n'
                    '    spark.driver.extraClassPath=HAIL_DIR/hail-all-spark.jar\n'
                    '    spark.executor.extraClassPath=./hail-all-spark.jar')
        else:
            SparkContext._ensure_initialized()

        self._gateway = SparkContext._gateway
        self._jvm = SparkContext._jvm

        Env._jvm = self._jvm
        Env._gateway = self._gateway

        # hail package
        hail = getattr(self._jvm, 'is').hail

        jsc = sc._jsc.sc() if sc else None

        if idempotent:
            self._jbackend = hail.backend.spark.SparkBackend.getOrCreate(
                jsc, app_name, master, local, True, min_block_size)
            self._jhc = hail.HailContext.getOrCreate(
                self._jbackend, log, True, append, branching_factor, tmp_dir, optimizer_iterations)
        else:
            self._jbackend = hail.backend.spark.SparkBackend.apply(
                jsc, app_name, master, local, True, min_block_size)
            self._jhc = hail.HailContext.apply(
                self._jbackend, log, True, append, branching_factor, tmp_dir, optimizer_iterations)

        self._jsc = self._jhc.sc()
        self.sc = sc if sc else SparkContext(gateway=self._gateway, jsc=self._jvm.JavaSparkContext(self._jsc))
        self._jspark_session = self._jbackend.sparkSession()
        self._spark_session = SparkSession(self.sc, self._jspark_session)

        from hail.context import version

        py_version = version()
        jar_version = self._jhc.version()
        if jar_version != py_version:
            raise RuntimeError(f"Hail version mismatch between JAR and Python library\n"
                               f"  JAR:    {jar_version}\n"
                               f"  Python: {py_version}")

        self._fs = None

        if not quiet:
            sys.stderr.write('Running on Apache Spark version {}\n'.format(self.sc.version))
            if self._jsc.uiWebUrl().isDefined():
                sys.stderr.write('SparkUI available at {}\n'.format(self._jsc.uiWebUrl().get()))

            connect_logger('localhost', 12888)

            self._jbackend.startProgressBar()

    def stop(self):
        self._jhc.stop()
        self._jhc = None
        self.sc.stop()
        self.sc = None

    @property
    def fs(self):
        if self._fs is None:
            from hail.fs.hadoop_fs import HadoopFS
            self._fs = HadoopFS()
        return self._fs

    def _to_java_ir(self, ir):
        if not hasattr(ir, '_jir'):
            r = CSERenderer(stop_at_jir=True)
            # FIXME parse should be static
            ir._jir = ir.parse(r(ir), ir_map=r.jirs)
        return ir._jir

    def execute(self, ir, timed=False):
        jir = self._to_java_ir(ir)
        result = json.loads(self._jhc.backend().executeJSON(jir))
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
        return Table._from_java(Env.jutils().pyFromDF(df._jdf, key))

    def to_spark(self, t, flatten):
        t = t.expand_types()
        if flatten:
            t = t.flatten()
        return pyspark.sql.DataFrame(self._to_java_ir(t._tir).pyToDF(), Env.spark_session()._wrapped)

    def to_pandas(self, t, flatten):
        return self.to_spark(t, flatten).toPandas()

    def from_pandas(self, df, key):
        return Table.from_spark(Env.spark_session().createDataFrame(df), key)

    def add_reference(self, config):
        Env.hail().variant.ReferenceGenome.fromJSON(json.dumps(config))

    def load_references_from_dataset(self, path):
        return json.loads(Env.hail().variant.ReferenceGenome.fromHailDataset(path))

    def from_fasta_file(self, name, fasta_file, index_file, x_contigs, y_contigs, mt_contigs, par):
        Env.hail().variant.ReferenceGenome.fromFASTAFile(
            self._jhc,
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
        return json.loads(self._jhc.pyParseVCFMetadataJSON(path))

    def index_bgen(self, files, index_file_map, rg, contig_recoding, skip_invalid_loci):
        self._jbackend.pyIndexBgen(files, index_file_map, joption(rg), contig_recoding, skip_invalid_loci)


class ServiceBackend(Backend):
    def __init__(self, deploy_config=None):
        from hailtop.config import get_deploy_config
        from hailtop.auth import service_auth_headers

        if not deploy_config:
            deploy_config = get_deploy_config()
        self.url = deploy_config.base_url('query')
        self.headers = service_auth_headers(deploy_config, 'query')
        self._fs = None

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
        resp = sync_retry_transient_errors(
            requests.post,
            f'{self.url}/execute', json=code, headers=self.headers)
        if resp.status_code == 400:
            resp_json = resp.json()
            raise FatalError(resp_json['message'])
        resp.raise_for_status()
        resp_json = resp.json()
        typ = dtype(resp_json['type'])
        value = typ._convert_from_json_na(resp_json['value'])
        # FIXME put back timings

        return (value, None) if timed else value

    def _request_type(self, ir, kind):
        code = self._render(ir)
        resp = sync_retry_transient_errors(
            requests.post,
            f'{self.url}/type/{kind}', json=code, headers=self.headers)
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
        resp = sync_retry_transient_errors(
            requests.post,
            f'{self.url}/references/create', json=config, headers=self.headers)
        if resp.status_code == 400:
            resp_json = resp.json()
            raise FatalError(resp_json['message'])
        resp.raise_for_status()

    def from_fasta_file(self, name, fasta_file, index_file, x_contigs, y_contigs, mt_contigs, par):
        resp = sync_retry_transient_errors(
            requests.post,
            f'{self.url}/references/create/fasta', json={
            'name': name,
            'fasta_file': fasta_file,
            'index_file': index_file,
            'x_contigs': x_contigs,
            'y_contigs': y_contigs,
            'mt_contigs': mt_contigs,
            'par': par
        }, headers=self.headers)
        if resp.status_code == 400:
            resp_json = resp.json()
            raise FatalError(resp_json['message'])
        resp.raise_for_status()

    def remove_reference(self, name):
        resp = sync_retry_transient_errors(
            requests.delete,
            f'{self.url}/references/delete',
            json={'name': name},
            headers=self.headers)
        if resp.status_code == 400:
            resp_json = resp.json()
            raise FatalError(resp_json['message'])
        resp.raise_for_status()

    def get_reference(self, name):
        resp = sync_retry_transient_errors(
            requests.get, f'{self.url}/references/get',
            json={'name': name},
            headers=self.headers)
        if resp.status_code == 400:
            resp_json = resp.json()
            raise FatalError(resp_json['message'])
        resp.raise_for_status()
        return resp.json()

    def load_references_from_dataset(self, path):
        # FIXME
        return []

    def add_sequence(self, name, fasta_file, index_file):
        resp = sync_retry_transient_errors(
            requests.post,
            f'{self.url}/references/sequence/set',
            json={'name': name, 'fasta_file': fasta_file, 'index_file': index_file},
            headers=self.headers)
        if resp.status_code == 400:
            resp_json = resp.json()
            raise FatalError(resp_json['message'])
        resp.raise_for_status()

    def remove_sequence(self, name):
        resp = sync_retry_transient_errors(
            requests.delete,
            f'{self.url}/references/sequence/delete',
            json={'name': name},
            headers=self.headers)
        if resp.status_code == 400:
            resp_json = resp.json()
            raise FatalError(resp_json['message'])
        resp.raise_for_status()

    def add_liftover(self, name, chain_file, dest_reference_genome):
        resp = sync_retry_transient_errors(
            requests.post,
            f'{self.url}/references/liftover/add',
            json={'name': name, 'chain_file': chain_file,
                  'dest_reference_genome': dest_reference_genome},
            headers=self.headers)
        if resp.status_code == 400:
            resp_json = resp.json()
            raise FatalError(resp_json['message'])
        resp.raise_for_status()

    def remove_liftover(self, name, dest_reference_genome):
        resp = sync_retry_transient_errors(
            requests.delete,
            f'{self.url}/references/liftover/remove',
            json={'name': name, 'dest_reference_genome': dest_reference_genome},
            headers=self.headers)
        if resp.status_code == 400:
            resp_json = resp.json()
            raise FatalError(resp_json['message'])
        resp.raise_for_status()

    def parse_vcf_metadata(self, path):
        resp = sync_retry_transient_errors(
            requests.post,
            f'{self.url}/parse-vcf-metadata',
            json={'path': path},
            headers=self.headers)
        if resp.status_code == 400:
            resp_json = resp.json()
            raise FatalError(resp_json['message'])
        resp.raise_for_status()
        return resp.json()

    def index_bgen(self, files, index_file_map, rg, contig_recoding, skip_invalid_loci):
        resp = requests.post(f'{self.url}/index-bgen',
                             json={
                                 'files': files,
                                 'index_file_map': index_file_map,
                                 'rg': rg,
                                 'contig_recoding': contig_recoding,
                                 'skip_invalid_loci': skip_invalid_loci
                             },
                             headers=self.headers)
        if resp.status_code == 400:
            resp_json = resp.json()
            raise FatalError(resp_json['message'])
        resp.raise_for_status()
        return resp.json()
