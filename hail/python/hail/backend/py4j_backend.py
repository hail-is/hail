from typing import Mapping
import abc
import json

import py4j
import py4j.java_gateway

import hail
from hail.expr import construct_expr
from hail.ir import JavaIR, finalize_randomness
from hail.ir.renderer import CSERenderer
from hail.utils.java import FatalError, Env
from hail.expr.blockmatrix_type import tblockmatrix
from hail.expr.matrix_type import tmatrix
from hail.expr.table_type import ttable
from hail.expr.types import dtype

from .backend import Backend, fatal_error_from_java_error_triplet


def handle_java_exception(f):
    def deco(*args, **kwargs):
        import pyspark
        try:
            return f(*args, **kwargs)
        except py4j.protocol.Py4JJavaError as e:
            s = e.java_exception.toString()

            # py4j catches NoSuchElementExceptions to stop array iteration
            if s.startswith('java.util.NoSuchElementException'):
                raise

            tpl = Env.jutils().handleForPython(e.java_exception)
            deepest, full, error_id = tpl._1(), tpl._2(), tpl._3()
            raise fatal_error_from_java_error_triplet(deepest, full, error_id) from None
        except pyspark.sql.utils.CapturedException as e:
            raise FatalError('%s\n\nJava stack trace:\n%s\n'
                             'Hail version: %s\n'
                             'Error summary: %s' % (e.desc, e.stackTrace, hail.__version__, e.desc)) from None

    return deco


class Py4JBackend(Backend):
    _jbackend: py4j.java_gateway.JavaObject

    @abc.abstractmethod
    def __init__(self):
        super(Py4JBackend, self).__init__()
        import base64

        def decode_bytearray(encoded):
            return base64.standard_b64decode(encoded)

        # By default, py4j's version of this function does extra
        # work to support python 2. This eliminates that.
        py4j.protocol.decode_bytearray = decode_bytearray

    @abc.abstractmethod
    def jvm(self):
        pass

    @abc.abstractmethod
    def hail_package(self):
        pass

    @abc.abstractmethod
    def utils_package_object(self):
        pass

    def execute(self, ir, timed=False):
        jir = self._to_java_value_ir(ir)
        stream_codec = '{"name":"StreamBufferSpec"}'
        # print(self._hail_package.expr.ir.Pretty.apply(jir, True, -1))
        try:
            result_tuple = self._jbackend.executeEncode(jir, stream_codec, timed)
            (result, timings) = (result_tuple._1(), result_tuple._2())
            value = ir.typ._from_encoding(result)

            return (value, timings) if timed else value
        except FatalError as e:
            raise e.maybe_user_error(ir) from None

    async def _async_execute(self, ir, timed=False):
        raise NotImplementedError('no async available in Py4JBackend')

    def persist_expression(self, expr):
        return construct_expr(
            JavaIR(self._jbackend.executeLiteral(self._to_java_value_ir(expr._ir))),
            expr.dtype
        )

    def set_flags(self, **flags: Mapping[str, str]):
        available = self._jbackend.availableFlags()
        invalid = []
        for flag, value in flags.items():
            if flag in available:
                self._jbackend.setFlag(flag, value)
            else:
                invalid.append(flag)
        if len(invalid) != 0:
            raise FatalError("Flags {} not valid. Valid flags: \n    {}"
                             .format(', '.join(invalid), '\n    '.join(available)))

    def get_flags(self, *flags) -> Mapping[str, str]:
        return {flag: self._jbackend.getFlag(flag) for flag in flags}

    def _add_reference_to_scala_backend(self, rg):
        self._jbackend.pyAddReference(json.dumps(rg._config))

    def _remove_reference_from_scala_backend(self, name):
        self._jbackend.pyRemoveReference(name)

    def from_fasta_file(self, name, fasta_file, index_file, x_contigs, y_contigs, mt_contigs, par):
        return json.loads(self._jbackend.pyFromFASTAFile(name, fasta_file, index_file, x_contigs, y_contigs, mt_contigs, par))

    def load_references_from_dataset(self, path):
        return json.loads(self._jbackend.pyLoadReferencesFromDataset(path))

    def add_sequence(self, name, fasta_file, index_file):
        self._jbackend.pyAddSequence(name, fasta_file, index_file)

    def remove_sequence(self, name):
        self._jbackend.pyRemoveSequence(name)

    def add_liftover(self, name, chain_file, dest_reference_genome):
        self._jbackend.pyAddLiftover(name, chain_file, dest_reference_genome)

    def remove_liftover(self, name, dest_reference_genome):
        self._jbackend.pyRemoveLiftover(name, dest_reference_genome)

    def parse_vcf_metadata(self, path):
        return json.loads(self._jhc.pyParseVCFMetadataJSON(self._jbackend.fs(), path))

    def index_bgen(self, files, index_file_map, referenceGenomeName, contig_recoding, skip_invalid_loci):
        self._jbackend.pyIndexBgen(files, index_file_map, referenceGenomeName, contig_recoding, skip_invalid_loci)

    def import_fam(self, path: str, quant_pheno: bool, delimiter: str, missing: str):
        return json.loads(self._jbackend.pyImportFam(path, quant_pheno, delimiter, missing))

    def _to_java_ir(self, ir, parse):
        if not hasattr(ir, '_jir'):
            r = CSERenderer(stop_at_jir=True)
            # FIXME parse should be static
            ir._jir = parse(r(finalize_randomness(ir)), ir_map=r.jirs)
        return ir._jir

    def _parse_value_ir(self, code, ref_map={}, ir_map={}):
        return self._jbackend.parse_value_ir(
            code,
            {k: t._parsable_string() for k, t in ref_map.items()},
            ir_map)

    def _parse_table_ir(self, code, ir_map={}):
        return self._jbackend.parse_table_ir(code, ir_map)

    def _parse_matrix_ir(self, code, ir_map={}):
        return self._jbackend.parse_matrix_ir(code, ir_map)

    def _parse_blockmatrix_ir(self, code, ir_map={}):
        return self._jbackend.parse_blockmatrix_ir(code, ir_map)

    def _to_java_value_ir(self, ir):
        return self._to_java_ir(ir, self._parse_value_ir)

    def _to_java_table_ir(self, ir):
        return self._to_java_ir(ir, self._parse_table_ir)

    def _to_java_matrix_ir(self, ir):
        return self._to_java_ir(ir, self._parse_matrix_ir)

    def _to_java_blockmatrix_ir(self, ir):
        return self._to_java_ir(ir, self._parse_blockmatrix_ir)

    def value_type(self, ir):
        jir = self._to_java_value_ir(ir)
        return dtype(jir.typ().toString())

    def table_type(self, tir):
        jir = self._to_java_table_ir(tir)
        return ttable._from_java(jir.typ())

    def matrix_type(self, mir):
        jir = self._to_java_matrix_ir(mir)
        return tmatrix._from_java(jir.typ())

    def blockmatrix_type(self, bmir):
        jir = self._to_java_blockmatrix_ir(bmir)
        return tblockmatrix._from_java(jir.typ())

    @property
    def requires_lowering(self):
        return True
