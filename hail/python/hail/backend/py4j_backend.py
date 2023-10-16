from typing import Mapping
import abc
import json

import requests
import py4j
import py4j.java_gateway

import hail
from hail.expr import construct_expr
from hail.ir import finalize_randomness, JavaIR
from hail.ir.renderer import CSERenderer
from hail.utils.java import FatalError, Env

from .backend import ActionTag, Backend, fatal_error_from_java_error_triplet

import http.client
http.client._MAXLINE = 2 ** 20


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


action_routes = {
    ActionTag.VALUE_TYPE: '/value/type',
    ActionTag.TABLE_TYPE: '/table/type',
    ActionTag.MATRIX_TABLE_TYPE: '/matrixtable/type',
    ActionTag.BLOCK_MATRIX_TYPE: '/blockmatrix/type',
    ActionTag.LOAD_REFERENCES_FROM_DATASET: '/references/load',
    ActionTag.FROM_FASTA_FILE: '/references/from_fasta',
    ActionTag.EXECUTE: '/execute',
    ActionTag.PARSE_VCF_METADATA: '/vcf/metadata/parse',
    ActionTag.IMPORT_FAM: '/fam/import',
}


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
        self._requests_session = requests.Session()

    @abc.abstractmethod
    def jvm(self):
        pass

    @abc.abstractmethod
    def hail_package(self):
        pass

    @abc.abstractmethod
    def utils_package_object(self):
        pass

    def persist_expression(self, expr):
        assert self._jbackend
        t = expr.dtype
        return construct_expr(
            JavaIR(t, self._jbackend.executeLiteral(self._render_ir(expr._ir))),
            t
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

    def add_sequence(self, name, fasta_file, index_file):
        self._jbackend.pyAddSequence(name, fasta_file, index_file)

    def remove_sequence(self, name):
        self._jbackend.pyRemoveSequence(name)

    def add_liftover(self, name, chain_file, dest_reference_genome):
        self._jbackend.pyAddLiftover(name, chain_file, dest_reference_genome)

    def remove_liftover(self, name, dest_reference_genome):
        self._jbackend.pyRemoveLiftover(name, dest_reference_genome)

    def index_bgen(self, files, index_file_map, referenceGenomeName, contig_recoding, skip_invalid_loci):
        self._jbackend.pyIndexBgen(files, index_file_map, referenceGenomeName, contig_recoding, skip_invalid_loci)

    def _to_java_ir(self, ir, parse):
        if not hasattr(ir, '_jir'):
            r = CSERenderer()
            # FIXME parse should be static
            ir._jir = parse(r(finalize_randomness(ir)))
        return ir._jir

    def _parse_value_ir(self, code, ref_map={}):
        return self._jbackend.parse_value_ir(
            code,
            {k: t._parsable_string() for k, t in ref_map.items()},
        )

    def _parse_table_ir(self, code):
        return self._jbackend.parse_table_ir(code)

    def _parse_matrix_ir(self, code):
        return self._jbackend.parse_matrix_ir(code)

    def _parse_blockmatrix_ir(self, code):
        return self._jbackend.parse_blockmatrix_ir(code)

    def _to_java_value_ir(self, ir):
        return self._to_java_ir(ir, self._parse_value_ir)

    def _to_java_blockmatrix_ir(self, ir):
        return self._to_java_ir(ir, self._parse_blockmatrix_ir)

    @property
    def requires_lowering(self):
        return True
