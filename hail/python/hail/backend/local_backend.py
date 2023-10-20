from typing import Optional, Union, Tuple, List
import os
import sys

from py4j.java_gateway import JavaGateway, GatewayParameters, launch_gateway

from hail.ir.renderer import CSERenderer
from hail.ir import finalize_randomness
from .py4j_backend import Py4JBackend, uninstall_exception_handler
from .backend import local_jar_information
from ..expr import Expression
from ..expr.types import HailType

from hailtop.utils import find_spark_home
from hailtop.fs.router_fs import RouterFS
from hailtop.aiotools.validators import validate_file


class LocalBackend(Py4JBackend):
    def __init__(self, tmpdir, log, quiet, append, branching_factor,
                 skip_logging_configuration, optimizer_iterations,
                 jvm_heap_size,
                 *,
                 gcs_requester_pays_project: Optional[str] = None,
                 gcs_requester_pays_buckets: Optional[str] = None
                 ):
        assert gcs_requester_pays_project is not None or gcs_requester_pays_buckets is None

        spark_home = find_spark_home()
        hail_jar_path = os.environ.get('HAIL_JAR')
        if hail_jar_path is None:
            try:
                local_jar_info = local_jar_information()
                hail_jar_path = local_jar_info.path
                extra_classpath = ':'.join([f'{spark_home}/jars/*', hail_jar_path, *local_jar_info.extra_classpath])
            except ValueError:
                raise RuntimeError('local backend requires a packaged jar or HAIL_JAR to be set')
        else:
            extra_classpath = ':'.join([f'{spark_home}/jars/*', hail_jar_path])

        jvm_opts = []
        if jvm_heap_size is not None:
            jvm_opts.append(f'-Xmx{jvm_heap_size}')

        port = launch_gateway(
            redirect_stdout=sys.stdout,
            redirect_stderr=sys.stderr,
            java_path=None,
            javaopts=jvm_opts,
            jarpath=f'{spark_home}/jars/py4j-0.10.9.5.jar',
            classpath=extra_classpath,
            die_on_exit=True)
        self._gateway = JavaGateway(
            gateway_parameters=GatewayParameters(port=port, auto_convert=True))

        hail_package = getattr(self._gateway.jvm, 'is').hail

        jbackend = hail_package.backend.local.LocalBackend.apply(
            tmpdir,
            gcs_requester_pays_project,
            gcs_requester_pays_buckets,
            log,
            True,
            append,
            skip_logging_configuration
        )
        jhc = hail_package.HailContext.apply(
            jbackend,
            branching_factor,
            optimizer_iterations
        )

        super(LocalBackend, self).__init__(self._gateway.jvm, jbackend, jhc)

        self._fs = RouterFS()
        self._logger = None

        self._initialize_flags({})

    def validate_file(self, uri: str) -> None:
        validate_file(uri, self._fs.afs)

    def register_ir_function(self,
                             name: str,
                             type_parameters: Union[Tuple[HailType, ...], List[HailType]],
                             value_parameter_names: Union[Tuple[str, ...], List[str]],
                             value_parameter_types: Union[Tuple[HailType, ...], List[HailType]],
                             return_type: HailType,
                             body: Expression):
        r = CSERenderer()
        code = r(finalize_randomness(body._ir))
        jbody = self._parse_value_ir(code, ref_map=dict(zip(value_parameter_names, value_parameter_types)))
        self._registered_ir_function_names.add(name)

        self.hail_package().expr.ir.functions.IRFunctionRegistry.pyRegisterIR(
            name,
            [ta._parsable_string() for ta in type_parameters],
            value_parameter_names,
            [pt._parsable_string() for pt in value_parameter_types],
            return_type._parsable_string(),
            jbody)

    def stop(self):
        super().stop()
        self._gateway.shutdown()
        uninstall_exception_handler()

    @property
    def fs(self):
        return self._fs

    @property
    def requires_lowering(self):
        return True
