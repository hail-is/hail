import glob
import os
import sys
from contextlib import ExitStack
from typing import Dict, List, Optional, Tuple, Union

from py4j.java_gateway import GatewayParameters, JavaGateway, launch_gateway

from hail.ir import finalize_randomness
from hail.ir.renderer import CSERenderer
from hail.version import __version__
from hailtop.aiocloud.aiogoogle import GCSRequesterPaysConfiguration
from hailtop.aiotools.validators import validate_file
from hailtop.fs.router_fs import RouterFS
from hailtop.utils import async_to_blocking, find_spark_home

from ..expr import Expression
from ..expr.types import HailType
from ..utils.java import scala_object, scala_package_object
from .backend import local_jar_information
from .py4j_backend import Py4JBackend, connect_logger, uninstall_exception_handler


class LocalBackend(Py4JBackend):
    def __init__(
        self,
        tmpdir,
        log,
        quiet,
        append,
        branching_factor,
        skip_logging_configuration,
        jvm_heap_size,
        gcs_requester_pays_configuration: Optional[GCSRequesterPaysConfiguration] = None,
    ):
        self._exit_stack = ExitStack()

        spark_home = find_spark_home()
        hail_jar_path = os.environ.get('HAIL_JAR')
        if hail_jar_path is None:
            try:
                _, hail_jar_path, extra_classpath = local_jar_information()
                extra_classpath = ':'.join([f'{spark_home}/jars/*', hail_jar_path, *extra_classpath])
            except ValueError:
                raise RuntimeError('local backend requires a packaged jar or HAIL_JAR to be set')
        else:
            extra_classpath = ':'.join([f'{spark_home}/jars/*', hail_jar_path])

        jvm_opts = []
        if jvm_heap_size is not None:
            jvm_opts.append(f'-Xmx{jvm_heap_size}')

        py4j_jars = glob.glob(f'{spark_home}/jars/py4j-*.jar')
        if len(py4j_jars) == 0:
            raise ValueError(f'No py4j JAR found in {spark_home}/jars')
        if len(py4j_jars) > 1:
            log.warning(f'found multiple p4yj jars arbitrarily choosing the first one: {py4j_jars}')

        port = launch_gateway(
            redirect_stdout=sys.stdout,
            redirect_stderr=sys.stderr,
            java_path=None,
            javaopts=jvm_opts,
            jarpath=py4j_jars[0],
            classpath=extra_classpath,
            die_on_exit=True,
        )
        self._gateway = JavaGateway(gateway_parameters=GatewayParameters(port=port, auto_convert=True))
        self._exit_stack.callback(self._gateway.shutdown)

        _is = getattr(self._gateway.jvm, 'is')
        py4jutils = scala_package_object(_is.hail.utils)

        if not skip_logging_configuration:
            py4jutils.configureLogging(log, quiet, append)

        if not quiet:
            connect_logger(py4jutils, 'localhost', 12888)

        py4jutils.log().info(f'Hail {__version__}')

        jbackend = scala_object(_is.hail.backend.local, 'LocalBackend')
        super().__init__(self._gateway.jvm, jbackend, tmpdir, tmpdir)
        self.gcs_requester_pays_configuration = gcs_requester_pays_configuration
        self._fs = None

        flags: Dict[str, str] = {}
        if branching_factor is not None:
            flags['branching_factor'] = str(branching_factor)

        self._initialize_flags(flags)

    def validate_file(self, uri: str) -> None:
        async_to_blocking(validate_file(uri, self.fs.afs))

    def register_ir_function(
        self,
        name: str,
        type_parameters: Union[Tuple[HailType, ...], List[HailType]],
        value_parameter_names: Union[Tuple[str, ...], List[str]],
        value_parameter_types: Union[Tuple[HailType, ...], List[HailType]],
        return_type: HailType,
        body: Expression,
    ):
        r = CSERenderer()
        code = r(finalize_randomness(body._ir))
        self._register_ir_function(
            name, type_parameters, value_parameter_names, value_parameter_types, return_type, code
        )

    def stop(self):
        super(Py4JBackend, self).stop()
        self._exit_stack.close()
        uninstall_exception_handler()

    @property
    def fs(self):
        if self._fs is None:
            self._fs = RouterFS(
                gcs_kwargs={"gcs_requester_pays_configuration": self.gcs_requester_pays_configuration},
            )
        return self._fs

    @property
    def gcs_requester_pays_configuration(self) -> Optional[GCSRequesterPaysConfiguration]:
        return self._gcs_requester_pays_config

    @gcs_requester_pays_configuration.setter
    def gcs_requester_pays_configuration(self, config: Optional[GCSRequesterPaysConfiguration]):
        self._gcs_requester_pays_config = config
        project, buckets = (None, None) if config is None else (config, None) if isinstance(config, str) else config
        self._jbackend.pySetGcsRequesterPaysConfig(project, buckets)
        # stale
        if self._fs is not None:
            self._fs.close()
            self._fs = None

    @property
    def requires_lowering(self):
        return True
