import glob
import logging
import os
import sys
from contextlib import ExitStack
from typing import List, Optional, Tuple, Union

from py4j.java_gateway import GatewayParameters, JavaGateway, launch_gateway

from hail.ir import finalize_randomness
from hail.ir.renderer import CSERenderer
from hailtop.aiocloud.aiogoogle import GCSRequesterPaysConfiguration
from hailtop.aiotools.validators import validate_file
from hailtop.fs.router_fs import RouterFS
from hailtop.utils import async_to_blocking, find_spark_home

from ..expr import Expression
from ..expr.types import HailType
from .backend import local_jar_information
from .py4j_backend import Py4JBackend, uninstall_exception_handler

log = logging.getLogger('hail.backend')


class LocalBackend(Py4JBackend):
    def __init__(
        self,
        tmpdir,
        log,
        quiet,
        append,
        branching_factor,
        skip_logging_configuration,
        optimizer_iterations,
        jvm_heap_size,
        gcs_requester_pays_configuration: Optional[GCSRequesterPaysConfiguration] = None,
    ):
        self._exit_stack = ExitStack()

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

        hail_package = getattr(self._gateway.jvm, 'is').hail

        jbackend = hail_package.backend.local.LocalBackend.apply(
            tmpdir,
            log,
            True,
            append,
            skip_logging_configuration,
        )
        jhc = hail_package.HailContext.apply(jbackend, branching_factor, optimizer_iterations)

        super(LocalBackend, self).__init__(self._gateway.jvm, jbackend, jhc)
        self._fs = self._exit_stack.enter_context(
            RouterFS(gcs_kwargs={'gcs_requester_pays_configuration': gcs_requester_pays_configuration})
        )

        self._logger = None

        flags = {}
        if gcs_requester_pays_configuration is not None:
            if isinstance(gcs_requester_pays_configuration, str):
                flags['gcs_requester_pays_project'] = gcs_requester_pays_configuration
            else:
                assert isinstance(gcs_requester_pays_configuration, tuple)
                flags['gcs_requester_pays_project'] = gcs_requester_pays_configuration[0]
                flags['gcs_requester_pays_buckets'] = ','.join(gcs_requester_pays_configuration[1])

        self._initialize_flags(flags)

    def validate_file(self, uri: str) -> None:
        async_to_blocking(validate_file(uri, self._fs.afs))

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
        super().stop()
        self._exit_stack.close()
        uninstall_exception_handler()

    @property
    def fs(self):
        return self._fs

    @property
    def requires_lowering(self):
        return True
