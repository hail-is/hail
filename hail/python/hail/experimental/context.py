import glob
import logging
import os
import sys
from typing import Dict, List, Optional

from py4j.java_gateway import GatewayParameters, JavaGateway, JavaObject, launch_gateway
from py4j.protocol import Py4JJavaError

from hail.backend.backend import fatal_error_from_java_error_triplet, local_jar_information
from hail.backend.py4j_backend import Log4jLogger, Py4JBackend, connect_logger
from hail.context import HailContext, _get_log
from hail.context import init as init_
from hail.utils import get_env_or_default
from hail.utils.java import BackendType, array_of, choose_backend, scala_object, scala_package_object
from hailtop.aiocloud.aiogoogle import GCSRequesterPaysConfiguration
from hailtop.aiotools.validators import validate_file
from hailtop.config import ConfigVariable, configuration_of, get_remote_tmpdir
from hailtop.fs.router_fs import RouterFS
from hailtop.utils import async_to_blocking, find_spark_home


def init(
    app_name=None,
    log=None,
    quiet=False,
    append=False,
    branching_factor=50,
    tmp_dir=None,
    default_reference='GRCh37',
    global_seed=None,
    _optimizer_iterations=None,
    *,
    backend: Optional[BackendType] = None,
    worker_cores=None,
    worker_memory=None,
    batch_id=None,
    gcs_requester_pays_configuration: Optional[GCSRequesterPaysConfiguration] = None,
    regions: Optional[List[str]] = None,
    gcs_bucket_allow_list: Optional[Dict[str, List[str]]] = None,
    jvm_heap_size: str | None = None,
    skip_logging_configuration: bool = False,
    **kwargs,
) -> None:
    backend = choose_backend(backend)
    if backend != 'batch':
        return init_(
            app_name=app_name,
            log=log,
            quiet=quiet,
            append=append,
            branching_factor=branching_factor,
            tmp_dir=tmp_dir,
            default_reference=default_reference,
            global_seed=global_seed,
            _optimizer_iterations=_optimizer_iterations,
            backend=backend,
            worker_cores=worker_cores,
            worker_memory=worker_memory,
            batch_id=batch_id,
            gcs_requester_pays_configuration=gcs_requester_pays_configuration,
            regions=regions,
            gcs_bucket_allow_list=gcs_bucket_allow_list,
            skip_logging_configuration=skip_logging_configuration,
            **kwargs,
        )

    log = _get_log(log)
    remote_tmpdir = get_remote_tmpdir('ServiceBackend', remote_tmpdir=tmp_dir)

    gateway = __start_py4j_gateway(jvm_heap_size)
    try:
        _is = getattr(gateway.jvm, 'is')
        jutils = scala_package_object(_is.hail.utils)

        try:
            if not skip_logging_configuration:
                _is.hail.HailContext.configureLogging(log, quiet, append)

            if not quiet:
                connect_logger(jutils, 'localhost', 12888)

            jbackend = __init_batch_backend(
                gateway=gateway,
                name=app_name,
                batch_id=batch_id,
                worker_cores=worker_cores,
                worker_memory=worker_memory,
                regions=regions,
            )

            optimizer_iterations = get_env_or_default(_optimizer_iterations, 'HAIL_OPTIMIZER_ITERATIONS', 3)
            jhc = _is.hail.HailContext.getOrCreate(jbackend, branching_factor, optimizer_iterations)
            driver = __Py4JQueryDriver(
                gateway, jbackend, jhc, remote_tmpdir, remote_tmpdir, gcs_requester_pays_configuration
            )
            HailContext.create(log, quiet, append, default_reference, global_seed, driver)
        except Py4JJavaError as e:
            tpl = jutils.handleForPython(e.java_exception)
            deepest, full, error_id = tpl._1(), tpl._2(), tpl._3()
            raise fatal_error_from_java_error_triplet(deepest, full, error_id) from None
    except:
        gateway.shutdown()
        raise


def __start_py4j_gateway(jvm_heap_size: str | None = None) -> JavaGateway:
    spark_home = find_spark_home()

    if (hail_jar_path := os.getenv('HAIL_JAR')) is not None:
        extra_classpath = ':'.join([f'{spark_home}/jars/*', hail_jar_path])
    else:
        try:
            local_jar_info = local_jar_information()
            hail_jar_path = local_jar_info.path
            extra_classpath = ':'.join([f'{spark_home}/jars/*', hail_jar_path, *local_jar_info.extra_classpath])
        except ValueError:
            raise RuntimeError('local backend requires a packaged jar or HAIL_JAR to be set')

    jvm_opts = []
    if (max_heap_size := jvm_heap_size or os.getenv('HAIL_LOCAL_BACKEND_HEAP_SIZE')) is not None:
        jvm_opts.append(f'-Xmx{max_heap_size}')

    py4j_jars = glob.glob(f'{spark_home}/jars/py4j-*.jar')
    if len(py4j_jars) == 0:
        raise ValueError(f'No py4j JAR found in {spark_home}/jars')
    if len(py4j_jars) > 1:
        logging.warning(f'found multiple p4yj jars arbitrarily choosing the first one: {py4j_jars}')

    port = launch_gateway(
        redirect_stdout=sys.stdout,
        redirect_stderr=sys.stderr,
        java_path=None,
        javaopts=jvm_opts,
        jarpath=py4j_jars[0],
        classpath=extra_classpath,
        die_on_exit=True,
    )

    return JavaGateway(gateway_parameters=GatewayParameters(port=port, auto_convert=True))


def __init_batch_backend(
    gateway: JavaGateway,
    name: str | None = None,
    batch_id: int | None = None,
    billing_project: str | None = None,
    deploy_config_file: str | None = None,
    worker_cores: str | None = None,
    worker_memory: str | None = None,
    storage: str | None = None,
    cloudfuse_configs: List[str] | None = None,
    regions: List[str] | None = None,
) -> JavaObject:
    jvm = gateway.jvm
    _is = getattr(jvm, 'is')

    if batch_id is None:
        billing_project = configuration_of(ConfigVariable.BATCH_BILLING_PROJECT, billing_project, None)
        if billing_project is None:
            raise ValueError(
                "No billing project.  Call 'init' with the billing "
                "project or run 'hailctl config set batch/billing_project "
                "MY_BILLING_PROJECT'"
            )

    if regions is None:
        config_regions = configuration_of(ConfigVariable.BATCH_REGIONS, None, None)
        if config_regions is not None:
            regions = config_regions.split(',')

    ServiceBackend = scala_object(_is.hail.backend.service, 'ServiceBackend')
    return ServiceBackend.pyServiceBackend(
        name or 'hail',
        batch_id,
        billing_project,
        deploy_config_file,
        configuration_of(ConfigVariable.QUERY_BATCH_WORKER_CORES, worker_cores, str(1)),
        configuration_of(ConfigVariable.QUERY_BATCH_WORKER_MEMORY, worker_memory, 'standard'),
        storage or '0Gi',
        array_of(gateway, _is.hail.services.CloudfuseConfig),
        array_of(gateway, jvm.String, *regions),
    )


class __Py4JQueryDriver(Py4JBackend):
    def __init__(
        self,
        jgateway: JavaGateway,
        jbackend: JavaObject,
        jhc: JavaObject,
        tmpdir: str,
        remote_tmpdir: str,
        gcs_requester_pays_configuration: GCSRequesterPaysConfiguration | None,
    ):
        super().__init__(jgateway.jvm, jbackend, jhc, tmpdir, remote_tmpdir)
        self._gateway = jgateway
        self.gcs_requester_pays_configuration = gcs_requester_pays_configuration
        self._fs = RouterFS(gcs_kwargs={'gcs_requester_pays_configuration': gcs_requester_pays_configuration})
        self._logger = Log4jLogger(self._utils_package_object)
        self._initialize_flags({})

    def validate_file(self, uri: str) -> None:
        async_to_blocking(validate_file(uri, self._fs.afs))

    @property
    def fs(self):
        return self._fs

    @property
    def requires_lowering(self):
        return True

    def stop(self):
        super().stop()
        self._gateway.shutdown()
        self._fs.close()
