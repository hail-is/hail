from py4j.java_gateway import JavaGateway, JavaObject
from py4j.protocol import Py4JJavaError

from hail.backend.backend import fatal_error_from_java_error_triplet
from hail.backend.local_backend import LocalBackend
from hail.backend.py4j_backend import raise_when_mismatched_hail_versions, start_py4j_gateway
from hail.context import HailContext, _get_log
from hail.context import init as init_
from hail.utils.java import BackendType, array_of, choose_backend, scala_object, scala_package_object
from hail.version import __version__
from hailtop.aiocloud.aiogoogle import GCSRequesterPaysConfiguration
from hailtop.config import ConfigVariable, configuration_of, get_remote_tmpdir


def init(
    app_name: str | None = None,
    log: str | None = None,
    quiet: bool = False,
    append: bool = False,
    branching_factor: int = 50,
    tmp_dir: str | None = None,
    default_reference: str = 'GRCh37',
    global_seed: int | None = None,
    _optimizer_iterations: int | None = None,
    *,
    backend: BackendType | None = None,
    worker_cores: int | None = None,
    worker_memory: str | None = None,
    batch_id: int | None = None,
    gcs_requester_pays_configuration: GCSRequesterPaysConfiguration | None = None,
    regions: list[str] | None = None,
    gcs_bucket_allow_list: dict[str, list[str]] | None = None,
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

    gateway = start_py4j_gateway(max_heap_size=jvm_heap_size)
    try:
        raise_when_mismatched_hail_versions(gateway.jvm)
        _is = getattr(gateway.jvm, 'is')
        py4jutils = scala_package_object(_is.hail.utils)

        try:
            if not skip_logging_configuration:
                py4jutils.configureLogging(log, quiet, append)

            jbackend = __init_batch_backend(
                gateway=gateway,
                name=app_name,
                batch_id=batch_id,
                worker_cores=worker_cores,
                worker_memory=worker_memory,
                regions=regions,
            )

            flags = {}
            if branching_factor is not None:
                flags['branching_factor'] = str(branching_factor)

            if _optimizer_iterations is not None:
                flags['optimizer_iterations'] = str(_optimizer_iterations)

            backend = LocalBackend(gateway, jbackend, flags)
            backend.remote_tmpdir = remote_tmpdir
            backend.local_tmpdir = tmp_dir
            backend.gcs_requester_pays_configuration = gcs_requester_pays_configuration

            backend.logger.info(f'Hail {__version__}')

            HailContext.create(log, quiet, append, default_reference, global_seed, backend)
        except Py4JJavaError as e:
            tpl = py4jutils.handleForPython(e.java_exception)
            deepest, full, error_id = tpl._1(), tpl._2(), tpl._3()
            raise fatal_error_from_java_error_triplet(deepest, full, error_id) from None
    except:
        gateway.shutdown()
        raise


def __init_batch_backend(
    gateway: JavaGateway,
    name: str | None = None,
    batch_id: int | None = None,
    billing_project: str | None = None,
    deploy_config_file: str | None = None,
    worker_cores: str | None = None,
    worker_memory: str | None = None,
    storage: str | None = None,
    cloudfuse_configs: list[str] | None = None,
    regions: list[str] | None = None,
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
        regions = config_regions.split(',') if config_regions is not None else []

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
