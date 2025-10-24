import os
from typing import Optional

from py4j.java_gateway import JavaGateway, JavaObject, Py4JJavaError

from hail.backend.backend import fatal_error_from_java_error_triplet
from hail.backend.py4j_backend import Py4JBackend, connect_logger, start_py4j_gateway
from hail.utils.java import scala_object, scala_package_object
from hail.version import __version__
from hailtop.aiocloud.aiogoogle import GCSRequesterPaysConfiguration
from hailtop.aiotools.validators import validate_file
from hailtop.fs.router_fs import RouterFS
from hailtop.utils import async_to_blocking


class LocalBackend(Py4JBackend):
    @classmethod
    def create(
        cls,
        tmpdir: str,
        logfile: str,
        quiet: bool,
        append: bool,
        branching_factor: int | None = None,
        skip_logging_configuration: bool = False,
        jvm_heap_size: str | None = None,
        gcs_requester_pays_configuration: Optional[GCSRequesterPaysConfiguration] = None,
    ) -> 'LocalBackend':
        max_heap_size = jvm_heap_size or os.getenv('HAIL_LOCAL_BACKEND_HEAP_SIZE')
        gateway = start_py4j_gateway(max_heap_size=max_heap_size)

        try:
            _is = getattr(gateway.jvm, 'is')
            py4jutils = scala_package_object(_is.hail.utils)
            try:
                if not skip_logging_configuration:
                    py4jutils.configureLogging(logfile, quiet, append)

                if not quiet:
                    connect_logger(py4jutils, 'localhost', 12888)

                backend = LocalBackend(
                    gateway,
                    scala_object(_is.hail.backend.local, 'LocalBackend'),
                    tmpdir,
                    tmpdir,
                    gcs_requester_pays_configuration,
                    branching_factor,
                )

                backend.logger.info(f'Hail {__version__}')

                return backend
            except Py4JJavaError as e:
                tpl = py4jutils.handleForPython(e.java_exception)
                deepest, full, error_id = tpl._1(), tpl._2(), tpl._3()
                raise fatal_error_from_java_error_triplet(deepest, full, error_id) from None
        except:
            gateway.shutdown()
            raise

    def __init__(
        self,
        jgateway: JavaGateway,
        jbackend: JavaObject,
        tmpdir: str,
        remote_tmpdir: str,
        gcs_requester_pays_configuration: GCSRequesterPaysConfiguration | None,
        branching_factor: int | None = None,
        optimizer_iterations: int | None = None,
    ):
        super().__init__(jgateway.jvm, jbackend, tmpdir, remote_tmpdir)
        self._gateway = jgateway

        flags = {}
        if branching_factor is not None:
            flags['branching_factor'] = branching_factor

        if optimizer_iterations is not None:
            flags['optimizer_iterations'] = optimizer_iterations

        self._fs = None
        self.gcs_requester_pays_configuration = gcs_requester_pays_configuration
        self._initialize_flags(flags)

    def validate_file(self, uri: str) -> None:
        async_to_blocking(validate_file(uri, self.fs.afs))

    @property
    def fs(self) -> RouterFS:
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

    def stop(self):
        super().stop()
        self._gateway.shutdown()

        if self._fs is not None:
            self._fs.close()
            self._fs = None
