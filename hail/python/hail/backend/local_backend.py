import os
from typing import Dict, Optional

from py4j.java_gateway import JavaGateway, JavaObject, Py4JJavaError

from hail.backend.backend import fatal_error_from_java_error_triplet
from hail.backend.py4j_backend import (
    Py4JBackend,
    raise_when_mismatched_hail_versions,
    start_py4j_gateway,
)
from hail.utils.java import scala_object, scala_package_object
from hail.version import __version__
from hailtop.aiocloud.aiogoogle import GCSRequesterPaysConfiguration


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
    ) -> Py4JBackend:
        max_heap_size = jvm_heap_size or os.getenv('HAIL_LOCAL_BACKEND_HEAP_SIZE')

        gateway = start_py4j_gateway(max_heap_size=max_heap_size)

        try:
            raise_when_mismatched_hail_versions(gateway.jvm)

            _is = getattr(gateway.jvm, 'is')
            py4jutils = scala_package_object(_is.hail.utils)
            try:
                if not skip_logging_configuration:
                    py4jutils.configureLogging(logfile, quiet, append)

                flags = {}
                if branching_factor is not None:
                    flags['branching_factor'] = str(branching_factor)

                jbackend = scala_object(_is.hail.backend.local, 'LocalBackend')
                backend = LocalBackend(gateway, jbackend, flags)

                backend.local_tmpdir = tmpdir
                backend.remote_tmpdir = tmpdir
                backend.gcs_requester_pays_configuration = gcs_requester_pays_configuration
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
        flags: Dict[str, str],
    ):
        self._gateway = jgateway
        super().__init__(jgateway.jvm, jbackend, flags)

    def stop(self):
        super().stop()
        self._gateway.shutdown()
