from typing import Optional, Callable, Type, Union, List
from types import TracebackType
from io import BytesIO
import sys
import asyncio
import concurrent
import dill
import functools

from hailtop.utils import secret_alnum_string

from .batch import Batch
from .backend import ServiceBackend
from .google_storage import GCS


class BatchPoolExecutor(concurrent.futures.Executor):
    """An executor which executes python functions in the cloud.

    :class:`.ProcessPoolExecutor` and :class:`.ThreadPoolExecutor` enable the
    use of all the computer cores available on a single
    computer. :class:`.BatchPoolExecutor` enables the use of an effectively
    arbitrary number of computer cores.

    Functions provided to :meth:`.submit` are serialized using `dill
    <https://dill.readthedocs.io/en/latest/dill.html>`__, sent to a Python
    docker container in the cloud, deserialized, and executed. The results are
    serialized and returned to the machine from which :meth:`.submit` was
    called. The Python version in the docker container will share a major and
    minor verison with the local process. The ``image`` parameter overrides this
    behavior.

    Examples
    --------
    >>> with BatchPoolExecutor() as bpe:
    ...     bpe.submit(f)
    """

    def __init__(self, *,
                 name: Optional[str] = None,
                 backend: Optional[ServiceBackend] = None,
                 image: Optional[str] = None):
        self.name = name or "BatchPoolExecutor-" + secret_alnum_string(4)
        self.backend = backend or ServiceBackend()
        if not isinstance(self.backend, ServiceBackend):
            raise ValueError(f'BatchPoolExecutor is not compatible with {type(backend)}')
        self.batches: List[Batch] = []
        bucket: str = self.backend._bucket_name
        directory = f'gs://{bucket}/batch-pool-executor/{self.name}/'
        self.inputs = directory + 'inputs/'
        self.outputs = directory + 'outputs/'
        self.gcs = GCS(blocking_pool=concurrent.futures.ThreadPoolExecutor())
        self.futures: List[BatchPoolFuture] = []
        self.finished_future_count = 0
        self.already_shutdown = False
        version = sys.version_info
        self.image = image or f'hailgenetics/python-dill:{version.major}.{version.minor}'

    @staticmethod
    def async_to_blocking(coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def __enter__(self):
        return self

    def submit(self, unapplied: Callable, *args, **kwargs):
    # def submit(self, *callable_and_args, **kwargs):
    #     unapplied, *args = callable_and_args
    #     return BatchPoolExecutor.async_to_blocking(self.async_submit(unapplied, args, kwargs))

    # async def async_submit(self, unapplied: Callable, *args, **kwargs):
        if self.already_shutdown:
            raise RuntimeError(f'BatchPoolExecutor has already been shutdown.')

        try:
            name = unapplied.__name__
        except AttributeError:
            name = '<anonymous>'
        name = f'{name}-{secret_alnum_string(4)}'
        batch = Batch(name=self.name + '-' + name,
                      backend=self.backend,
                      default_image=self.image)
        self.batches.append(batch)
        j = batch.new_job(name)

        pipe = BytesIO()
        dill.dump(functools.partial(unapplied, *args, **kwargs), pipe)
        pipe.seek(0)
        pickledfun_gcs = self.inputs + f'{name}/pickledfun'
        BatchPoolExecutor.async_to_blocking(self.gcs.write_gs_file_from_file_like_object(pickledfun_gcs, pipe))
        pickledfun_local = batch.read_input(pickledfun_gcs)
        j.command('set -ex')
        j.command(f'''python3 -c "
import base64
import dill
with open(\\"{j.ofile}\\", \\"wb\\") as out:
    try:
        with open(\\"{pickledfun_local}\\", \\"rb\\") as f:
            dill.dump(dill.load(f)(), out)
    except Exception as e:
        dill.dump(e, out)
"''')
        output_gcs = self.outputs + f'{name}/output'
        batch.write_output(j.ofile, output_gcs)
        backend_batch = batch.run(wait=False,
                                  disable_progress_bar=True)

        future = BatchPoolFuture(self, backend_batch, output_gcs)
        return future

    def __exit__(self,
                 exc_type: Optional[Type[BaseException]],
                 exc_value: Optional[BaseException],
                 traceback: Optional[TracebackType]):
        self.shutdown()

    def add_future(self, f):
        self.futures.append(f)

    def finish_future(self):
        self.finished_future_count += 1
        if self.already_shutdown and self.finished_future_count == len(self.futures):
            self.gcs.shutdown(False)

    def shutdown(self, wait=True):
        if wait:
            for f in self.futures:
                f._fetch_result()
            self.gcs.shutdown(True)
        else:
            if self.finished_future_count == len(self.futures):
                self.gcs.shutdown(False)
        self.already_shutdown = True


NO_VALUE = object()
CANCELLED = object()


class BatchPoolFuture:
    def __init__(self, executor, batch, output_gcs):
        self.executor = executor
        self.batch = batch
        self.output_gcs = output_gcs
        self.value = NO_VALUE
        self._exception = None
        executor.add_future(self)

    def cancel(self):
        if self.value == NO_VALUE:
            self.batch.cancel()
            self.value = CANCELLED
            return True
        return False

    def cancelled(self):
        return self.value == CANCELLED

    def running(self):
        # FIXME: document that futures are never running and always cancellable
        return False

    def done(self):
        return self.value != NO_VALUE

    def result(self, timeout: Optional[Union[float, int]] = None):
        return BatchPoolExecutor.async_to_blocking(self.async_result(timeout))

    async def async_result(self, timeout: Optional[Union[float, int]] = None):
        await self._async_fetch_result(timeout)
        if self._exception:
            raise ValueError(f'submitted job failed') from self._exception
        return self.value

    def _fetch_result(self, timeout: Optional[Union[float, int]] = None):
        BatchPoolExecutor.async_to_blocking(self._async_fetch_result(timeout))

    async def _async_fetch_result(self, timeout: Optional[Union[float, int]]):
        if self.value != NO_VALUE:
            return
        if self.cancelled():
            raise concurrent.futures.CancelledError()
        try:
            await asyncio.wait_for(self.batch._async_batch.wait(), timeout)
        except asyncio.TimeoutError:
            raise concurrent.futures.TimeoutError()
        try:
            self.value = dill.loads(await self.executor.gcs.read_binary_gs_file(self.output_gcs))
        except Exception as exc:
            self.value = exc
        if isinstance(self.value, BaseException):
            self._exception = self.value

    def exception(self, timeout: Optional[Union[float, int]] = None):
        self.result(timeout)

    def add_done_callback(self, fn):
        pass
