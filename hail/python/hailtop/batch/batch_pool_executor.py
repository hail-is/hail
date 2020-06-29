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
    arbitrary number of cloud computer cores.

    Functions provided to :meth:`.submit` are serialized using `dill
    <https://dill.readthedocs.io/en/latest/dill.html>`__, sent to a Python
    docker container in the cloud, deserialized, and executed. The results are
    serialized and returned to the machine from which :meth:`.submit` was
    called. The Python version in the docker container will share a major and
    minor verison with the local process. The `image` parameter overrides this
    behavior.

    When used as a context manager (the ``with`` syntax), the executor will wait
    for all jobs to finish before finishing the ``with`` statement. This
    behavior can be controlled by the `wait_on_exit` parameter.

    This class creates a folder ``batch-pool-executor`` at the root of the
    bucket specified by the `backend`. This folder can be safely deleted after
    all jobs have completed.

    Examples
    --------

    Add ``3`` to ``6`` on a machine in the cloud and send the result back to
    this machine:

    >>> with BatchPoolExecutor() as bpe:
    ...     future_seven = bpe.submit(lambda: 3 + 6)
    >>> future_seven.result()
    9

    :meth:`.map` facilitates the common case of executing a function on many
    values in parallel:

    >>> with BatchPoolExecutor() as bpe:
    ...     list(bpe.map(lambda x: x * 3, range(4)))
    [0, 3, 6, 9]

    Parameters
    ----------
    name: :obj:`str`, optional
        A name for the executor. Executors produce many batches and each batch
        will include this name as a prefix.
    backend: :class:`.ServiceBackend`, optional
        Backend used to execute the jobs. Must be a :class:`.ServiceBackend`.
    image: :obj:`str`, optional
        The name of a Docker image used for each submitted job. The image must
        include Python 3.6 or later and must have the ``dill`` Python package
        installed. If you intend to use ``numpy``, ensure that OpenBLAS is also
        installed. If unspecified, an image with a matching Python verison and
        ``numpy``, ``scipy``, and ``sklearn`` installed is used.
    cpus_per_job: :obj:`int`, :obj:`float`, :obj:`str`, optional
        The number of CPU cores to allocate to each job. The default value is
        ``1``. The parameter is passed unaltered to :meth:`.Job.cpu`. This
        parameter's value is used to set several environment variables
        instructing BLAS and LAPACK to limit core use.
    wait_on_exit: :obj:`bool`
        If ``True`` or unspecified, wait for all jobs to complete when exiting a
        context. If ``False``, do not wait.
    cleanup_bucket: :obj:`bool`
        If ``True`` or unspecified, delete all temporary files in the cloud
        storage bucket when this executor fully shuts down. If Python crashes
        before the executor is shutdown, the files will not be deleted.
    """

    def __init__(self, *,
                 name: Optional[str] = None,
                 backend: Optional[ServiceBackend] = None,
                 image: Optional[str] = None,
                 cpus_per_job: Optional[Union[int, str]] = None,
                 wait_on_exit: bool = True,
                 cleanup_bucket: bool = True):
        self.name = name or "BatchPoolExecutor-" + secret_alnum_string(4)
        self.backend = backend or ServiceBackend()
        if not isinstance(self.backend, ServiceBackend):
            raise ValueError(f'BatchPoolExecutor is not compatible with {type(backend)}')
        self.batches: List[Batch] = []
        bucket: str = self.backend._bucket_name
        self.directory = f'gs://{bucket}/batch-pool-executor/{self.name}/'
        self.inputs = self.directory + 'inputs/'
        self.outputs = self.directory + 'outputs/'
        self.gcs = GCS(blocking_pool=concurrent.futures.ThreadPoolExecutor())
        self.futures: List[BatchPoolFuture] = []
        self.finished_future_count = 0
        self.already_shutdown = False
        version = sys.version_info
        self.image = image or f'hailgenetics/python-dill:{version.major}.{version.minor}'
        self.cpus_per_job = cpus_per_job or 1
        self.cleanup_bucket = cleanup_bucket

    @staticmethod
    def async_to_blocking(coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def __enter__(self):
        return self

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        """On cloud machines, call `fn` on every value of `iterables`.

        The returned list contains values in the same order as `iterables`.

        Parameters
        ----------
        fn: Callable
            The function to execute.
        iterables: Any
            Each value is provided as an argument to `fn`.
        
            A name for the executor. Executors produce many batches and each batch
            will include this name as a prefix.
        """

        unapplied, *args = callable_and_args
        return BatchPoolExecutor.async_to_blocking(
            self.async_map(callable, args))

    async def async_map(self, unapplied: Callable, args):
        submissions = [
            self.async_submit(unapplied, arg) for arg in args]
        futures = asyncio.gather(*submissions)
        results = [future.async_result() for future in futures]
        return asyncio.gather(*results)

    def submit(self, *callable_and_args, **kwargs):
        """
        Serialize a function and 
        """
        sunapplied, *args = callable_and_args
        return BatchPoolExecutor.async_to_blocking(
            self.async_submit(unapplied, args, kwargs))

    async def async_submit(self, unapplied: Callable, *args, **kwargs):
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
        dill.dump(functools.partial(unapplied, *args, **kwargs), pipe, recurse=True)
        pipe.seek(0)
        pickledfun_gcs = self.inputs + f'{name}/pickledfun'
        BatchPoolExecutor.async_to_blocking(
            self.gcs.write_gs_file_from_file_like_object(pickledfun_gcs, pipe))
        pickledfun_local = batch.read_input(pickledfun_gcs)
        j.cpu(self.cpus_per_job)
        thread_limit = str(int(max(1.0, cpu_spec_to_float(self.cpus_per_job))))
        j.env("OMP_NUM_THREADS", thread_limit)
        j.env("OPENBLAS_NUM_THREADS", thread_limit)
        j.env("MKL_NUM_THREADS", thread_limit)
        j.env("VECLIB_MAXIMUM_THREADS", thread_limit)
        j.env("NUMEXPR_NUM_THREADS", thread_limit)

        j.command('set -ex')
        j.command(f'''python3 -c "
import base64
import dill
import traceback
with open(\\"{j.ofile}\\", \\"wb\\") as out:
    try:
        with open(\\"{pickledfun_local}\\", \\"rb\\") as f:
            dill.dump((dill.load(f)(), None), out, recurse=True)
    except Exception as e:
        dill.dump((e, traceback.format_exception(type(e), e, e.__traceback__)), out, recurse=True)
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
            self._cleanup(False)

    def shutdown(self, wait=True):
        if wait:
            for f in self.futures:
                f._fetch_result()
            self._cleanup(True)
        else:
            if self.finished_future_count == len(self.futures):
                self._cleanup(False)
        self.already_shutdown = True

    def _cleanup(self, wait):
        if self.cleanup_bucket:
            BatchPoolExecutor.async_to_blocking(
                self.gcs.delete_gs_files(self.directory))
        self.gcs.shutdown(wait)


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
            raise self._exception
        return self.value

    def _fetch_result(self, timeout: Optional[Union[float, int]] = None):
        BatchPoolExecutor.async_to_blocking(self._async_fetch_result(timeout))

    async def _async_fetch_result(self, timeout: Optional[Union[float, int]]):
        if self.value != NO_VALUE:
            return
        if self.cancelled():
            raise concurrent.futures.CancelledError()
        try:
            await asyncio.wait_for(self.batch._async_batch.wait(disable_progress_bar=True), timeout)
        except asyncio.TimeoutError:
            raise concurrent.futures.TimeoutError()
        try:
            value, traceback = dill.loads(await self.executor.gcs.read_binary_gs_file(self.output_gcs))
            if traceback is not None:
                assert isinstance(value, BaseException)
                self.value = None
                traceback = ''.join(traceback)
                self._exception = ValueError(
                    f'submitted job failed:\n{traceback}')
            else:
                self.value = value
        except Exception as exc:
            self.value = None
            self._exception = exc

    def exception(self, timeout: Optional[Union[float, int]] = None):
        self.result(timeout)

    def add_done_callback(self, fn):
        pass


def cpu_spec_to_float(spec: Union[int, str]) -> float:
    if isinstance(spec, str):
        if spec[-1] == 'm':
            return int(spec[:-1]) / 1000
        return float(spec)
    return float(spec)
