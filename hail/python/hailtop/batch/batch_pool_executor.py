from typing import Optional, Callable, Type, Union, List, Any, Iterable
from types import TracebackType
from io import BytesIO
import asyncio
import concurrent
import dill  # type: ignore
import functools
import sys
import time

from hailtop.utils import secret_alnum_string, partition  # type: ignore
import hailtop.batch_client.client as low_level_batch_client  # type: ignore
from hailtop.batch_client.parse import parse_cpu_in_mcpu  # type: ignore

from .batch import Batch
from .backend import ServiceBackend
from ..google_storage import GCS  # type: ignore

if sys.version_info < (3, 7):
    def create_task(coro, *, name=None):  # pylint: disable=unused-argument
        asyncio.ensure_future(coro)
    asyncio.create_task = create_task


def cpu_spec_to_float(spec: Union[int, str]) -> float:
    if isinstance(spec, str):
        mcpu = parse_cpu_in_mcpu(spec)
        assert mcpu is not None
        return mcpu / 1000
    return float(spec)


def chunk(fn):
    def chunkedfn(*args):
        return [fn(*arglist) for arglist in zip(*args)]
    return chunkedfn


def async_to_blocking(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class BatchPoolExecutor:
    """An executor which executes Python functions in the cloud.

    :class:`.concurrent.futures.ProcessPoolExecutor` and
    :class:`.concurrent.futures.ThreadPoolExecutor` enable the use of all the
    computer cores available on a single computer. :class:`.BatchPoolExecutor`
    enables the use of an effectively arbitrary number of cloud computer cores.

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

    >>> with BatchPoolExecutor() as bpe:  # doctest: +SKIP
    ...     future_nine = bpe.submit(lambda: 3 + 6)
    >>> future_nine.result()  # doctest: +SKIP
    9

    :meth:`.map` facilitates the common case of executing a function on many
    values in parallel:

    >>> with BatchPoolExecutor() as bpe:  # doctest: +SKIP
    ...     list(bpe.map(lambda x: x * 3, range(4)))
    [0, 3, 6, 9]

    Parameters
    ----------
    name:
        A name for the executor. Executors produce many batches and each batch
        will include this name as a prefix.
    backend:
        Backend used to execute the jobs. Must be a :class:`.ServiceBackend`.
    image:
        The name of a Docker image used for each submitted job. The image must
        include Python 3.6 or later and must have the ``dill`` Python package
        installed. If you intend to use ``numpy``, ensure that OpenBLAS is also
        installed. If unspecified, an image with a matching Python verison and
        ``numpy``, ``scipy``, and ``sklearn`` installed is used.
    cpus_per_job:
        The number of CPU cores to allocate to each job. The default value is
        ``1``. The parameter is passed unaltered to :meth:`.Job.cpu`. This
        parameter's value is used to set several environment variables
        instructing BLAS and LAPACK to limit core use.
    wait_on_exit:
        If ``True`` or unspecified, wait for all jobs to complete when exiting a
        context. If ``False``, do not wait. This option has no effect if this
        executor is not used with the ``with`` syntax.
    cleanup_bucket:
        If ``True`` or unspecified, delete all temporary files in the cloud
        storage bucket when this executor fully shuts down. If Python crashes
        before the executor is shutdown, the files will not be deleted.
    project:
        If specified, the project to use when authenticating with Google
        Storage. Google Storage is used to transfer serialized values between
        this computer and the cloud machines that execute jobs.
    """

    def __init__(self, *,
                 name: Optional[str] = None,
                 backend: Optional[ServiceBackend] = None,
                 image: Optional[str] = None,
                 cpus_per_job: Optional[Union[int, str]] = None,
                 wait_on_exit: bool = True,
                 cleanup_bucket: bool = True,
                 project: Optional[str] = None):
        self.name = name or "BatchPoolExecutor-" + secret_alnum_string(4)
        self.backend = backend or ServiceBackend()
        if not isinstance(self.backend, ServiceBackend):
            raise ValueError(f'BatchPoolExecutor is not compatible with {type(backend)}')
        self.batches: List[Batch] = []
        bucket: str = self.backend._bucket_name
        self.directory = f'gs://{bucket}/batch-pool-executor/{self.name}/'
        self.inputs = self.directory + 'inputs/'
        self.outputs = self.directory + 'outputs/'
        self.gcs = GCS(blocking_pool=concurrent.futures.ThreadPoolExecutor(),
                       project=project)
        self.futures: List[BatchPoolFuture] = []
        self.finished_future_count = 0
        self._shutdown = False
        version = sys.version_info
        self.image = image or f'hailgenetics/python-dill:{version.major}.{version.minor}'
        self.cpus_per_job = cpus_per_job or 1
        self.cleanup_bucket = cleanup_bucket
        self.wait_on_exit = wait_on_exit

    def __enter__(self):
        return self

    def map(self,
            fn: Callable,
            *iterables: Iterable[Any],
            timeout: Optional[Union[int, float]] = None,
            chunksize: int = 1):
        """Call `fn` on cloud machines with arguments from `iterables`.

        This function returns a generator which will produce each result in the
        same order as the `iterables`, only blocking if the result is not yet
        ready. You can convert the generator to a list with :class:`.list`.

        Examples
        --------

        Do nothing, but on the cloud:

        >>> with BatchPoolExecutor() as bpe:  # doctest: +SKIP
        ...     list(bpe.map(lambda x: x, range(4)))
        [0, 1, 2, 3]

        Call a function with two parameters, on the cloud:

        >>> with BatchPoolExecutor() as bpe:  # doctest: +SKIP
        ...     list(bpe.map(lambda x, y: x + y,
        ...                  ["white", "cat", "best"],
        ...                  ["house", "dog", "friend"]))
        ["whitehouse", "catdog", "bestfriend"]

        Generate products of random matrices, on the cloud:

        >>> def random_product(seed):
        ...     np.random.seed(seed)
        ...     w = np.random.rand(1, 100)
        ...     u = np.random.rand(100, 1)
        ...     return float(w @ u)
        >>> with BatchPoolExecutor() as bpe:  # doctest: +SKIP
        ...     list(bpe.map(random_product, range(4)))
        [24.440006386777277, 23.325755364428026, 23.920184804993806, 25.47912882125101]

        Parameters
        ----------
        fn:
            The function to execute.
        iterables:
            The `iterables` are zipped together and each tuple is used as
            arguments to `fn`. See the second example for more detail. It is not
            possible to pass keyword arguments. Each element of `iterables` must
            have the same length.
        timeout:
            This is roughly a timeout on how long we wait on each function
            call. Specifically, each call to the returned generator's
            :class:`.BatchPoolFuture`
            :meth:`.iterator.__next__` invokes :meth:`.BatchPoolFuture.result` with this
            `timeout`.
        chunksize:
            The number of tasks to schedule in the same docker container. Docker
            containers take about 5 seconds to start. Ideally, each task should
            take an order of magnitude more time than start-up time. You can
            make the chunksize larger to reduce parallelism but increase the
            amount of meaningful work done per-container.
        """

        agen = async_to_blocking(
            self.async_map(fn, iterables, timeout=timeout, chunksize=chunksize))

        def generator_from_async_generator(aiter):
            try:
                while True:
                    yield async_to_blocking(aiter.__anext__())
            except StopAsyncIteration:
                return
        return generator_from_async_generator(agen.__aiter__())

    async def async_map(self,
                        fn: Callable,
                        iterables: Iterable[Iterable[Any]],
                        timeout: Optional[Union[int, float]] = None,
                        chunksize: int = 1):
        """Aysncio compatible version of :meth:`.map`."""
        if chunksize > 1:
            list_per_argument = [list(x) for x in iterables]
            n = len(list_per_argument[0])
            assert all(n == len(x) for x in list_per_argument)
            n_chunks = (n + chunksize - 1) // chunksize
            iterables_chunks = [list(partition(n_chunks, x)) for x in list_per_argument]
            iterables_chunks = [
                chunk for chunk in iterables_chunks if len(chunk) > 0]
            fn = chunk(fn)
            iterables = iterables_chunks
        submissions = [self.async_submit(fn, *arguments)
                       for arguments in zip(*iterables)]
        futures = await asyncio.gather(*submissions)
        fetching_tasks = [asyncio.create_task(future._async_fetch_result())
                          for future in futures]

        async def async_result_or_cancel_all(future):
            try:
                return await future.async_result(timeout=timeout)
            except Exception as exc:
                for task in fetching_tasks:
                    task.cancel()
                raise exc
        if chunksize > 1:
            return (val
                    for future in futures
                    for val in await async_result_or_cancel_all(future))
        return (await async_result_or_cancel_all(future)
                for future in futures)

    def submit(self,
               fn: Callable,
               *args: Any,
               **kwargs: Any
               ) -> 'BatchPoolFuture':
        """Call `fn` on a cloud machine with all remaining arguments and keyword arguments.

        The function, any objects it references, the arguments, and the keyword
        arguments will be serialized to the cloud machine. Python modules are
        not serialized, so you must ensure any needed Python modules and
        packages already present in the underlying Docker image. For more
        details see the `default_image` argument to :class:`.BatchPoolExecutor`

        This function does not return the function's output, it returns a
        :class:`.BatchPoolFuture` whose :meth:`.BatchPoolFuture.result` method
        can be used to access the value.

        Examples
        --------

        Do nothing, but on the cloud:

        >>> with BatchPoolExecutor() as bpe:  # doctest: +SKIP
        ...     future = bpe.submit(lambda x: x, 4)
        ...     future.result()
        4

        Call a function with two arguments and one keyword argument, on the
        cloud:

        >>> with BatchPoolExecutor() as bpe:  # doctest: +SKIP
        ...     future = bpe.submit(lambda x, y, z: x + y + z,
        ...                         "poly", "ethyl", z="ene")
        "polyethylene"

        Generate a product of two random matrices, on the cloud:

        >>> def random_product(seed):
        ...     np.random.seed(seed)
        ...     w = np.random.rand(1, 100)
        ...     u = np.random.rand(100, 1)
        ...     return float(w @ u)
        >>> with BatchPoolExecutor() as bpe:  # doctest: +SKIP
        ...     future = bpe.submit(random_product, 1)
        ...     future.result()
        [23.325755364428026]

        Parameters
        ----------
        fn:
            The function to execute.
        args:
            Arguments for the funciton.
        kwargs:
            Keyword arguments for the function.
        """
        return async_to_blocking(
            self.async_submit(fn, *args, **kwargs))

    async def async_submit(self,
                           unapplied: Callable,
                           *args: Any,
                           **kwargs: Any
                           ) -> 'BatchPoolFuture':
        """Aysncio compatible version of :meth:`BatchPoolExecutor.submit`."""

        if self._shutdown:
            raise RuntimeError('BatchPoolExecutor has already been shutdown.')

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
        await self.gcs.write_gs_file_from_file_like_object(pickledfun_gcs, pipe)
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
        print(\\"BatchPoolExecutor encountered an exception:\\")
        traceback.print_exc()
        dill.dump((e, traceback.format_exception(type(e), e, e.__traceback__)), out, recurse=True)
"''')
        output_gcs = self.outputs + f'{name}/output'
        batch.write_output(j.ofile, output_gcs)
        backend_batch = batch.run(wait=False,
                                  disable_progress_bar=True)

        return BatchPoolFuture(self, backend_batch, output_gcs)

    def __exit__(self,
                 exc_type: Optional[Type[BaseException]],
                 exc_value: Optional[BaseException],
                 traceback: Optional[TracebackType]):
        self.shutdown(wait=self.wait_on_exit)

    def _add_future(self, f):
        self.futures.append(f)

    def _finish_future(self):
        self.finished_future_count += 1
        if self._shutdown and self.finished_future_count == len(self.futures):
            self._cleanup(False)

    def shutdown(self, wait: bool = True):
        """Allow temporary resources to be cleaned up.

        Until shutdown is called, some temporary cloud storage files will
        persist. After shutdown has been called *and* all outstanding jobs have
        completed, these files will be deleted.

        Parameters
        ----------
        wait:
            If true, wait for all jobs to complete before returning from this
            method.
        """
        if wait:
            async_to_blocking(
                asyncio.gather(*[f._async_fetch_result() for f in self.futures]))
        if self.finished_future_count == len(self.futures):
            self._cleanup(False)
        self._shutdown = True

    def _cleanup(self, wait):
        if self.cleanup_bucket:
            async_to_blocking(
                self.gcs.delete_gs_files(self.directory))
        self.gcs.shutdown(wait)


class NoValue:
    pass


class Cancelled:
    pass


NO_VALUE = NoValue()
CANCELLED = Cancelled()


class BatchPoolFuture:
    def __init__(self,
                 executor: BatchPoolExecutor,
                 batch: low_level_batch_client.Batch,
                 output_gcs: str):
        self.executor = executor
        self.batch = batch
        self.output_gcs = output_gcs
        self.value: Any = NO_VALUE
        self._exception: Optional[BaseException] = None
        self.fetch_lock = asyncio.Lock()
        executor._add_future(self)

    def cancel(self):
        """Cancels this job if it has not yet been cancelled.

        ``True`` is returned if the job is cancelled. ``False`` is returned if
        the job has already completed.
        """
        if self.value == NO_VALUE:
            self.batch.cancel()
            self.value = CANCELLED
            self.executor._finish_future()
            return True
        return False

    def cancelled(self):
        """Returns ``True`` if :meth:`.cancel` was called before a value was produced.
        """
        return self.value == CANCELLED

    def running(self):  # pylint: disable=no-self-use
        """Always returns False.

        This future can always be cancelled, so this function always returns False.
        """
        return False

    def done(self):
        """Returns `True` if the function is complete and not cancelled.
        """
        return self.value != NO_VALUE

    def result(self, timeout: Optional[Union[float, int]] = None):
        """Blocks until the job is complete.

        If the job has been cancelled, this method raises a
        :class:`.concurrent.futures.CancelledError`.

        Parameters
        ----------
        timeout:
            Wait this long before raising a timeout error.
        """
        return async_to_blocking(self.async_result(timeout))

    async def async_result(self, timeout: Optional[Union[float, int]] = None):
        """Asynchronously wait until the job is complete.

        If the job has been cancelled, this method rasies a
        :class:`.concurrent.futures.CancelledError`.

        Parameters
        ----------
        timeout:
            Wait this long before raising a timeout error.
        """
        if self.cancelled():
            raise concurrent.futures.CancelledError()
        await self._async_fetch_result(timeout)
        if self._exception:
            raise self._exception
        return self.value

    def _fetch_result(self, timeout: Optional[Union[float, int]] = None):
        async_to_blocking(self._async_fetch_result(timeout))

    async def _async_fetch_result(self, timeout: Optional[Union[float, int]] = None):
        try:
            before = time.time()
            await asyncio.wait_for(self.fetch_lock.acquire(), timeout=timeout)
            if timeout:
                timeout -= time.time() - before
        except asyncio.TimeoutError:
            raise concurrent.futures.TimeoutError()
        try:
            if self.value != NO_VALUE:
                return
            try:
                await asyncio.wait_for(self.batch._async_batch.wait(disable_progress_bar=True),
                                       timeout=timeout)
            except asyncio.TimeoutError:
                raise concurrent.futures.TimeoutError()
            try:
                value, traceback = dill.loads(
                    await self.executor.gcs.read_binary_gs_file(self.output_gcs))
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
            finally:
                self.executor._finish_future()
        finally:
            self.fetch_lock.release()

    def exception(self, timeout: Optional[Union[float, int]] = None):
        """Block until the job is complete and raise any exceptions.
        """
        if self.cancelled():
            raise concurrent.futures.CancelledError()
        self.result(timeout)

    def add_done_callback(self, fn):
        """NOT IMPLEMENTED
        """
        raise NotImplementedError()
