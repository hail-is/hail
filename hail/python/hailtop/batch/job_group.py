from typing import Dict, Optional

import hailtop.batch_client.aioclient as _aiobc

from . import batch as _batch, job  # pylint: disable=cyclic-import


class JobGroup:
    @staticmethod
    def from_job_group_id(batch: '_batch.Batch', job_group_id: int) -> 'JobGroup':
        jg = batch.create_job_group()
        assert batch._async_batch
        jg._async_job_group = batch._async_batch.get_job_group(job_group_id)
        return jg

    def __init__(self,
                 batch: '_batch.Batch',
                 parent_job_group: Optional['JobGroup'],
                 *,
                 attributes: Optional[Dict[str, str]] = None,
                 cancel_after_n_failures: Optional[int] = None
                 ):
        self._batch = batch
        self._parent_job_group = parent_job_group
        self.attributes = attributes
        self.cancel_after_n_failures = cancel_after_n_failures
        self._async_job_group: Optional[_aiobc.JobGroup] = None

    @property
    def _is_root_job_group(self):
        return self._parent_job_group is None

    def _create_aiobc_job(self, *args, **kwargs):
        if self._is_root_job_group:
            return self._batch._async_batch.create_job(*args, **kwargs)
        assert self._async_job_group
        return self._async_job_group.create_job(*args, **kwargs)

    def create_job_group(self,
                         attributes: Optional[Dict[str, str]] = None,
                         cancel_after_n_failures: Optional[int] = None) -> 'JobGroup':
        return self._batch._create_job_group(self, attributes=attributes, cancel_after_n_failures=cancel_after_n_failures)

    def new_bash_job(
        self, name: Optional[str] = None, attributes: Optional[Dict[str, str]] = None, shell: Optional[str] = None
    ) -> 'job.BashJob':
        """
        Initialize a :class:`.BashJob` object with default memory, storage,
        image, and CPU settings (defined in :class:`.Batch`) upon batch creation.

        Examples
        --------
        Create and execute a batch `b` with one job `j` that prints "hello world":

        >>> import hailtop.batch as hb
        >>> b = hb.Batch()
        >>> jg = b.create_job_group()
        >>> j = jg.new_bash_job(name='hello', attributes={'language': 'english'})
        >>> j.command('echo "hello world"')
        >>> b.run()

        Parameters
        ----------
        name:
            Name of the job.
        attributes:
            Key-value pairs of additional attributes. 'name' is not a valid keyword.
            Use the name argument instead.
        """

        return self._batch._new_bash_job(self, name=name, attributes=attributes, shell=shell)

    def new_python_job(self, name: Optional[str] = None, attributes: Optional[Dict[str, str]] = None) -> 'job.PythonJob':
        """
        Initialize a new :class:`.PythonJob` object with default
        Python image, memory, storage, and CPU settings (defined in :class:`.Batch`)
        upon batch creation.

        Examples
        --------
        Create and execute a batch `b` with one job `j` that prints "hello alice":

        .. code-block:: python

            b = Batch(default_python_image='hailgenetics/python-dill:3.9-slim')
            jg = b.create_job_group()

            def hello(name):
                return f'hello {name}'

            j = jg.new_python_job()
            output = j.call(hello, 'alice')

            # Write out the str representation of result to a file

            b.write_output(output.as_str(), 'hello.txt')

            b.run()

        Notes
        -----

        The image to use for Python jobs can be specified by `default_python_image`
        when constructing a :class:`.Batch`. The image specified must have the `dill`
        package installed. If ``default_python_image`` is not specified, then a Docker
        image will automatically be created for you with the base image
        `hailgenetics/python-dill:[major_version].[minor_version]-slim` and the Python
        packages specified by ``python_requirements`` will be installed. The default name
        of the image is `batch-python` with a random string for the tag unless ``python_build_image_name``
        is specified. If the :class:`.ServiceBackend` is the backend, the locally built
        image will be pushed to the repository specified by ``image_repository``.

        Parameters
        ----------
        name:
            Name of the job.
        attributes:
            Key-value pairs of additional attributes. 'name' is not a valid keyword.
            Use the name argument instead.
        """

        return self._batch._new_python_job(self, name=name, attributes=attributes)
