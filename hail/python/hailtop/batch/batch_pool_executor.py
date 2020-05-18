from typing import *
import dill
import base64
import asyncio
import concurrent

from hailtop.utils import secret_alnum_string, async_to_blocking

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
    <https://dill.readthedocs.io/en/latest/dill.html>`__, sent to a Python 3.7
    docker container in the cloud, deserialized, and executed. The results are
    serialized and returned to the machine from which :meth:`.submit` was called.

    Examples
    --------
    >>> with BatchPoolExecutor() as bpe:
    ...     bpe.submit(f)
    """

    def __init__(self, *, name=None, backend=None):
        self.name = name or "BatchPoolExecutor-" + secret_alnum_string(4)
        self.backend = backend or ServiceBackend()
        self.batch = Batch(name=self.name, backend=self.backend, default_image='python:3.7')
        self.jobs = []

    def __enter__(self):
        return self

    def submit(self, f):
        try:
            name = f.__name__
        except AttributeError:
            name = '<anonymous>'
        j = self.batch.new_job(f'{name}-{secret_alnum_string(4)}')

        pickledfoo = base64.b64encode(dill.dumps(f))
        j.command(f'pip install -U dill')
        j.command(f'''
cat >code <<'EOF'
{pickledfoo.decode()}
EOF
''')
        j.command(f'''python3 -c "
import base64
import dill
with open(\\"code\\", \\"rb\\") as f:
    with open(\\"{j.ofile}\\", 'wb') as out:
        dill.dump(dill.loads(base64.b64decode(f.read()))(), out)"''')
        self.jobs.append(j)
        return j

    def __exit__(self, type, value, traceback):
        bucket = self.backend._batch_client.bucket
        remote_tmpdir = f'gs://{bucket}/batch/{self.name}/'
        for j in self.jobs:
            self.batch.write_output(j.ofile, remote_tmpdir + j.name + '/' + j.ofile)
        self.batch.run()
        gcs = GCS(blocking_pool=concurrent.futures.ThreadPoolExecutor())

        async def read_all():
            results = await asyncio.gather(*[gcs.read_binary_gs_file(remote_tmpdir + j.name + '/' + j.ofile) for j in self.jobs])
            return [dill.loads(x) for x in results]
        self.results = async_to_blocking(read_all())


