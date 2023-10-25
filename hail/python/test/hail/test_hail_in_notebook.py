from hailtop.utils.process import sync_check_exec
import os
import pathlib
from .helpers import skip_when_local_backend


@skip_when_local_backend('In the LocalBackend, writing to a gs:// URL hangs indefinitely https://github.com/hail-is/hail/issues/13904')
def test_hail_in_notebook():
    folder = pathlib.Path(__file__).parent.resolve()
    source_ipynb = os.path.join(folder, 'test_hail_in_notebook.ipynb')
    output_ipynb = os.path.join(folder, 'test_hail_in_notebook_out.ipynb')
    sync_check_exec('jupyter', 'nbconvert', '--to', 'notebook', '--execute', str(source_ipynb), '--output', str(output_ipynb))
