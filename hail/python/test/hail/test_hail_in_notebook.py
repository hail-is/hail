from hailtop.utils.process import sync_check_exec
import os
import pathlib


def test_hail_in_notebook():
    folder = pathlib.Path(__file__).parent.resolve()
    source_ipynb = os.path.join(folder, 'test_hail_in_notebook.ipynb')
    output_ipynb = os.path.join(folder, 'test_hail_in_notebook_out.ipynb')
    sync_check_exec('jupyter', 'nbconvert', '--to', 'notebook', '--execute', str(source_ipynb), '--output', str(output_ipynb))
