import pytest
import asyncio

import hail as hl
from hail.utils.java import Env, scala_object
from ..helpers import *


def create_backward_compatibility_files():
    import os

    all_values_table, all_values_matrix_table = create_all_values_datasets()

    file_version = Env.hail().expr.ir.FileFormat.version().toString()
    supported_codecs = scala_object(Env.hail().io, 'BufferSpec').specs()

    table_dir = resource(os.path.join('backward_compatability', str(file_version), 'table'))
    if not os.path.exists(table_dir):
        os.makedirs(table_dir)

    matrix_table_dir = resource(os.path.join('backward_compatability', str(file_version), 'matrix_table'))
    if not os.path.exists(matrix_table_dir):
        os.makedirs(matrix_table_dir)

    i = 0
    for codec in supported_codecs:
        all_values_table.write(os.path.join(table_dir, f'{i}.ht'), overwrite=True, _codec_spec=codec.toString())
        all_values_matrix_table.write(os.path.join(matrix_table_dir, f'{i}.hmt'), overwrite=True,
                                      _codec_spec=codec.toString())
        i += 1


@pytest.mark.skip(reason='comment this line to generate files for new versions')
def test_write():
    create_backward_compatibility_files()


@pytest.fixture(scope="module")
def all_values_matrix_table_fixture(init_hail):
    return create_all_values_matrix_table()


@pytest.fixture(scope="module")
def all_values_table_fixture(init_hail):
    return create_all_values_table()


# pytest sometimes uses background threads, named "Dummy-1", to collect tests. Our synchronous
# interfaces will try to get an event loop by calling `asyncio.get_event_loop()`. asyncio will
# create an event loop when `get_event_loop()` is called if and only if the current thread is the
# main thread. We therefore manually create an event loop which is used only for collecting the
# files.
# try:
#     old_loop = asyncio.get_running_loop()
# except RuntimeError as err:
#     assert 'no running event loop' in err.args[0]
#     old_loop = None
# loop = asyncio.new_event_loop()
# try:
#     asyncio.set_event_loop(loop)
#     resource_dir = resource('backward_compatability')
#     fs = hl.current_backend().fs
#     try:
#         ht_paths = [x.path for x in fs.ls(resource_dir + '/*/table/')]
#         mt_paths = [x.path for x in fs.ls(resource_dir + '/*/matrix_table/')]
#     finally:
#         hl.stop()
# finally:
#     loop.stop()
#     loop.close()
#     asyncio.set_event_loop(old_loop)


# @pytest.mark.parametrize("path", mt_paths)
# def test_backward_compatability_mt(path, all_values_matrix_table_fixture):
#     assert len(mt_paths) == 56, str((resource_dir, ht_paths))

#     old = hl.read_matrix_table(path)

#     current = all_values_matrix_table_fixture
#     current = current.select_globals(*old.globals)
#     current = current.select_rows(*old.row_value)
#     current = current.select_cols(*old.col_value)
#     current = current.select_entries(*old.entry)

#     assert current._same(old)


# @pytest.mark.parametrize("path", ht_paths)
# def test_backward_compatability_ht(path, all_values_table_fixture):
#     assert len(ht_paths) == 52, str((resource_dir, ht_paths))

#     old = hl.read_table(path)

#     current = all_values_table_fixture
#     current = current.select_globals(*old.globals)
#     current = current.select(*old.row_value)

#     assert current._same(old)
