import pytest
import os
from typing import List
from pathlib import Path

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


resource_dir = resource('backward_compatability')
def add_paths(dirname):
    file_paths: List[str] = []
    with os.scandir(resource_dir) as versions:
        for version_dir in versions:
            try:
                with os.scandir(Path(resource_dir, version_dir, dirname)) as old_files:
                    for file in old_files:
                        file_paths.append(file.path)
            except FileNotFoundError:
                pass
    return file_paths

ht_paths = add_paths('table')
mt_paths = add_paths('matrix_table')


@pytest.mark.parametrize("path", mt_paths)
def test_backward_compatability_mt(path, all_values_matrix_table_fixture):
    assert len(mt_paths) == 56, str((resource_dir, mt_paths))

    old = hl.read_matrix_table(path)

    current = all_values_matrix_table_fixture
    current = current.select_globals(*old.globals)
    current = current.select_rows(*old.row_value)
    current = current.select_cols(*old.col_value)
    current = current.select_entries(*old.entry)

    assert current._same(old)


@pytest.mark.parametrize("path", ht_paths)
def test_backward_compatability_ht(path, all_values_table_fixture):
    assert len(ht_paths) == 52, str((resource_dir, ht_paths))

    old = hl.read_table(path)

    current = all_values_table_fixture
    current = current.select_globals(*old.globals)
    current = current.select(*old.row_value)

    assert current._same(old)
