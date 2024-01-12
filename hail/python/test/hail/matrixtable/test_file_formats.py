import asyncio
from typing import List, Tuple

import pytest

import hail as hl
from hail.utils.java import Env, scala_object

from ..helpers import (
    create_all_values_datasets,
    create_all_values_matrix_table,
    create_all_values_table,
    resource,
    resource_dir,
)


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
        all_values_matrix_table.write(
            os.path.join(matrix_table_dir, f'{i}.hmt'), overwrite=True, _codec_spec=codec.toString()
        )
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


async def collect_paths() -> Tuple[List[str], List[str]]:
    resource_dir = resource('backward_compatability/')
    from hailtop.aiotools.router_fs import RouterAsyncFS

    fs = RouterAsyncFS()

    async def contents_if_present(url: str):
        try:
            return await fs.listfiles(url)
        except FileNotFoundError:

            async def empty():
                if False:
                    yield

            return empty()

    try:
        versions = [await x.url() async for x in await fs.listfiles(resource_dir)]
        ht_paths = [await x.url() for version in versions async for x in await contents_if_present(version + 'table/')]
        mt_paths = [
            await x.url() for version in versions async for x in await contents_if_present(version + 'matrix_table/')
        ]
        return ht_paths, mt_paths
    finally:
        await fs.close()


# pytest sometimes uses background threads, named "Dummy-1", to collect tests. Asyncio dislikes
# automatically creating event loops in these threads, so we just explicitly create one.
ht_paths, mt_paths = asyncio.new_event_loop().run_until_complete(collect_paths())


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
