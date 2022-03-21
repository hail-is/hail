import pytest
import os.path
from hailtop.aiotools.copy import copy_from_dict
import tempfile


def write_file(path, data):
    with open(path, "w") as f:
        f.write(data)


def read_file(path):
    with open(path, "r") as f:
        return f.read()


@pytest.mark.asyncio
async def test_copy_file():
    with tempfile.TemporaryDirectory() as test_dir:
        write_file(f"{test_dir}/file1", "hello world\n")

        inputs = [{"from": f"{test_dir}/file1", "to": f"{test_dir}/file2"},
                  {"from": f"{test_dir}/file1", "into": f"{test_dir}/dir1"}]

        await copy_from_dict(files=inputs)

        expected_files = [f"{test_dir}/file1", f"{test_dir}/file2", f"{test_dir}/dir1/file1"]
        for file in expected_files:
            assert read_file(file) == "hello world\n"


@pytest.mark.asyncio
async def test_copy_dir():
    with tempfile.TemporaryDirectory() as test_dir:
        os.makedirs(f"{test_dir}/subdir1")
        write_file(f"{test_dir}/subdir1/file1", "hello world\n")

        inputs = [{"from": f"{test_dir}/subdir1", "into": f"{test_dir}/subdir2"}]

        await copy_from_dict(files=inputs)
        assert read_file(f"{test_dir}/subdir2/subdir1/file1") == "hello world\n"


@pytest.mark.asyncio
async def test_error_function():
    with tempfile.TemporaryDirectory() as test_dir:
        write_file(f"{test_dir}/foo", "hello world\n")
        write_file(f"{test_dir}/bar", "hello world\n")

        inputs = [{"from": f"{test_dir}/bar", "into": f"{test_dir}/foo"}]

        try:
            await copy_from_dict(files=inputs)
        except NotADirectoryError:
            pass
        else:
            assert False
