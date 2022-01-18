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


# 1. What should we call a test function so that pytest will run? It needs to start with test_. 
# The suffix of test_ should be something descriptive of what the test actually is doing. So dangerous_function is not okay.
# 2. The try...finally you have right now is outside of the test function. It's going to run that code, but dangerous_function() is an async function.
# You have to await an async function. But that code should be inside your test_... function anyways and not outside of it. 
# 3. Lastly, the dangerous function we want to run is just await copy_from_dict(files=inputs). 
# You shouldn't need to assert the files exist because they shouldn't exist.
# Instead you want to show that you appropriately get a NotADirectoryError raised as an exception. 
# You can read a little bit about exceptions if you don't know what they are.

@pytest.mark.asyncio
async def test_error_function():
    with tempfile.TemporaryDirectory() as test_dir:
        # write_file(f"{test_dir}/foo", "hello world\n")
        write_file(f"{test_dir}/bar", "hello world\n")
           
        inputs = [{"from": f"{test_dir}/bar", "into": f"{test_dir}/foo"}]

    try:
    # ... try to do the thing that causes an error here ...
         await copy_from_dict(files=inputs)
    except NotADirectoryError:
        pass
    else:
        assert False  # we come to "else:" if there was no error