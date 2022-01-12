from os import path
import pytest
import asyncio
import os.path
from hailtop.aiotools.copy import  copy_test
import tempfile





async def opening_files(path, data):
    # path = specific file path 
    path = f'{test_dir}/file1'
    f = open(path, 'w')
    try:
         x = str(data)
         data = f.write(x)
    finally:
        f.close()
        



# TEMP DIRECTORY FOR FILE1 TEST
@pytest.mark.asyncio
async def test_copy_file():
    with tempfile.TemporaryDirectory() as test_dir:
        with open(f'{test_dir}/file1', 'w') as f:
            f.write('hello world\n')

        res = await copy_test( 
        None, [{"from": f"{test_dir}/file1", "to":f"{test_dir}/file2"}, 
        {"from": f"{test_dir}/file1", "into": f"{test_dir}/dir1"} ]
        )

        files = [f'{test_dir}/file1', f'{test_dir}/file2', f'{test_dir}/dir1/file1' ]
        for file in files :
            file_exist = os.path.exists(file)
            print(file_exist)
            assert file_exist


# TEMP DIRECTORY FOR SUB DIRECTORY TEST
@pytest.mark.asyncio
async def test_copy_dir():
    with tempfile.TemporaryDirectory() as test_dir:
        os.makedirs(f'{test_dir}/subdir1')
        
        with open(f'{test_dir}/subdir1/file1', 'w') as d:
            d.write('hello world\n')
        
        with open(f'{test_dir}/subdir1/file2', 'w') as f:
            f.write('hello world\n')

        res = await copy_test( 
        None,[{"from": f"{test_dir}/subdir1", "into": f"{test_dir}/subdir2"},
        {"from": f"{test_dir}/subdir1/file1", "into": f"{test_dir}/subdir2"}]
        )

        files = [  f'{test_dir}/subdir2/file1', f'{test_dir}/subdir2/subdir1/file1']
        for file in files :
            file_exist = os.path.exists(file)
            assert file_exist

# Find how in python to check whether a file exsist 
# something like os.path.exsist dir1/file1 also test file2.exsist, file3.exsist and dir1/file1
#This function gives a boolean and I should assert whether this call is true
#test_subdirectories figure out how to write file1 
#test for a directory in a subdirectory 


# Find how in python to check whether a file exsist 
# something like os.path.exsist dir1/file1 also test file2.exsist, file3.exsist and dir1/file1
#This function gives a boolean and I should assert whether this call is true
#test_subdirectories figure out how to write file1 
#test for a directory in a subdirectory 

# Make sure this is in a new branch. Copy.py and add test into copy.py
# Save existing changes with git 
#clean up the file, formatting, get rid of comments, and unecessary test 
# Save changes with new commit 
# write a function that just writes to a file with two arguments (path, data) that can then be used in the other functions