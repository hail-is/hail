from os import path, read
import pytest
import asyncio
import os.path
from hailtop.aiotools.copy import  copy_test
import tempfile



def write_file(path, data):
    with open(path, 'w') as f:
         f.write(data)

def read_file(path):
    with open(path, 'r') as f:
       return f.read()

# TEMP DIRECTORY FOR FILE1 TEST
@pytest.mark.asyncio
async def test_copy_file():
    with tempfile.TemporaryDirectory() as test_dir:
        write_file(f'{test_dir}/file1', 'hello world\n')

        res = await copy_test( 
        None,[{"from": f"{test_dir}/file1", "to":f"{test_dir}/file2"},
        {"from": f"{test_dir}/file1", "into": f"{test_dir}/dir1"}]
        )

        files = [f'{test_dir}/file1', f'{test_dir}/file2', f'{test_dir}/dir1/file1']
        for file in files :
            assert read_file(file)  == 'hello world\n'


# TEMP DIRECTORY FOR SUB DIRECTORY TEST
@pytest.mark.asyncio
async def test_copy_dir():
    with tempfile.TemporaryDirectory() as test_dir:
        os.makedirs(f'{test_dir}/subdir1')
        write_file(f'{test_dir}/subdir1/file1', 'hello world\n')

        res = await copy_test( 
        None,[{"from": f"{test_dir}/subdir1/file1", "into": f"{test_dir}/subdir2"}]
        )
        files = [ f'{test_dir}/subdir2/file1', f'{test_dir}/subdir1/file1']
        for file in files :
            assert read_file(file)  == 'hello world\n'

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

# Add a read function like we did for the write file
# use that function in the asserts 
# Reviewing code and check formatting (spacing )look up tools for formatting after 
# saving all the changes
# import copy and make_transfer