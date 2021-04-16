#!/usr/bin/env python

import os
import re
import shutil
import subprocess

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
# setuptools must be imported before distutils
from distutils.sysconfig import get_config_var

with open('hail/hail_pip_version') as f:
    hail_pip_version = f.read().strip()

with open("README.md", "r") as fh:
    long_description = fh.read()

dependencies = []
with open('requirements.txt', 'r') as f:
    for line in f:
        stripped = line.strip()
        if stripped.startswith('#') or len(stripped) == 0:
            continue

        pkg = stripped

        if pkg.startswith('pyspark') and os.path.exists('../env/SPARK_VERSION'):
            with open('../env/SPARK_VERSION', 'r') as file:
                spark_version = file.read()
            dependencies.append(f'pyspark=={spark_version}')
        else:
            dependencies.append(pkg)

# machinery for building libhail, mostly wrapping cmake
lib_suffix = get_config_var('EXT_SUFFIX')
# check for ninja
if shutil.which('ninja') is not None:
    CMAKE_GENERATOR = 'Ninja'
else:
    CMAKE_GENERATOR = 'Unix Makefiles'


class CMakeExtension(Extension):
    def __init__(self, name, cmake_lists_dir='.', **kwargs):
        Extension.__init__(self, name, sources=[], **kwargs)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class cmake_build_ext(build_ext):
    def build_extensions(self):
        # find cmake
        if shutil.which('cmake') is None:
            raise RuntimeError('Cannot find CMake')

        assert all(isinstance(ext, CMakeExtension) for ext in self.extensions), \
               'can only build cmake extensions'

        cmake_args = [
            '-G', CMAKE_GENERATOR,
            f'-DHAIL_PYTHON_MODULE_LIBDIR={os.path.abspath(self.build_lib)}',
            '-DCMAKE_BUILD_TYPE=RelWithDebInfo']
        for ext in self.extensions:
            tmp_dir = os.path.join(self.build_temp, ext.name)
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)
                print('creating', tmp_dir)
            subprocess.check_call(['cmake', ext.cmake_lists_dir] + cmake_args,
                                  cwd=tmp_dir)
            subprocess.check_call(['cmake', '--build', '.', '--verbose'],
                                  cwd=tmp_dir)
            subprocess.check_call(['cmake', '--install', '.'],
                                  cwd=tmp_dir)


setup(
    name="hail",
    version=hail_pip_version,
    author="Hail Team",
    author_email="hail@broadinstitute.org",
    description="Scalable library for exploring and analyzing genomic data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://hail.is",
    project_urls={
        'Documentation': 'https://hail.is/docs/0.2/',
        'Repository': 'https://github.com/hail-is/hail',
        'Change Log': 'https://hail.is/docs/0.2/change_log.html',
    },
    packages=find_packages('.'),
    package_dir={
        'hail': 'hail',
        'hailtop': 'hailtop'},
    package_data={
        'hail': ['hail_pip_version',
                 'hail_version',
                 'experimental/datasets.json'],
        'hail.backend': ['hail-all-spark.jar'],
        'hailtop': ['hail_version', 'py.typed'],
        'hailtop.hailctl': ['hail_version', 'deploy.yaml']},
    ext_modules=[CMakeExtension('_hail', cmake_lists_dir='_hail')],
    cmdclass=dict(build_ext=cmake_build_ext),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.6",
    install_requires=dependencies,
    entry_points={
        'console_scripts': ['hailctl = hailtop.hailctl.__main__:main']
    },
    setup_requires=["pytest-runner", "wheel"],
    tests_require=["pytest"],
    include_package_data=True,
)
