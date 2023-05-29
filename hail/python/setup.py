#!/usr/bin/env python

import os
import re
from setuptools import setup, find_packages

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

def add_dependencies(fname):
    with open(fname, 'r') as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith('#') or len(stripped) == 0:
                continue
            if stripped.startswith('-c'):
                continue
            if stripped.startswith('-r'):
                additional_requirements = stripped[len('-r'):].strip()
                add_dependencies(additional_requirements)
                continue
            pkg = stripped
            if pkg.startswith('pyspark') and os.path.exists('../env/SPARK_VERSION'):
                with open('../env/SPARK_VERSION', 'r') as file:
                    spark_version = file.read()
                [major, minor, patch] = spark_version.split('.')
                dependencies.append(f'pyspark>={major}.{minor},<{int(major)+1}')
            else:
                dependencies.append(pkg)
add_dependencies('requirements.txt')

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
                 'hail_revision',
                 'experimental/datasets.json'],
        'hail.backend': ['hail-all-spark.jar'],
        'hailtop': ['hail_version', 'py.typed'],
        'hailtop.hailctl': ['hail_version', 'deploy.yaml']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.8",
    install_requires=dependencies,
    entry_points={
        'console_scripts': ['hailctl = hailtop.hailctl.__main__:main']
    },
    setup_requires=["pytest-runner", "wheel"],
    tests_require=["pytest"],
    include_package_data=True,
)
