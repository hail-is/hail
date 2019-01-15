#!/usr/bin/env python

from glob import glob
from os.path import basename
from os.path import splitext
from setuptools import setup, find_packages

with open('src/hail/hail_pip_version') as f:
    hail_pip_version = f.read().strip()

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("spark_version", "r") as f:
    spark_version = f.read()

setup(
    name="hail",
    version=hail_pip_version,
    author="Hail Team",
    author_email="hail-team@broadinstitute.org",
    description="Scalable library for exploring and analyzing genomic data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://hail.is",
    packages=find_packages('src'),
    package_dir={'': 'src'},
    package_data={
        '': ['hail-all-spark.jar',
             'hail_pip_version',
             'hail_version',
             *[x[9:] for x in glob('src/hail/docs/**/*', recursive=True)]]},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.6",
    install_requires=[
        'numpy<2',
        'pandas>0.22,<0.24',
        'matplotlib<3',
        'seaborn<0.9',
        'bokeh<0.14',
        'pyspark=='+spark_version,
        'parsimonious<0.9',
        'ipykernel<5',
        'decorator<5',
        'requests>=2.21.0,<2.21.1',
    ]
)
