#!/usr/bin/env python

from setuptools import setup, find_packages

with open('src/hail/hail_pip_version') as f:
    hail_pip_version = f.read().strip()

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="hail",
    version=hail_pip_version,
    author="Hail Team",
    author_email="hail-team@broadinstitute.org",
    description="Scalable library for exploring and analyzing genomic data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://hail.is",
    packages=find_packages('./src'),
    package_dir={
        'hail': 'src/hail',
        'hailctl': 'src/hailctl'},
    package_data={
        'hail': ['hail-all-spark.jar', 'hail_pip_version', 'hail_version'],
        'hailctl': ['deploy.yaml', 'hail_pip_version', 'hail_version']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.6",
    install_requires=[
        'numpy<2',
        'pandas>0.22,<0.24',
        'bokeh>1.1,<1.3',
        'pyspark>=2.4,<2.4.2',
        'parsimonious<0.9',
        'ipykernel<5',
        'decorator<5',
        'requests>=2.21.0,<2.21.1',
        'gcsfs==0.2.1',
        'hurry.filesize==0.9',
        'scipy>1.2, <1.4'
    ],
    entry_points={
        'console_scripts': ['hailctl = hailctl.__main__:main']
    },
)
