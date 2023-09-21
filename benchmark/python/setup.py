#!/usr/bin/env python
import os

from setuptools import setup, find_packages

setup(
    name="benchmark_hail",
    version=os.environ['HAIL_BENCHMARK_VERSION'],
    author="Hail Team",
    author_email="hail-team@broadinstitute.org",
    description="Hail benchmarking library.",
    url="https://hail.is",
    packages=find_packages("."),
    package_dir={'benchmark_hail': 'benchmark_hail'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.9",
    install_requires=[
        'hail>=0.2',
    ],
    entry_points={'console_scripts': ['hail-bench = benchmark_hail.__main__:main']},
)
