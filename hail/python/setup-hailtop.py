#!/usr/bin/env python3

import sys
from importlib.util import module_from_spec, spec_from_file_location

from setuptools import find_packages, setup


def load_module(name, path):
    spec = spec_from_file_location(name, path)
    mod = module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


setup(
    name='hailtop',
    version=load_module('version', 'hailtop/version.py').__pip_version__,
    author="Hail Team",
    author_email="hail@broadinstitute.org",
    description="Top level Hail module.",
    url="https://hail.is",
    project_urls={
        'Documentation': 'https://hail.is/docs/0.2/',
        'Repository': 'https://github.com/hail-is/hail',
    },
    packages=find_packages('.'),
    package_dir={'hailtop': 'hailtop'},
    package_data={"hailtop": ["py.typed"], 'hailtop.hailctl': ['deploy.yaml']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.10",
    entry_points={'console_scripts': ['hailctl = hailtop.hailctl.__main__:main']},
    setup_requires=["pytest-runner", "wheel"],
    include_package_data=True,
)
