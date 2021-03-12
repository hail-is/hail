#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='hailtop',
    version="0.0.1",
    author="Hail Team",
    author_email="hail@broadinstitute.org",
    description="Top level Hail module.",
    url="https://hail.is",
    project_urls={
        'Documentation': 'https://hail.is/docs/0.2/',
        'Repository': 'https://github.com/hail-is/hail',
    },
    packages=find_packages('.'),
    package_dir={
        'hailtop': 'hailtop'},
    package_data={
        "hailtop": ["py.typed", "hail_version"],
        'hailtop.hailctl': ['hail_version', 'deploy.yaml']
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.6",
    entry_points={
        'console_scripts': ['hailctl = hailtop.hailctl.__main__:main']
    },
    setup_requires=["pytest-runner", "wheel"],
    include_package_data=True,
)
