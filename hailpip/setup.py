#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="hailpip",
    version='0.0.1',
    author="Hail Team",
    author_email="hail-team@broadinstitute.org",
    description="Better pip.",
    url="https://hail.is",
    project_urls={
        'Documentation': 'https://hail.is/docs/0.2/',
        'Repository': 'https://github.com/hail-is/hail',
    },
    packages=find_packages('.'),
    package_dir={
        'hailpip': 'hailpip'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.6",
    install_requires=[
        'pip'
    ],
    entry_points={
        'console_scripts': ['hailpip = hailpip.__main__:main']
    }
)
