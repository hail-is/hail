#!/usr/bin/env python

from setuptools import setup, find_packages

with open('hail/hail_pip_version') as f:
    hail_pip_version = f.read().strip()

with open("README.md", "r") as fh:
    long_description = fh.read()

dependencies = []
with open('requirements.txt', 'r') as f:
    for line in f:
        dependencies.append(line.strip())

setup(
    name="hail",
    version=hail_pip_version,
    author="Hail Team",
    author_email="hail-team@broadinstitute.org",
    description="Scalable library for exploring and analyzing genomic data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://hail.is",
    packages=find_packages('.'),
    package_dir={
        'hail': 'hail',
        'hailtop': 'hailtop'},
    package_data={
        'hail': ['hail_pip_version',
                 'hail_version',
                 'experimental/annotation_db.json'],
        'hail.backend': ['hail-all-spark.jar'],
        'hailtop.hailctl': ['hail_version', 'deploy.yaml']},
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
    tests_require=["pytest"]
)
