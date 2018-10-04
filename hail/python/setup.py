#!/usr/bin/env python

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

# https://stackoverflow.com/questions/45150304/how-to-force-a-python-wheel-to-be-platform-specific-when-building-it
try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False
except ImportError:
    bdist_wheel = None

with open('hail/version', 'r') as f:
    version = f.read().strip()

setup(
    name="hail",
    version=version,
    author="Hail Team",
    author_email="hail-team@broadinstitute.org",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://hail.is",
    packages=find_packages(),
    package_data={'': ['version', 'hail-all-spark.jar']},
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
        'pyspark>=2.2,<2.3',
        'parsimonious<0.9',
        'ipykernel<5',
        'decorator<5',
    ],
    cmdclass={'bdist_wheel': bdist_wheel},
)
