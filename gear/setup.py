from setuptools import setup, find_packages

setup(
    name="gear",
    version='0.0.1',
    author="Hail Team",
    author_email="hail@broadinstitute.org",
    description="Hail gear (utilities) for building microservices",
    url="https://hail.is",
    packages=find_packages(),
    python_requires=">=3.6"
)
