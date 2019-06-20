from setuptools import setup, find_packages

setup(
    name = 'batch_client',
    version = '0.0.1',
    url = 'https://github.com/hail-is/hail.git',
    author = 'Hail Team',
    author_email = 'hail@broadinstitute.org',
    description = 'Job manager for kubernetes',
    packages = find_packages()
)
