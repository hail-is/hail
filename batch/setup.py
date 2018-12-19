from setuptools import setup, find_packages

setup(
    name = 'batch',
    version = '0.0.1',
    url = 'https://github.com/hail-is/batch.git',
    author = 'Hail Team',
    author_email = 'hail@broadinstitute.org',
    description = 'Job manager for k8s',
    packages = find_packages(),
    install_requires=[
        'cerberus',
        'kubernetes',
        'flask',
    ],
)
