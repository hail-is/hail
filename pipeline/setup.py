from setuptools import setup, find_packages

setup(
    name = 'pipeline',
    version = '0.0.1',
    url = 'https://github.com/hail-is/hail/tree/master/pipeline',
    author = 'Hail Team',
    author_email = 'hail@broadinstitute.org',
    description = 'Pipeline builder',
    packages = find_packages(),
    install_requires=[
        'nest_asyncio'
    ],
)
