from setuptools import setup, find_packages

setup(
    name = 'batch',
    version = '0.0.1',
    url = 'https://github.com/hail-is/batch.git',
    author = 'Hail Team',
    author_email = 'hail@broadinstitute.org',
    description = 'Job manager for k8s',
    packages = find_packages(),
    include_package_data=True,
    install_requires=[
        'cerberus',
        'kubernetes',
        'flask',
        'requests',
        'aiohttp',
        'aiodns',
        'cchardet',
        'aiohttp_jinja2',
        'jinja2',
        'uvloop>=0.12'
    ],
)
