from setuptools import setup, find_packages

setup(
    name = 'batch',
    version = '0.0.1',
    url = 'https://github.com/hail-is/hail.git',
    author = 'Hail Team',
    author_email = 'hail@broadinstitute.org',
    description = 'Kubernetes job manager client',
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
        'uvloop>=0.12',
        'pymysql',
        'google-cloud-storage==1.14.0'
    ],
)
