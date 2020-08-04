from setuptools import setup, find_packages

setup(
    name='address',
    version='0.0.1',
    url='https://github.com/hail-is/hail.git',
    author='Hail Team',
    author_email='hail@broadinstitute.org',
    description='Convert names to IP addresses',
    packages=find_packages(),
    include_package_data=True
)
