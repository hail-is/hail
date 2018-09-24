from setuptools import setup, find_packages

setup(
    name='hail-ci',
    version='0.0.1',
    url='https://github.com/hail-is/ci',
    author='Hail Team',
    author_email='hail@broadinstitute.org',
    description='Description of my package',
    packages=find_packages(),
    install_requires=['requests',
                      'flask'],
)
