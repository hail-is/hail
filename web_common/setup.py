from setuptools import find_packages, setup

setup(
    name='web_common',
    version='0.0.1',
    url='https://github.com/hail-is/hail.git',
    author='Hail Team',
    author_email='hail@broadinstitute.org',
    description='Shared web support',
    package_data={"web_common": ["py.typed"]},
    packages=find_packages(),
    include_package_data=True,
)
