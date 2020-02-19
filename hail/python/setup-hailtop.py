from setuptools import setup, find_packages

setup(
    name = 'hailtop',
    version = '0.0.1',
    url = 'https://github.com/hail-is/hail.git',
    author = 'Hail Team',
    author_email = 'hail@broadinstitute.org',
    description = 'Toplevel Hail module',
    packages = find_packages(),
    entry_points={
        'console_scripts': ['hailctl = hailtop.hailctl.__main__:main']
    }
)
