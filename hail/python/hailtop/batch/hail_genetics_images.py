import sys

from hailtop import pip_version


HAIL_GENETICS = 'hailgenetics/'
HAIL_GENETICS_IMAGES = [
    HAIL_GENETICS + name for name in ('hail', 'hailtop', 'genetics', 'python-dill', 'vep-grch37-85', 'vep-grch38-95')
]


def hailgenetics_hail_image_for_current_python_version():
    version = sys.version_info
    if version.major != 3 or version.minor not in (9, 10, 11):
        raise ValueError(
            f'You must specify an "image" for Python jobs and the BatchPoolExecutor if you are '
            f'using a Python version other than 3.9, 3.10, or 3.11 (you are using {version})'
        )
    if version.minor == 9:
        return f'hailgenetics/hail:{pip_version()}'
    return f'hailgenetics/hail:{pip_version()}-py{version.major}.{version.minor}'
