import os
import sys

wheel_size = os.path.getsize(os.environ["WHEEL"])
pypi_used_storage = int(os.environ["PYPI_USED_STORAGE"])

print(f'Wheel size: {wheel_size / 1024 / 1024:.0f}MiB. PyPI used storage: {pypi_used_storage / 1024 / 1024:.0f}MiB.')

if wheel_size + pypi_used_storage > 10 * 1024 * 1024 * 1024:
    print('Insufficient space available at PyPI.')
    sys.exit(1)
print('Sufficient space is available at PyPI.')
