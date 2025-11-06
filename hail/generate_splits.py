import os
import random
from pathlib import Path
from typing import List, TypeVar

T = TypeVar("T")


def partition(k: int, ls: List[T]) -> List[List[T]]:
    assert k > 0
    assert ls

    n = len(ls)
    parts = [(n - i + k - 1) // k for i in range(k)]
    assert sum(parts) == n
    assert max(parts) - min(parts) <= 1

    out = []
    start = 0
    for part in parts:
        out.append(ls[start : start + part])
        start += part
    return out


test_src_root = Path('hail/test/src')
services_root = test_src_root / 'is/hail/services'
fs_root = test_src_root / 'is/hail/io/fs'
classes = [
    str(Path(dirpath, file).relative_to(test_src_root).with_suffix('')).replace('/', '.')
    for dirpath, _dirnames, filenames in map(lambda x: (Path(x[0]), *x[1:]), os.walk(test_src_root))
    for file in filenames
    if not dirpath.is_relative_to(services_root)
    and not dirpath.is_relative_to(fs_root)
    and (file.endswith('.java') or file.endswith('.scala'))
]

random.shuffle(classes)

n_splits = int(os.environ['TESTNG_SPLITS'])
splits = partition(n_splits, classes)

for split_index, split in enumerate(splits):
    classes = '\n'.join(f'<class name="{name}"/>' for name in split)
    with open(f'testng-splits-{split_index}.xml', 'w', encoding='utf-8') as f:
        xml = f"""<!DOCTYPE suite SYSTEM "https://testng.org/testng-1.1.dtd">
<suite name="SuiteAll" allow-return-values="true" verbose="1">
    <test name="Split{split_index}">
      <classes>
        {classes}
      </classes>
    </test>
</suite>
"""
        f.write(xml)
