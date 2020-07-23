import os
import random
import subprocess as sp
import jinja2


def partition(k, ls):
    assert k > 0
    assert ls

    n = len(ls)
    parts = [(n - i + k - 1) // k for i in range(k)]
    assert sum(parts) == n
    assert max(parts) - min(parts) <= 1

    out = []
    start = 0
    for part in parts:
        out.append(ls[start:start + part])
        start += part
    return out


sp.run("find src/test \\( -name \"*.scala\" -o -name \"*.java\" \\) -type f > classes",
       shell=True,
       check=True)

n_splits = int(os.environ['TESTNG_SPLITS'])

with open('classes', 'r') as f:
    foo = f.readlines()
    classes = [x.replace('src/test/scala/', '')
                .replace('.scala\n', '')
                .replace('.java\n', '')
                .replace('/', '.')
               for x in foo]
    classes = [cls for cls in classes if not cls.startswith('is.hail.services')]

random.shuffle(classes, lambda: 0.0)

splits = partition(n_splits, classes)

with open('testng-splits.xml', 'r') as f:
    template = jinja2.Template(f.read(), undefined=jinja2.StrictUndefined, trim_blocks=True, lstrip_blocks=True)

for split_index, split in enumerate(splits):
    with open(f'testng-splits-{split_index}.xml', 'w') as f:
        f.write(template.render(split_index=split_index, names=split))
