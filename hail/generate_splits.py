import os
import random
import subprocess as sp
import jinja2
from hailtop.utils import partition

sp.run("find src/test \\( -name \"*.scala\" -o -name \"*.java\" \\) -type f > classes",
       shell=True,
       check=True)

n_splits = int(os.environ['TESTNG_SPLITS'])

with open('classes', 'r') as f:
    classes = [x.rstrip('.scala\n').rstrip('.java\n').lstrip('src/test/').replace('/', '.')
               for x in f.readlines()]

random.shuffle(classes, lambda: 0.0)

splits = list(partition(n_splits, classes))

with open('testng-splits.xml', 'r') as f:
    template = jinja2.Template(f.read(), undefined=jinja2.StrictUndefined, trim_blocks=True, lstrip_blocks=True)

for split, split_index in enumerate(splits):
    with open(f'testng-splits-{split_index}.xml', 'w') as f:
        f.write(template.render(split_index=split_index, names=split))
