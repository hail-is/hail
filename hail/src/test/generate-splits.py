import os
import random
import subprocess as sp
import jinja2
from hailtop.utils import partition

sp.run("find -E src/test  -regex '.*(scala|java)' -type file | "
       "sed -e -e 's:.scala$::' -e 's:/:.:g' > classes ",
       shell=True,
       check=True)

splits = int(os.environ['TESTNG_SPLITS'])
split_index = int(os.environ['TESTNG_SPLIT_INDEX'])

assert split_index < splits

with open('classes', 'r') as f:
    classes = [x.rstrip('\n') for x in f.readlines()]

random.shuffle(classes, lambda: 0.0)

split = list(partition(splits, classes))[split_index]

with open('testng-splits.xml', 'r') as f:
    template = jinja2.Template(f.read(), undefined=jinja2.StrictUndefined, trim_blocks=True, lstrip_blocks=True)

with open('testng-splits.xml.out', 'w') as f:
    f.write(template.render(split_index=split_index, names=split))
