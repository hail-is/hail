import sys
from random import shuffle

import numpy

# python split.py N_APPLICATIONS N_REVIEWERS

x = range(int(sys.argv[1]))
shuffle(x)
print(numpy.array_split(x, int(sys.argv[2])))
