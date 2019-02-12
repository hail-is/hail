import sys
import numpy
from random import shuffle

# python split.py N_APPLICATIONS N_REVIEWERS

apps = range(1, int(sys.argv[1]) + 1)
shuffle(apps)
for line in numpy.array_split(apps, int(sys.argv[2])):
    print(' '.join([str(x) + '.pdf' for x in line]))
